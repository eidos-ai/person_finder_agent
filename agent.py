#!/usr/bin/env python3
"""
Person Research Agent using Google ADK.
Receives a name and surname, uses Google Search to find profile information.
Returns its findings along with per-model token consumption.
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum

from loguru import logger

# Set up service account credentials before any google imports
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./gcp-sa Manuel_Correa JS.json"
os.environ["GOOGLE_CLOUD_PROJECT"] = "dev-sandbox-359102"
os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"

from google.adk.agents import Agent
from google.adk.runners import Runner, RunConfig
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.adk.tools.mcp_tool import McpToolset, StreamableHTTPConnectionParams
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig

from skills.fetch_google import (
    fetch_page_content as _fetch_page_content,
)
from skills.fetch_linkedin import (
    search_profiles as _linkedin_search_profiles,
    get_full_linkedin_profile as _get_full_linkedin_profile,
)
from token_usage import (
    TokenUsage,
    compute_cost_usd,
)


def _extract_json(raw: str):
    """Extract and parse JSON from an LLM response, handling markdown fences and minor formatting issues."""
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as first_err:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if inner and not inner.startswith("{"):
                return json.loads("[{" + inner + "}]")
        raise first_err


# ── input dataclass ───────────────────────────────────────

@dataclass
class PersonInitialDescription:
    first_name: str
    last_name: str
    city: str
    state: str
    school_name: str
    school_department: str
    graduation_date: str


@dataclass
class PersonProfile:
    has_job: bool
    job_type: str  # RelatedJob | StillLooking | MilitaryService | ContinuingEducation | UnrelatedJob | Other
    job_related: bool
    job_schedule: str  # "FullTime" | "PartTime" | "Unknown"
    self_employed: bool
    employer_name: str
    employer_city: str
    employer_state: str
    continuing_education: bool
    current_education_institution: str
    current_education_field: str
    sources: list[str]  # URLs where this information was sourced from


@dataclass
class ModelUsage:
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cached_tokens: int
    cost_usd: float


@dataclass
class Consumption:
    tool_calls_used: int
    max_llm_calls: int
    models: dict[str, ModelUsage]
    coresignal_calls: dict[str, int]  # tool_name -> call count


class Confidence(Enum):
    LOW = "LOW"
    MID = "MID"
    HIGH = "HIGH"


@dataclass
class AgentResult:
    found: bool
    confidence: Confidence
    profile: PersonProfile | None
    response: str
    consumption: Consumption


# ── configuration ──────────────────────────────────────────
APP_NAME = "person_research_agent"
USER_ID = "user1"
SESSION_ID = "session1"
MAX_LLM_CALLS = 10
AGENT_MODEL = "gemini-2.5-flash"

CORESIGNAL_MCP_URL = "https://mcp.coresignal.com/mcp"
# Prefer env var so secrets aren't committed to source control.
CORESIGNAL_API_KEY = os.environ.get("CORESIGNAL_API_KEY")

CORESIGNAL_AGENT_INSTRUCTION = """\
You research recent college graduates to find their current status (employed, studying, military, or job-seeking).

You receive: first name, last name, city, state, school name, department, graduation date.
These are typically 18-22 year old community college graduates.

## Search Strategy (be efficient — minimize tool calls)

### Step 1 — Coresignal (try first, strict call caps)
Hard limits across this person lookup:
- At most 3 profile search calls on Coresignal
- At most 7 profile get/detail calls on Coresignal
Never exceed these limits.
Definitions:
- A "search" is a query that returns multiple candidate profiles for the query.
- A "get" is a request for one specific profile by id that returns detailed profile fields.

Use coresignal_employee_multisource_api with:
- query: Elasticsearch Query DSL dict
- keys: ["id", "full_name", "headline", "location_full", "linkedin_url", "is_working", "active_experience_title"]
- limit: 3

First search: name + school nested query:
  {"bool": {"must": [{"match": {"first_name": "John"}}, {"match": {"last_name": "Doe"}}, {"nested": {"path": "education", "query": {"match": {"education.institution_name": "UCLA"}}}}]}}

If no results, try broader queries (e.g., name-only), but stay within the 5-search cap.

If you find a plausible match and need details, use get/detail calls (e.g., keys ["experience", "education"] filtered by id), but stay within the 10-get cap.

### Step 2 — Google Search (fallback if Coresignal fails)
Use google_search with full name + school. Use fetch_web_page only on the most promising result.

### Step 3 — Output
Return ONLY a raw JSON object (no markdown fences, no extra text):
{
  "found": bool, "confidence": "HIGH"|"MID"|"LOW",
  "has_job": bool, "job_type": "RelatedJob"|"StillLooking"|"MilitaryService"|"ContinuingEducation"|"UnrelatedJob"|"Other",
  "job_related": bool, "job_schedule": "FullTime"|"PartTime"|"Unknown",
  "self_employed": bool, "employer_name": "", "employer_city": "", "employer_state": "",
  "continuing_education": bool, "current_education_institution": "", "current_education_field": "",
  "sources": []
}

## Identity Verification
Reject matches with 5+ years of experience (not a recent grad). Check location/school consistency.
When in doubt, return best guess with LOW/MID confidence rather than found=false.

## job_type values
- RelatedJob: employed in field related to their studies
- UnrelatedJob: employed but unrelated to studies
- StillLooking: no current position, seeking work
- MilitaryService: serving in US military
- ContinuingEducation: transferred to university or enrolled in further education
- Other: none of the above
"""

LOCAL_AGENT_INSTRUCTION = """\
You research recent college graduates to find their current status (employed, studying, military, or job-seeking).

You receive: first name, last name, city, state, school name, department, graduation date.
These are typically 18-22 year old community college graduates.

## Search Strategy (be efficient — minimize tool calls)

### Step 1 — Local LinkedIn (try first, strict call caps)
Hard limits across this person lookup:
- At most 3 LinkedIn search calls (using `linkedin_search_profiles`)
- At most 7 LinkedIn profile detail calls (using `linkedin_get_full_profile`)

Use `linkedin_search_profiles` with keywords derived from:
  - The provided first name, last name, and school name
  - If needed, broaden to first name + last name (still within the call cap)

If you find a plausible match, use `linkedin_get_full_profile(profile_url)` to retrieve profile markdown text.

### Step 2 — Google Search (fallback if local LinkedIn fails)
Use `google_search` with full name + school. Use `fetch_web_page` only on the most promising result.

### Step 3 — Output
Return ONLY a raw JSON object (no markdown fences, no extra text):
{
  "found": bool, "confidence": "HIGH"|"MID"|"LOW",
  "has_job": bool, "job_type": "RelatedJob"|"StillLooking"|"MilitaryService"|"ContinuingEducation"|"UnrelatedJob"|"Other",
  "job_related": bool, "job_schedule": "FullTime"|"PartTime"|"Unknown",
  "self_employed": bool, "employer_name": "", "employer_city": "", "employer_state": "",
  "continuing_education": bool, "current_education_institution": "", "current_education_field": "",
  "sources": []
}

## Identity Verification
Reject matches with 5+ years of experience (not a recent grad). Check location/school consistency.
When in doubt, return best guess with LOW/MID confidence rather than found=false.

## job_type values
- RelatedJob: employed in field related to their studies
- UnrelatedJob: employed but unrelated to studies
- StillLooking: no current position, seeking work
- MilitaryService: serving in US military
- ContinuingEducation: transferred to university or enrolled in further education
- Other: none of the above
"""

# Accumulator for token usage from LLM calls *inside* tools
_tool_token_usages: list[TokenUsage] = []
_profile_sources: list[str] = []


# ── tool functions ─────────────────────────────────────────

def fetch_web_page(url: str) -> str:
    """
    Open a web page URL with a headless browser and return its text content.
    Use this to visit any URL (company websites, news articles, etc.) and read its content.

    Args:
        url: The full URL to fetch (e.g. https://example.com/about).

    Returns:
        The text content of the page.
    """
    content = _fetch_page_content(url)
    _profile_sources.append(url)
    return content


# LinkedIn (local headless browser) tool wrappers
def linkedin_search_profiles(keywords: str) -> list[dict]:
    """Search LinkedIn people results page and return candidate profile URLs."""
    profiles, usage = _linkedin_search_profiles(keywords)
    _tool_token_usages.append(usage)

    # Return only JSON-safe primitives (ADK tool serialization is simplest here).
    return [
        {
            "name": p.name,
            "headline": p.headline,
            "location": p.location,
            "profile_url": p.profile_url,
        }
        for p in profiles
        if getattr(p, "profile_url", None)
    ]


def linkedin_get_full_profile(profile_url: str) -> str:
    """Fetch a LinkedIn profile page and convert it to markdown via Gemini."""
    full_profile, usage = _get_full_linkedin_profile(profile_url)
    _tool_token_usages.append(usage)
    _profile_sources.append(profile_url)
    return full_profile.markdown


# ── ADK tool definitions ──────────────────────────────────

google_search_tool = GoogleSearchTool(bypass_multi_tools_limit=True)
web_page_tool = FunctionTool(fetch_web_page)
linkedin_search_tool = FunctionTool(linkedin_search_profiles)
linkedin_get_full_profile_tool = FunctionTool(linkedin_get_full_profile)


# ── runner ─────────────────────────────────────────────────

async def run_agent(person: PersonInitialDescription, source: str = "coresignal") -> AgentResult:
    """Run the person research agent for the given person.

    source:
      - `coresignal`: use Coresignal MCP tools for LinkedIn profile search/details
      - `local`: use local Selenium headless browser tools from `skills/fetch_linkedin.py`
    """
    _tool_token_usages.clear()
    _profile_sources.clear()

    if source not in ("coresignal", "local"):
        raise ValueError(f"Unknown source={source}. Expected 'coresignal' or 'local'.")

    if source == "coresignal":
        if not CORESIGNAL_API_KEY:
            raise ValueError("CORESIGNAL_API_KEY is required for source='coresignal'.")

        # Create MCP toolset for Coresignal (per-run lifecycle)
        coresignal_toolset = McpToolset(
            connection_params=StreamableHTTPConnectionParams(
                url=CORESIGNAL_MCP_URL,
                headers={"apikey": CORESIGNAL_API_KEY},
                timeout=30,
            ),
        )
        tools = [coresignal_toolset, google_search_tool, web_page_tool]
        instruction = CORESIGNAL_AGENT_INSTRUCTION
    else:
        # `local` mode: no Coresignal MCP tools.
        tools = [linkedin_search_tool, linkedin_get_full_profile_tool, google_search_tool, web_page_tool]
        instruction = LOCAL_AGENT_INSTRUCTION

    agent = Agent(
        name="person_research_agent",
        model=AGENT_MODEL,
        description="Agent that researches a person's employment status.",
        instruction=instruction,
        tools=tools,
        generate_content_config=GenerateContentConfig(
            thinking_config=ThinkingConfig(include_thoughts=True),
        ),
    )

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)

    prompt = (
        f"Find information about: {person.first_name} {person.last_name}\n"
        f"Location: {person.city}, {person.state}\n"
        f"School: {person.school_name}, {person.school_department}\n"
        f"Graduation date: {person.graduation_date}\n"
        "Use all of this context to identify the correct profile among search results."
    )
    content = types.Content(
        role="user",
        parts=[types.Part(text=prompt)],
    )

    tool_call_count = 0
    coresignal_calls: dict[str, int] = {}
    final_text = ""
    all_thinking_text: list[str] = []  # collect thinking for fallback parsing

    # Agent-level token usage (orchestration LLM calls)
    agent_input_tokens = 0
    agent_output_tokens = 0
    agent_thinking_tokens = 0
    agent_cached_tokens = 0

    events = runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
        run_config=run_config,
    )

    async for event in events:
        # Count function calls and debug output
        if event.content and event.content.parts:
            for part in event.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    tool_call_count += 1
                    args = dict(fc.args) if fc.args else {}
                    logger.debug("CALL #{} {}({})", tool_call_count, fc.name, args)
                    if fc.name.startswith("coresignal_"):
                        coresignal_calls[fc.name] = coresignal_calls.get(fc.name, 0) + 1
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    resp_text = str(fr.response)
                    if len(resp_text) > 300:
                        resp_text = resp_text[:300] + "..."
                    logger.debug("RESULT {} -> {}", fr.name, resp_text)
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    logger.debug("THINKING: {}", part.text)
                    all_thinking_text.append(part.text)
                elif getattr(part, "text", None) and not event.is_final_response():
                    logger.debug("REASONING: {}", part.text)

        # Accumulate agent-level token usage
        um = event.usage_metadata
        if um is not None:
            agent_input_tokens += getattr(um, "prompt_token_count", 0) or 0
            agent_output_tokens += getattr(um, "candidates_token_count", 0) or 0
            agent_thinking_tokens += getattr(um, "thoughts_token_count", 0) or 0
            agent_cached_tokens += getattr(um, "cached_content_token_count", 0) or 0

        if event.is_final_response():
            # Collect non-thought text from the final response parts
            non_thought_parts = []
            thought_parts = []
            if event.content is None or event.content.parts is None:
                logger.warning("Final response event has no content/parts (possibly filtered or empty)")
                continue
            for part in event.content.parts:
                text = getattr(part, "text", None)
                if text:
                    if getattr(part, "thought", False):
                        thought_parts.append(text)
                    else:
                        non_thought_parts.append(text)
            # Prefer actual output text; fall back to thought text
            final_text = "\n".join(non_thought_parts) if non_thought_parts else "\n".join(thought_parts)

    # ── build consumption report ───────────────────────────
    # Per-model aggregation
    models: dict[str, ModelUsage] = {}

    # Agent orchestration usage
    agent_cost = compute_cost_usd(agent_input_tokens, agent_output_tokens, AGENT_MODEL, agent_thinking_tokens, agent_cached_tokens)
    models[AGENT_MODEL] = ModelUsage(
        input_tokens=agent_input_tokens,
        output_tokens=agent_output_tokens,
        thinking_tokens=agent_thinking_tokens,
        cached_tokens=agent_cached_tokens,
        cost_usd=agent_cost,
    )

    # Tool-internal LLM usage
    for u in _tool_token_usages:
        key = f"{u.model_name} (tools)"
        if key not in models:
            models[key] = ModelUsage(input_tokens=0, output_tokens=0, thinking_tokens=0, cached_tokens=0, cost_usd=0.0)
        models[key].input_tokens += u.input_tokens
        models[key].output_tokens += u.output_tokens
        models[key].thinking_tokens += u.thinking_tokens
        models[key].cached_tokens += u.cached_tokens
        models[key].cost_usd += u.cost_usd

    consumption = Consumption(
        tool_calls_used=tool_call_count,
        max_llm_calls=MAX_LLM_CALLS,
        models=models,
        coresignal_calls=coresignal_calls,
    )

    person_profile = None
    person_found = False
    person_confidence = Confidence.LOW
    # Try to parse JSON from the agent response.  The model sometimes emits
    # the JSON only inside its thinking and returns narrative text as the
    # "final" response.  We try multiple sources in order of preference.
    parse_candidates = [final_text] + list(reversed(all_thinking_text))
    data = None
    for candidate in parse_candidates:
        if not candidate:
            continue
        try:
            data = _extract_json(candidate)
            if isinstance(data, dict) and "job_type" in data:
                break  # found a valid profile JSON
            data = None
        except (json.JSONDecodeError, ValueError, AttributeError):
            # Also try to find a JSON object embedded in narrative text
            m = re.search(r'\{[^{}]*"job_type"\s*:.*?\}', candidate, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                    break
                except json.JSONDecodeError:
                    data = None

    if data and isinstance(data, dict):
        person_found = bool(data.get("found", False))
        raw_conf = str(data.get("confidence", "LOW")).upper()
        person_confidence = Confidence(raw_conf) if raw_conf in ("LOW", "MID", "HIGH") else Confidence.LOW
        try:
            person_profile = PersonProfile(
                has_job=bool(data.get("has_job", False)),
                job_type=data.get("job_type", "Other"),
                job_related=bool(data.get("job_related", False)),
                job_schedule=data.get("job_schedule", "Unknown"),
                self_employed=bool(data.get("self_employed", False)),
                employer_name=data.get("employer_name", ""),
                employer_city=data.get("employer_city", ""),
                employer_state=data.get("employer_state", ""),
                continuing_education=bool(data.get("continuing_education", False)),
                current_education_institution=data.get("current_education_institution", ""),
                current_education_field=data.get("current_education_field", ""),
                sources=data.get("sources", []) or list(_profile_sources),
            )
        except Exception:
            logger.warning("Could not build profile from parsed data")
    else:
        logger.warning("Could not parse structured profile from agent response")

    # Close the MCP toolset (coresignal mode only)
    if source == "coresignal":
        try:
            await coresignal_toolset.close()
        except RuntimeError:
            pass  # anyio cancel scope cleanup — known ADK/MCP issue

    return AgentResult(
        found=person_found,
        confidence=person_confidence,
        profile=person_profile,
        response=final_text,
        consumption=consumption,
    )


# ── CLI entry point ────────────────────────────────────────

async def main():
    if len(sys.argv) < 3:
        logger.error("Usage: python agent.py 'First' 'Last'")
        sys.exit(1)

    person = PersonInitialDescription(
        first_name=sys.argv[1],
        last_name=sys.argv[2],
        city=sys.argv[3] if len(sys.argv) > 3 else "",
        state=sys.argv[4] if len(sys.argv) > 4 else "",
        school_name=sys.argv[5] if len(sys.argv) > 5 else "",
        school_department=sys.argv[6] if len(sys.argv) > 6 else "",
        graduation_date=sys.argv[7] if len(sys.argv) > 7 else "",
    )
    logger.info("Searching for: {} {}", person.first_name, person.last_name)

    result = await run_agent(person)

    logger.info("AGENT RESPONSE:\n{}", result.response)

    c = result.consumption
    total_in = 0
    total_out = 0
    total_thinking = 0
    total_cost = 0.0
    for model, usage in c.models.items():
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        total_thinking += usage.thinking_tokens
        total_cost += usage.cost_usd
        logger.info(
            "{} | in={} | out={} | thinking={} | cost=${:.6f}",
            model, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens, usage.cost_usd,
        )

    logger.info(
        "TOTAL | in={} | out={} | thinking={} | cost=${:.6f} | LLM limit={} | Tool calls={}",
        total_in, total_out, total_thinking, total_cost, c.max_llm_calls, c.tool_calls_used,
    )
    if c.coresignal_calls:
        for tool_name, count in c.coresignal_calls.items():
            logger.info("Coresignal | {} = {}", tool_name, count)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Error: {}", e)
        raise
