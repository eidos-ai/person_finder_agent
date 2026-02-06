#!/usr/bin/env python3
"""
LinkedIn Profile Agent using Google ADK.
Receives a name and surname, uses LinkedIn tools to find profile information.
Returns its findings along with per-model token consumption.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass

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
from google.genai import types
from google.genai.types import GenerateContentConfig, ThinkingConfig

from skills.fetch_linkedin import (
    search_profiles as _search_profiles,
    get_full_linkedin_profile as _get_full_linkedin_profile,
    FullLinkedinProfile,
    _extract_json,
)
from skills.fetch_google import (
    EmptyLLMResponseError as _GoogleEmptyError,
    fetch_page_content as _fetch_page_content,
    google_search as _google_search,
)
from token_usage import (
    TokenUsage,
    compute_cost_usd,
)


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
    education_institution: str
    education_field: str
    sources: list[str]  # URLs where this information was sourced from


@dataclass
class ModelUsage:
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float


@dataclass
class Consumption:
    tool_calls_used: int
    max_llm_calls: int
    models: dict[str, ModelUsage]


@dataclass
class AgentResult:
    profile: PersonProfile | None
    response: str
    consumption: Consumption


# ── configuration ──────────────────────────────────────────
APP_NAME = "linkedin_profile_agent"
USER_ID = "user1"
SESSION_ID = "session1"
MAX_LLM_CALLS = 20
AGENT_MODEL = "gemini-2.5-pro"

AGENT_INSTRUCTION = """\
You are a LinkedIn profile research agent.

You will receive a person description with: first name, last name, city, state,
school name, school department, and graduation date.

Your goal is to find their LinkedIn profile and current employment information.

You also have access to fetch_web_page to open any URL and read its content.
Use it to complement LinkedIn information — for example, to verify an employer,
look up a company website, or find additional details about the person.

Steps:
1. Use search_linkedin_profiles with the person's full name to find matching profiles.
2. Review the results and pick the best match using all available context
   (location, school, department, graduation date) to disambiguate.
3. Use get_linkedin_job_info with the profile URL to get their profile markdown.
4. If needed, use fetch_web_page to fill in missing information
   (e.g. if missing info for employee).
5. Return your final answer as ONLY a JSON object (no markdown fences, no extra text) with these fields:
   - has_job: boolean (true if currently employed)
   - job_type: one of "RelatedJob", "StillLooking", "MilitaryService", "ContinuingEducation", "UnrelatedJob", "Other"
   - job_related: boolean (true if the current job appears related to their field of study)
   - job_schedule: one of "FullTime", "PartTime", "Unknown"
   - self_employed: boolean (true if freelancing or running their own business)
   - employer_name: string (empty string if not employed)
   - employer_city: string (empty string if unknown)
   - employer_state: string (empty string if unknown)
   - continuing_education: boolean (true if currently enrolled in education)
   - education_institution: string (empty string if not in continuing education)
   - education_field: string (what they are studying, empty string if N/A)

Be efficient — do not repeat searches unnecessarily.
If the first search returns no good match, try adding the school name or city
to the search keywords.
If no profiles are found, return the JSON with has_job=false, job_type="Other", and empty strings for all other fields.
"""

# Accumulator for token usage from LLM calls *inside* tools
_tool_token_usages: list[TokenUsage] = []
_last_profile: FullLinkedinProfile | None = None
_profile_sources: list[str] = []


# ── tool functions ─────────────────────────────────────────

def search_linkedin_profiles(keywords: str) -> str:
    """
    Search LinkedIn for people matching the given keywords (typically a name).
    Returns a list of matching profiles with name, headline, location, and URL.

    Args:
        keywords: Search terms, e.g. "John Smith".

    Returns:
        Formatted search results, one profile per line.
    """
    profiles, usage = _search_profiles(keywords)
    _tool_token_usages.append(usage)

    if not profiles:
        return "No profiles found for these keywords."

    lines = []
    for i, p in enumerate(profiles, 1):
        lines.append(
            f"{i}. {p.name} | {p.headline} | "
            f"{p.location} | {p.profile_url}"
        )
    return "\n".join(lines)


def get_linkedin_job_info(profile_url: str) -> str:
    """
    Fetch a LinkedIn profile page and extract current employment and education information.

    Args:
        profile_url: Full LinkedIn profile URL (e.g. https://www.linkedin.com/in/username).

    Returns:
        Employment and education details from the profile.
    """
    global _last_profile
    info, usage = _get_full_linkedin_profile(profile_url)
    _tool_token_usages.append(usage)
    _last_profile = info
    _profile_sources.append(profile_url)

    return info.markdown


def google_search(query: str) -> str:
    """
    Search Google for the given query and return a list of results with URLs, titles, and descriptions.
    Use this to find web pages about a person, company, or topic.

    Args:
        query: Search terms, e.g. "Acme Corp Denver Colorado".

    Returns:
        Formatted search results, one per line.
    """
    try:
        results, usage = _google_search(query)
    except _GoogleEmptyError as exc:
        return f"Google search failed: {exc}"
    _tool_token_usages.append(usage)

    if not results:
        return "No search results found."

    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title} | {r.page_url}\n   {r.description}")
    return "\n".join(lines)


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


# ── ADK agent definition ──────────────────────────────────

search_tool = FunctionTool(search_linkedin_profiles)
job_info_tool = FunctionTool(get_linkedin_job_info)
google_search_tool = GoogleSearchTool(bypass_multi_tools_limit=True)
# google_search_tool = FunctionTool(google_search)
# web_page_tool = FunctionTool(fetch_web_page)

root_agent = Agent(
    name="linkedin_profile_agent",
    model=AGENT_MODEL,
    description="Agent that researches a person's LinkedIn profile and employment status.",
    instruction=AGENT_INSTRUCTION,
    tools=[search_tool, job_info_tool, google_search_tool],
    generate_content_config=GenerateContentConfig(
        thinking_config=ThinkingConfig(include_thoughts=True),
    ),
)


# ── runner ─────────────────────────────────────────────────

async def run_agent(person: PersonInitialDescription) -> AgentResult:
    """Run the LinkedIn profile agent for the given person."""
    global _last_profile
    _tool_token_usages.clear()
    _last_profile = None
    _profile_sources.clear()

    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)

    prompt = (
        f"Find LinkedIn profile information for: {person.first_name} {person.last_name}\n"
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
    final_text = ""

    # Agent-level token usage (orchestration LLM calls)
    agent_input_tokens = 0
    agent_output_tokens = 0
    agent_thinking_tokens = 0

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
                fr = getattr(part, "function_response", None)
                if fr is not None:
                    resp_text = str(fr.response)
                    if len(resp_text) > 300:
                        resp_text = resp_text[:300] + "..."
                    logger.debug("RESULT {} -> {}", fr.name, resp_text)
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    logger.debug("THINKING: {}", part.text)
                elif getattr(part, "text", None) and not event.is_final_response():
                    logger.debug("REASONING: {}", part.text)

        # Accumulate agent-level token usage
        um = event.usage_metadata
        if um is not None:
            agent_input_tokens += getattr(um, "prompt_token_count", 0) or 0
            agent_output_tokens += getattr(um, "response_token_count", 0) or 0
            agent_thinking_tokens += getattr(um, "thoughts_token_count", 0) or 0

        if event.is_final_response():
            final_text = event.content.parts[0].text

    # ── build consumption report ───────────────────────────
    # Per-model aggregation
    models: dict[str, ModelUsage] = {}

    # Agent orchestration usage
    agent_cost = compute_cost_usd(agent_input_tokens, agent_output_tokens, AGENT_MODEL, agent_thinking_tokens)
    models[AGENT_MODEL] = ModelUsage(
        input_tokens=agent_input_tokens,
        output_tokens=agent_output_tokens,
        thinking_tokens=agent_thinking_tokens,
        cost_usd=agent_cost,
    )

    # Tool-internal LLM usage
    for u in _tool_token_usages:
        key = f"{u.model_name} (tools)"
        if key not in models:
            models[key] = ModelUsage(input_tokens=0, output_tokens=0, thinking_tokens=0, cost_usd=0.0)
        models[key].input_tokens += u.input_tokens
        models[key].output_tokens += u.output_tokens
        models[key].thinking_tokens += u.thinking_tokens
        models[key].cost_usd += u.cost_usd

    consumption = Consumption(
        tool_calls_used=tool_call_count,
        max_llm_calls=MAX_LLM_CALLS,
        models=models,
    )

    person_profile = None
    if final_text:
        try:
            data = _extract_json(final_text)
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
                education_institution=data.get("education_institution", ""),
                education_field=data.get("education_field", ""),
                sources=list(_profile_sources),
            )
        except (json.JSONDecodeError, AttributeError):
            logger.warning("Could not parse structured profile from agent response")

    return AgentResult(
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


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.exception("Error: {}", e)
        raise