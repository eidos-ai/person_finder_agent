import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

# Add parent directory to path for token_usage import
sys.path.insert(0, str(Path(__file__).parent.parent))

from google import genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from loguru import logger
from token_usage import TokenUsage, extract_token_usage

MASTER_PROFILE_DIR = os.path.expanduser("~/.chromium-profiles/linkedin")
GCP_SA_PATH = "gcp-sa Manuel_Correa JS.json"


def _default_chromium_bin() -> str:
    if sys.platform == "darwin":
        return "/Applications/Chromium.app/Contents/MacOS/Chromium"
    if sys.platform == "win32":
        return r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    return "chromium-browser"


CHROMIUM_BIN = os.environ.get("CHROMIUM_BIN") or _default_chromium_bin()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_SA_PATH

client = genai.Client(
    vertexai=True,
    project="dev-sandbox-359102",
    location="us-central1",
)

class PageLoadError(Exception):
    pass


class NotAuthenticatedError(Exception):
    pass


class EmptyLLMResponseError(Exception):
    pass


class NoSearchResultsError(Exception):
    pass


@dataclass
class LinkedInProfile:
    name: str
    headline: str
    location: str
    profile_url: str


@dataclass
class FullLinkedinProfile:
    markdown: str


def _base_options(headless: bool = False) -> Options:
    opts = Options()
    opts.binary_location = CHROMIUM_BIN
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    if sys.platform == "darwin":
        opts.add_argument("--use-mock-keychain")
    opts.add_argument(f"--user-data-dir={MASTER_PROFILE_DIR}")
    if headless:
        opts.add_argument("--headless=new")
    return opts


def open_chromium_for_login():
    cmd = [
        CHROMIUM_BIN,
        f"--user-data-dir={MASTER_PROFILE_DIR}",
        "https://www.linkedin.com/login",
    ]
    if sys.platform == "darwin":
        cmd.insert(2, "--use-mock-keychain")
    subprocess.Popen(cmd)
    logger.info("Chromium opened — log in to LinkedIn, then close the browser.")


def fetch_linkedin_html(url: str = "https://www.linkedin.com/feed", wait: int = 10) -> str:
    # Fix URLs mangled by LLMs (e.g. "https.www." instead of "https://www.")
    if not re.match(r"^https?://", url):
        url = re.sub(r"^https?[.\s:/]+", "https://", url)
    if not url.startswith("http"):
        url = "https://" + url
    opts = _base_options(headless=True)
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)
        time.sleep(wait)
        html = driver.page_source

        if not html or len(html.strip()) < 100:
            raise PageLoadError(f"Empty HTML for: {url}")

        if "LinkedIn Login" in html or ("Sign in" in html and "/login" in html):
            raise NotAuthenticatedError("Not logged in to LinkedIn")

        try:
            main = driver.find_element("tag name", "main")
            html = main.get_attribute("outerHTML")
        except Exception:

            
            pass

        return html
    finally:
        driver.quit()


def _strip_html(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_json(raw: str):
    """Extract and parse JSON from an LLM response, handling markdown fences and minor formatting issues."""
    # Try to pull content from a markdown code fence first
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    # Fallback: strip leading/trailing fences loosely
    raw = re.sub(r"^```json\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw)
    # Remove trailing commas before } or ] (common LLM mistake)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as first_err:
        # Fix: array with bare key-value pairs instead of wrapped objects
        # e.g. '[\n "name": "val", ...]' -> '[{\n "name": "val", ...}]'
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1].strip()
            if inner and not inner.startswith("{"):
                return json.loads("[{" + inner + "}]")
        raise first_err


def fetch_profile_html(profile_url: str, wait: int = 10) -> str:
    return fetch_linkedin_html(profile_url, wait=wait)


def parse_full_linkedin_profile(html: str) -> tuple[FullLinkedinProfile, TokenUsage]:
    text = _strip_html(html)[:60000]

    model_name = "gemini-2.0-flash"
    prompt = f"""Convert this LinkedIn profile page into a clean, well-structured markdown document describing the person.
Include all relevant information: name, headline, location, current and past work experience, education, skills, certifications, and any other notable details.
Return ONLY the markdown text, no extra commentary.

Profile content:
{text}"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    usage = extract_token_usage(response, model_name)

    md = response.text.strip()
    if not md:
        raise EmptyLLMResponseError("No profile information extracted")

    return FullLinkedinProfile(markdown=md), usage


def get_full_linkedin_profile(profile_url: str, wait: int = 10) -> tuple[FullLinkedinProfile, TokenUsage]:
    html = fetch_profile_html(profile_url, wait=wait)
    return parse_full_linkedin_profile(html)


def search_profiles(keywords: str) -> tuple[list[LinkedInProfile], TokenUsage]:
    url = f"https://www.linkedin.com/search/results/people/?keywords={quote(keywords)}"
    html = fetch_linkedin_html(url)

    if "No results found" in html or "no-results" in html:
        logger.info(f"No LinkedIn search results found for: {keywords}")
        # Return empty list instead of raising exception - caller handles empty results gracefully
        empty_usage = TokenUsage(
            model_name="gemini-2.0-flash",
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            cached_tokens=0,
            cost_usd=0.0,
        )
        return [], empty_usage

    profile_urls = re.findall(r'href="(https://www\.linkedin\.com/in/[^"?]+)', html)
    profile_urls = list(dict.fromkeys(profile_urls))

    text = _strip_html(html)[:60000]

    model_name = "gemini-2.0-flash"
    prompt = f"""Extract all people/profiles from this LinkedIn search results page.

These profile URLs were found on the page — match each person to their URL:
{json.dumps(profile_urls)}

Return ONLY a JSON array with:
- name
- headline
- location
- profile_url (must be one of the URLs listed above)

Page content:
{text}"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    usage = extract_token_usage(response, model_name)

    items = _extract_json(response.text)
    if not items:
        raise EmptyLLMResponseError("No profiles extracted")
    # Handle LLM returning stringified JSON objects inside the array
    if isinstance(items, list) and items and isinstance(items[0], str):
        items = [json.loads(i) if isinstance(i, str) else i for i in items]

    profiles = [
        LinkedInProfile(
            name=i.get("name", ""),
            headline=i.get("headline", ""),
            location=i.get("location", ""),
            profile_url=i.get("profile_url", ""),
        )
        for i in items
    ]

    return profiles, usage


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] in ("login", "--login", "-l"):
        open_chromium_for_login()
    elif len(sys.argv) > 1 and sys.argv[1] == "profile" and len(sys.argv) > 2:
        profile_url = sys.argv[2]
        info, usage = get_full_linkedin_profile(profile_url)
        print(info.markdown)
        logger.info(
            "USAGE | {} | in={} | out={} | cost_usd={:.6f}",
            usage.model_name, usage.input_tokens, usage.output_tokens, usage.cost_usd,
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "search":
        keywords = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "software engineer"
        results, usage = search_profiles(keywords)
        for p in results:
            logger.info("{} | {} | {} | {}", p.name, p.headline, p.location, p.profile_url)
        logger.info(
            "USAGE | {} | in={} | out={} | cost_usd={:.6f}",
            usage.model_name, usage.input_tokens, usage.output_tokens, usage.cost_usd,
        )
