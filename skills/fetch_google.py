import json
import os
import re
import time
from dataclasses import dataclass
from urllib.parse import quote

from google import genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from loguru import logger
from token_usage import TokenUsage, extract_token_usage

CHROMIUM_BIN = "/Applications/Chromium.app/Contents/MacOS/Chromium"
GCP_SA_PATH = "/Users/marceloortega/eidos/jarlath_scrap/gcp-sa Manuel_Correa JS.json"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCP_SA_PATH

client = genai.Client(
    vertexai=True,
    project="dev-sandbox-359102",
    location="us-central1",
)


class PageLoadError(Exception):
    pass


class EmptyLLMResponseError(Exception):
    pass


class NoSearchResultsError(Exception):
    pass


@dataclass
class SearchResult:
    page_url: str
    title: str
    description: str


def _base_options(headless: bool = False) -> Options:
    opts = Options()
    opts.binary_location = CHROMIUM_BIN
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--use-mock-keychain")
    if headless:
        opts.add_argument("--headless=new")
    return opts


def _fetch_google_html(url: str, wait: int = 5) -> str:
    opts = _base_options(headless=True)
    driver = webdriver.Chrome(options=opts)
    try:
        driver.get(url)
        time.sleep(wait)
        html = driver.page_source

        if not html or len(html.strip()) < 100:
            raise PageLoadError(f"Empty HTML for: {url}")

        try:
            main = driver.find_element("id", "search")
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


def google_search(query: str, wait: int = 5) -> tuple[list[SearchResult], TokenUsage]:
    """Search Google and return structured results parsed by LLM.

    Args:
        query: The search query string.
        wait: Seconds to wait for page load.

    Returns:
        A tuple of (list of SearchResult, TokenUsage).
    """
    url = f"https://www.google.com/search?q={quote(query)}"
    html = _fetch_google_html(url, wait=wait)

    if "did not match any documents" in html:
        raise NoSearchResultsError(f"No results for: {query}")

    text = _strip_html(html)[:60000]

    model_name = "gemini-2.0-flash"
    prompt = f"""Extract all search results from this Google search results page.
Return ONLY a JSON array where each element has:
- page_url: the URL of the search result
- title: the title of the search result
- description: the snippet/description shown under the title

Only include actual search results, not ads, "People also ask", or related searches.

Page content:
{text}"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    usage = extract_token_usage(response, model_name)

    raw = response.text.strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    items = json.loads(raw)
    if not items:
        raise EmptyLLMResponseError("No search results extracted")

    results = [
        SearchResult(
            page_url=i.get("page_url", ""),
            title=i.get("title", ""),
            description=i.get("description", ""),
        )
        for i in items
    ]

    return results, usage


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "OpenAI latest news"
    results, usage = google_search(query)
    for r in results:
        logger.info("{} | {}\n  {}", r.title, r.page_url, r.description)
    logger.info(
        "USAGE | {} | in={} | out={} | cost_usd={:.6f}",
        usage.model_name, usage.input_tokens, usage.output_tokens, usage.cost_usd,
    )
