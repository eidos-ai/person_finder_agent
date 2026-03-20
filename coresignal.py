"""Coresignal API helpers (direct HTTP, not MCP)."""

import requests
from loguru import logger

from agent import CORESIGNAL_API_KEY

_BASE_URL = "https://api.coresignal.com/cdapi/v2"


def get_credits_remaining() -> int | None:
    """Return remaining search credits by making a cheap API call.

    Queries a nonsense name so the result is empty ([]) but the
    response still includes the x-credits-remaining header.
    Costs 1 search credit per call.

    Returns the number of remaining credits, or None on failure.
    """
    if not CORESIGNAL_API_KEY:
        logger.warning("CORESIGNAL_API_KEY env var is not set; cannot fetch credits remaining.")
        return None

    headers = {"apikey": CORESIGNAL_API_KEY, "Content-Type": "application/json"}

    try:
        resp = requests.post(
            f"{_BASE_URL}/employee_multi_source/search/es_dsl",
            headers=headers,
            json={"query": {"match": {"first_name": "xyznonexistent"}}},
            timeout=10,
        )
        val = resp.headers.get("x-credits-remaining")
        if val is not None:
            return int(val)
    except Exception as e:
        logger.warning("Failed to fetch Coresignal credits: {}", e)

    return None
