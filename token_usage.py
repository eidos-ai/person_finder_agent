"""Shared TokenUsage dataclass and helpers for tracking LLM consumption."""

from dataclasses import dataclass

# ---- pricing (USD per 1M tokens) ----
# (input, output)
MODEL_PRICING = {
    "gemini-2.0-flash": (0.30, 2.50),
    "gemini-2.5-flash": (0.15, 3.50),
    "gemini-2.5-pro": (1.25, 10.00),
}
DEFAULT_PRICING = (0.30, 2.50)


@dataclass
class TokenUsage:
    model_name: str
    input_tokens: int
    output_tokens: int
    thinking_tokens: int
    cost_usd: float


def _safe_int(x, default: int = 0) -> int:
    try:
        return int(x) if x is not None else default
    except Exception:
        return default


def compute_cost_usd(
    input_tokens: int, output_tokens: int, model_name: str = "gemini-2.0-flash",
    thinking_tokens: int = 0,
) -> float:
    input_price, output_price = MODEL_PRICING.get(model_name, DEFAULT_PRICING)
    return (
        (input_tokens / 1_000_000) * input_price
        + ((output_tokens + thinking_tokens) / 1_000_000) * output_price
    )


def extract_token_usage(response, model_name: str) -> TokenUsage:
    """Extract token usage from a google-genai response object."""
    input_tokens = 0
    output_tokens = 0
    thinking_tokens = 0

    um = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
    if um is not None:
        input_tokens = _safe_int(getattr(um, "prompt_token_count", None), input_tokens)
        output_tokens = _safe_int(getattr(um, "candidates_token_count", None), output_tokens)

        input_tokens = _safe_int(getattr(um, "input_tokens", None), input_tokens)
        output_tokens = _safe_int(getattr(um, "output_tokens", None), output_tokens)

        input_tokens = _safe_int(getattr(um, "prompt_tokens", None), input_tokens)
        output_tokens = _safe_int(getattr(um, "completion_tokens", None), output_tokens)

        thinking_tokens = _safe_int(getattr(um, "thoughts_token_count", None), thinking_tokens)

        total = getattr(um, "total_token_count", None)
        if total is not None and output_tokens == 0 and input_tokens > 0:
            total_i = _safe_int(total, 0)
            if total_i >= input_tokens:
                output_tokens = total_i - input_tokens

    cost_usd = compute_cost_usd(input_tokens, output_tokens, model_name, thinking_tokens)

    return TokenUsage(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        thinking_tokens=thinking_tokens,
        cost_usd=cost_usd,
    )
