"""LLM Customer Feedback Classifier.

Classifies customer feedback into structured, schema-validated JSON.
Demonstrates production-shaped LLM workflow patterns:
  - Pydantic schema constraints on the LLM output
  - Retry with exponential backoff for transient API errors
  - Defensive parsing (handles markdown-fenced JSON)
  - Graceful fallback when the LLM repeatedly fails
"""
from __future__ import annotations

import json
import os
import time
from enum import Enum
from typing import List, Optional
from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic, APIConnectionError, APIError, RateLimitError
from pydantic import BaseModel, Field, ValidationError


# ---------- Output schema ----------

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ChurnRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FeedbackAnalysis(BaseModel):
    """Strictly-typed analysis of one piece of customer feedback."""
    sentiment: Sentiment = Field(..., description="Overall sentiment.")
    themes: List[str] = Field(
        ..., max_length=3,
        description="Up to 3 short snake_case themes (e.g. 'pricing').",
    )
    churn_risk: ChurnRisk = Field(..., description="Likelihood the customer churns.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., max_length=200)


# ---------- Prompting ----------

SYSTEM_PROMPT = """You are a customer-feedback analyst. Read the review and \
return a JSON object with these fields and types:

{
  "sentiment": "positive" | "neutral" | "negative",
  "themes": ["short_snake_case", ...]   // up to 3 items
  "churn_risk": "low" | "medium" | "high",
  "confidence": 0.0-1.0 float,
  "rationale": "<= 200 char single sentence"
}

Return ONLY the JSON object. No prose. No markdown code fences."""


# ---------- Fallback (returned when LLM fails after all retries) ----------

def _fallback(reason: str) -> FeedbackAnalysis:
    return FeedbackAnalysis(
        sentiment=Sentiment.NEUTRAL,
        themes=["unparseable"],
        churn_risk=ChurnRisk.MEDIUM,
        confidence=0.0,
        rationale=f"LLM call failed; safe fallback. {reason}"[:200],
    )


# ---------- Helpers ----------

def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model adds them despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------- Main entry point ----------

def analyse(
    review: str,
    *,
    client: Optional[Anthropic] = None,
    model: str = "claude-haiku-4-5-20251001",
    max_retries: int = 3,
    base_backoff: float = 1.0,
) -> FeedbackAnalysis:
    """Classify one customer review.

    Returns a validated FeedbackAnalysis. On repeated LLM failure
    returns a safe fallback rather than raising — so callers in a
    pipeline don't crash on a single bad call.
    """
    if not review or not review.strip():
        return _fallback("empty input")

    client = client or Anthropic()
    last_error = "unknown"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=400,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": review}],
            )
            raw = response.content[0].text
            cleaned = _strip_fences(raw)
            parsed = json.loads(cleaned)
            return FeedbackAnalysis.model_validate(parsed)

        except (json.JSONDecodeError, ValidationError) as e:
            # Schema/parse failure — worth one more shot
            last_error = f"parse: {type(e).__name__}: {str(e)[:80]}"

        except (APIConnectionError, RateLimitError) as e:
            # Transient — backoff and retry
            last_error = f"transient: {type(e).__name__}"

        except APIError as e:
            last_error = f"api: {type(e).__name__}"
            status = getattr(e, "status_code", 0)
            # 4xx (except 429) won't fix themselves — bail early
            if 400 <= status < 500 and status != 429:
                break

        # Exponential backoff before the next attempt
        if attempt < max_retries:
            time.sleep(base_backoff * (2 ** (attempt - 1)))

    return _fallback(f"after {max_retries} attempts: {last_error}")
