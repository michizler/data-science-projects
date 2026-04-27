# LLM Customer Feedback Analyzer

A small, production-shaped LLM workflow that classifies customer feedback into structured, schema-validated JSON. Built to demonstrate end-to-end LLM engineering patterns — structured outputs, schema validation, retry/backoff, fallback handling, and evaluation against a labelled gold set — in a focused, reviewable codebase.

## What it does

Given a piece of customer feedback (review, NPS comment, support ticket), the classifier returns a strictly-typed analysis:

```json
{
  "sentiment": "negative",
  "themes": ["delivery_speed", "support_quality"],
  "churn_risk": "high",
  "confidence": 0.85,
  "rationale": "Customer explicitly states intent to cancel due to repeated late deliveries and poor support."
}
```

Useful as a building block for: customer-feedback dashboards, churn-risk pipelines, support-ticket triage, and any downstream analytics that needs reliable structured signal from unstructured text.

## Engineering patterns demonstrated

- **Schema-constrained outputs** — Pydantic `BaseModel` with enums and bounds defines the expected shape; the LLM response is validated before it can pollute downstream code.
- **Retry with exponential backoff** — handles transient API errors (rate limits, connection issues) without burning quota on permanent failures. Bails fast on 4xx errors that won't fix themselves.
- **Graceful fallback** — when the LLM returns invalid JSON or hits its retry ceiling, the function returns a safe default (`neutral` / `medium`) with a rationale explaining the failure, so downstream pipelines don't crash on a single bad call.
- **Defensive parsing** — strips markdown code fences, handles the common LLM failure mode of wrapping JSON in ```` ```json ... ``` ```` despite explicit instructions not to.
- **Evaluation harness** — small labelled gold set (`data/gold_set.csv`) and an evaluation script that reports per-field accuracy and surfaces error cases for inspection.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
touch .env
# then edit .env and add your ANTHROPIC_API_KEY

# 3. Classify a single review
export ANTHROPIC_API_KEY=sk-...
python demo.py "Cancelling my subscription. The product has gone downhill."

# 4. Evaluate against the gold set
python evaluate.py
```

## Project structure

```
.
├── classifier.py       # Schema + LLM call + retries + fallback (~140 LOC)
├── demo.py             # CLI for a single classification
├── evaluate.py         # Run against gold set, report metrics + error analysis
├── data/
│   └── gold_set.csv    # 15 labelled reviews
├── requirements.txt
├── .env.example
└── .gitignore
```

## Possible extensions

This is a starting point, not a finished product. Natural next steps:

- **Expand the gold set** to 100+ examples across more diverse domains, edge cases (sarcasm, mixed sentiment), and multiple languages.
- **Add inter-annotator agreement** — have two reviewers label and compute Cohen's kappa as a quality floor for "the dataset itself."
- **A/B test models** — compare Haiku vs Sonnet on accuracy / cost / latency tradeoffs.
- **Add observability** — structured logging, latency histograms, per-failure-mode counters.
- **Batch mode** — process many reviews in parallel with concurrency control and progress reporting.
- **Drift monitoring** — periodically re-evaluate on the gold set to detect quality regressions over time.

## Tech stack

- Python 3.10+
- [`anthropic`](https://pypi.org/project/anthropic/) — Anthropic API client
- [`pydantic`](https://pypi.org/project/pydantic/) — schema validation
