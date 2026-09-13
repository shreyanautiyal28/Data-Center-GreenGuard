"""
ai/generative_narrator.py
─────────────────────────────────────────────────────────────────────
GreenGuard Generative Narrator — optional, safety-gated module.

PUBLIC INTERFACE
────────────────
    generate_narrative(grounded_result: dict) -> str

Accepts the grounded_result dict produced by
ai/grounded_explainer.py::build_grounded_explanation() and returns
an operator-readable narrative string.

DEFAULT BEHAVIOUR
─────────────────
    GREENGUARD_NARRATOR_ENABLED is absent or "false"
    → returns grounded_result["summary"] immediately.
      No API call is made.  No latency is added.

FALLBACK CONTRACT
─────────────────
Every failure path (missing credentials, API error, timeout, empty
output, failed safety validation) returns grounded_result["summary"].
This function NEVER raises an exception into the caller.

SAFETY CONTRACT
───────────────
The LLM output must:
  • contain none of the FORBIDDEN_PHRASES
  • contain all of the REQUIRED_SAFETY_PHRASES
If either check fails the output is discarded and the deterministic
summary is returned instead.

SUPPORTED PROVIDERS
───────────────────
  "none"    — always returns the deterministic summary (default)
  "watsonx" — IBM watsonx.ai via ibm-watsonx-ai SDK (structure
               prepared; live call is not yet wired because the
               installed SDK version has not been verified in this
               environment)

ENVIRONMENT VARIABLES
─────────────────────
  GREENGUARD_NARRATOR_ENABLED          true / false  (default: false)
  GREENGUARD_LLM_PROVIDER              none / watsonx (default: none)
  WATSONX_API_KEY                      IBM watsonx.ai API key
  WATSONX_PROJECT_ID                   IBM watsonx.ai project ID
  WATSONX_URL                          IBM watsonx.ai endpoint URL
  WATSONX_MODEL_ID                     Model ID to use
  GREENGUARD_NARRATOR_MAX_TOKENS       Max output tokens (default: 300)
  GREENGUARD_NARRATOR_TIMEOUT_SECONDS  Call timeout in seconds (default: 10)

IMPORTANT: No credentials are ever hardcoded in this file.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# SAFETY CONSTANTS
# These must remain identical to the values used in
# evaluation/end_to_end_validation.py so that the same gate covers
# both the deterministic and the AI-generated outputs.
# ─────────────────────────────────────────────────────────────────────

FORBIDDEN_PHRASES = [
    "confirmed root cause",
    "confirmed equipment fault",
    "guaranteed savings",
    "guaranteed energy savings",
]

REQUIRED_SAFETY_PHRASES = [
    "not a confirmed equipment fault",
    "modeled estimate",
    "reviewed by an operator",
]


# ─────────────────────────────────────────────────────────────────────
# CONFIGURATION HELPERS
# ─────────────────────────────────────────────────────────────────────

def _narrator_enabled() -> bool:
    """Return True only when explicitly enabled via environment."""
    return os.environ.get(
        "GREENGUARD_NARRATOR_ENABLED", "false"
    ).strip().lower() == "true"


def _provider() -> str:
    """Return the configured LLM provider name in lower-case."""
    return os.environ.get(
        "GREENGUARD_LLM_PROVIDER", "none"
    ).strip().lower()


def _max_tokens() -> int:
    try:
        return int(
            os.environ.get("GREENGUARD_NARRATOR_MAX_TOKENS", "300")
        )
    except ValueError:
        return 300


def _timeout() -> float:
    try:
        return float(
            os.environ.get("GREENGUARD_NARRATOR_TIMEOUT_SECONDS", "10")
        )
    except ValueError:
        return 10.0


# ─────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# Separated from _call_llm so it can be tested without any API key.
# ─────────────────────────────────────────────────────────────────────

def build_prompt(grounded_result: dict) -> tuple[str, str]:
    """
    Build (system_prompt, user_message) from grounded_result.

    Uses ONLY:
        grounded_result["summary"]
        grounded_result["observed_evidence"]
        grounded_result["retrieved_knowledge"]
        grounded_result["investigation_actions"]
        grounded_result["caveats"]

    Returns a (system_prompt, user_message) tuple.
    Does not call any external service.
    Does not mutate grounded_result.
    """

    # ------------------------------------------------------------------
    # Pull evidence values — read-only, never mutated
    # ------------------------------------------------------------------

    evidence = grounded_result.get("observed_evidence", {})
    retrieved = grounded_result.get("retrieved_knowledge", [])
    actions = grounded_result.get("investigation_actions", [])
    caveats = grounded_result.get("caveats", [])
    deterministic_summary = str(
        grounded_result.get("summary", "")
    )

    # ------------------------------------------------------------------
    # SYSTEM PROMPT — fixed safety contract
    # ------------------------------------------------------------------

    caveats_block = "\n".join(
        f"  - {c}" for c in caveats
    ) if caveats else "  (none)"

    system_prompt = f"""You are an operator-facing sustainability assistant \
for a data-center monitoring system called GreenGuard.

Your ONLY task is to rewrite the provided deterministic summary in clear, \
concise, operator-readable English. Write 3 to 5 sentences maximum.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY SAFETY RULES — NEVER VIOLATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DO NOT invent, round, or extrapolate any numeric value.
   Use ONLY the exact numbers provided in the evidence block.

2. DO NOT use the phrases (in any form):
   - "confirmed root cause"
   - "confirmed equipment fault"
   - "guaranteed savings"
   - "guaranteed energy savings"

3. The likely subsystem MUST always be described as an investigation
   hypothesis, not a confirmed finding.
   Use language such as: "the leading hypothesis", "may indicate",
   "warrants investigation", or equivalent.

4. Estimated excess power MUST always be described as a modelled
   estimate relative to a statistical baseline — never as a measured
   saving or a guaranteed outcome.

5. Every investigation action MUST be qualified as a step requiring
   operator review before any operational change is made.

6. You MUST preserve the meaning of ALL of the following mandatory
   caveats. They may be incorporated into the narrative but must not
   be omitted or contradicted:

{caveats_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT YOU MUST NOT DO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Do not perform anomaly detection.
- Do not perform impact calculations.
- Do not perform root-cause classification.
- Do not add information not present in the evidence block.
- Do not remove or replace the investigation actions.
- Do not produce more than 5 sentences.
"""

    # ------------------------------------------------------------------
    # USER MESSAGE — per-event evidence block
    # ------------------------------------------------------------------

    # Observed evidence
    evidence_lines = []
    for key, value in evidence.items():
        evidence_lines.append(f"  {key}: {value}")
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "  (none)"

    # Retrieved knowledge topics + conditions
    knowledge_lines = []
    for item in retrieved:
        topic = item.get("topic", "")
        condition = item.get("condition", "")
        similarity = item.get("similarity", "")
        knowledge_lines.append(
            f"  - Topic: {topic} | Condition: {condition} "
            f"| Similarity: {similarity}"
        )
    knowledge_block = (
        "\n".join(knowledge_lines) if knowledge_lines else "  (none)"
    )

    # Investigation actions
    actions_block = "\n".join(
        f"  {i + 1}. {a}" for i, a in enumerate(actions)
    ) if actions else "  (none)"

    user_message = f"""DETERMINISTIC SUMMARY (rewrite this in natural language):
{deterministic_summary}

OBSERVED EVIDENCE (use these exact values — do not alter them):
{evidence_block}

RETRIEVED SUSTAINABILITY KNOWLEDGE (use for context only):
{knowledge_block}

INVESTIGATION ACTIONS (preserve these exactly, require operator review):
{actions_block}

Write the operator narrative now, following all mandatory safety rules \
in the system prompt.
"""

    return system_prompt, user_message


# ─────────────────────────────────────────────────────────────────────
# PROVIDER ABSTRACTION
# ─────────────────────────────────────────────────────────────────────

def _call_llm(system_prompt: str, user_message: str) -> str:
    """
    Route the LLM call to the configured provider.

    provider="none"
        Returns an empty string immediately.
        The caller (generate_narrative) will fall back to the
        deterministic summary.

    provider="watsonx"
        Architecture is prepared for IBM watsonx.ai via the
        ibm-watsonx-ai SDK. The live call is stubbed with a
        NotImplementedError until the installed SDK version is
        verified in this environment and the integration is
        explicitly approved.

    Any exception raised here is caught by generate_narrative.
    """

    provider = _provider()

    if provider == "none":
        # Explicit no-op — return empty so generate_narrative falls back.
        return ""

    if provider == "watsonx":
        return _call_watsonx(system_prompt, user_message)

    # Unknown provider — log and return empty so caller falls back.
    logger.warning(
        "GreenGuard narrator: unknown provider %r — "
        "falling back to deterministic summary.",
        provider,
    )
    return ""


def _call_watsonx(system_prompt: str, user_message: str) -> str:
    """
    IBM watsonx.ai provider stub.

    Structure is prepared for ibm-watsonx-ai SDK integration.
    The live call raises NotImplementedError until:
      1. The installed SDK version has been verified.
      2. The integration has been explicitly approved.

    Required environment variables:
        WATSONX_API_KEY
        WATSONX_PROJECT_ID
        WATSONX_URL         (default: https://us-south.ml.cloud.ibm.com)
        WATSONX_MODEL_ID    (default: ibm/granite-3-8b-instruct)

    When the live implementation is added, it should:
      - Read credentials from environment only (never hardcode)
      - Pass max_new_tokens=_max_tokens()
      - Enforce a timeout of _timeout() seconds
      - Return the model's generated text as a plain string
    """

    api_key = os.environ.get("WATSONX_API_KEY", "")
    project_id = os.environ.get("WATSONX_PROJECT_ID", "")
    url = os.environ.get(
        "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
    )
    model_id = os.environ.get(
        "WATSONX_MODEL_ID", "ibm/granite-3-8b-instruct"
    )

    if not api_key:
        logger.warning(
            "GreenGuard narrator: WATSONX_API_KEY is not set — "
            "falling back to deterministic summary."
        )
        return ""

    if not project_id:
        logger.warning(
            "GreenGuard narrator: WATSONX_PROJECT_ID is not set — "
            "falling back to deterministic summary."
        )
        return ""

    # Credential presence verified. Live call is not yet wired.
    raise NotImplementedError(
        "watsonx live call is not yet implemented. "
        "Verify ibm-watsonx-ai SDK version and wire the call here. "
        f"Configured: url={url}, model_id={model_id}, "
        f"max_tokens={_max_tokens()}, timeout={_timeout()}s."
    )


# ─────────────────────────────────────────────────────────────────────
# OUTPUT SAFETY VALIDATION
# ─────────────────────────────────────────────────────────────────────

def _validate_output(text: str) -> tuple[bool, list[str]]:
    """
    Validate a generated narrative string against the safety rules.

    Returns (passed: bool, reasons: list[str]).

    Rules mirror evaluation/end_to_end_validation.py exactly so that
    the same gate covers both deterministic and AI-generated outputs.

    Forbidden-phrase scanning works on a scrubbed copy of the text
    where all REQUIRED_SAFETY_PHRASES have been blanked out first.
    This prevents a false positive where a required phrase such as
    "not a confirmed equipment fault" triggers the forbidden-phrase
    check for "confirmed equipment fault" as a substring.
    """

    reasons = []
    lower = text.lower()

    # Build a scrubbed copy: blank out required-safety-phrase occurrences
    # before scanning for forbidden phrases.  This prevents the negating
    # context ("not a confirmed equipment fault") from being caught by the
    # substring search for the forbidden phrase ("confirmed equipment fault").
    scrubbed = lower
    for safe_phrase in REQUIRED_SAFETY_PHRASES:
        scrubbed = scrubbed.replace(safe_phrase, " " * len(safe_phrase))

    # Forbidden phrases — checked against the scrubbed text
    for phrase in FORBIDDEN_PHRASES:
        if phrase in scrubbed:
            reasons.append(f"forbidden phrase present: {phrase!r}")

    # Required safety phrases — checked against the original text
    for phrase in REQUIRED_SAFETY_PHRASES:
        if phrase not in lower:
            reasons.append(f"required safety phrase absent: {phrase!r}")

    # Non-empty
    if not text.strip():
        reasons.append("output is empty")

    passed = len(reasons) == 0
    return passed, reasons


# ─────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────

def generate_narrative(grounded_result: dict) -> str:
    """
    Generate an operator-readable narrative from a grounded_result dict.

    The grounded_result dict is NEVER mutated.

    Returns the AI-generated narrative string on success.
    Returns grounded_result["summary"] in every failure case:
        - narrator disabled (default)
        - provider is "none"
        - credentials missing
        - API error or timeout
        - empty output
        - safety validation failure

    Never raises an exception.

    Parameters
    ----------
    grounded_result : dict
        The dict returned by build_grounded_explanation().
        Required keys: "summary", "observed_evidence",
        "retrieved_knowledge", "investigation_actions", "caveats".

    Returns
    -------
    str
        Operator-readable narrative, or the deterministic summary
        on any failure.
    """

    # Retrieve the deterministic fallback first — used on every failure path.
    deterministic_summary = str(grounded_result.get("summary", ""))

    # ------------------------------------------------------------------
    # Gate 1: narrator must be explicitly enabled
    # ------------------------------------------------------------------

    if not _narrator_enabled():
        return deterministic_summary

    # ------------------------------------------------------------------
    # Gate 2: provider must be something other than "none"
    # ------------------------------------------------------------------

    if _provider() == "none":
        logger.debug(
            "GreenGuard narrator: provider is 'none' — "
            "returning deterministic summary."
        )
        return deterministic_summary

    # ------------------------------------------------------------------
    # Build the prompt (no API call, safe to run)
    # ------------------------------------------------------------------

    try:
        system_prompt, user_message = build_prompt(grounded_result)
    except Exception:
        logger.exception(
            "GreenGuard narrator: prompt build failed — "
            "returning deterministic summary."
        )
        return deterministic_summary

    # ------------------------------------------------------------------
    # Call the LLM
    # ------------------------------------------------------------------

    try:
        raw_output = _call_llm(system_prompt, user_message)
    except NotImplementedError as exc:
        logger.warning(
            "GreenGuard narrator: %s — returning deterministic summary.",
            exc,
        )
        return deterministic_summary
    except Exception:
        logger.exception(
            "GreenGuard narrator: API call failed — "
            "returning deterministic summary."
        )
        return deterministic_summary

    # ------------------------------------------------------------------
    # Gate 3: reject empty output
    # ------------------------------------------------------------------

    if not raw_output or not raw_output.strip():
        logger.warning(
            "GreenGuard narrator: received empty output — "
            "returning deterministic summary."
        )
        return deterministic_summary

    # ------------------------------------------------------------------
    # Gate 4: safety validation
    # ------------------------------------------------------------------

    passed, reasons = _validate_output(raw_output)

    if not passed:
        logger.warning(
            "GreenGuard narrator: output failed safety validation "
            "(%s) — returning deterministic summary.",
            "; ".join(reasons),
        )
        return deterministic_summary

    # ------------------------------------------------------------------
    # All gates passed — return the AI-generated narrative
    # ------------------------------------------------------------------

    return raw_output.strip()
