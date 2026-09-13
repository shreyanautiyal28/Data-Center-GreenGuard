import sys
from pathlib import Path

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from rag.retriever import SustainabilityRetriever
from ai.evidence_rag_pipeline import build_evidence_query
from ai.grounded_explainer import build_grounded_explanation


INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "greenguard_final_anomalies.parquet"
)

REQUIRED_COLUMNS = [
    "ts",
    "impact_rank",
    "impact_priority",
    "likely_subsystem",
    "pue",
    "it_power_kw",
    "estimated_excess_power_kw",
    "pue_deviation",
    "hvac_deviation_ratio",
    "pump_deviation_ratio",
    "cooling_deviation_ratio",
    "hvac_to_it_ratio",
    "pump_to_it_ratio",
    "cooling_to_it_ratio",
]

FORBIDDEN_PHRASES = [
    "confirmed root cause",
    "confirmed equipment fault",
    "guaranteed savings",
    "guaranteed energy savings",
]


def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
        return True

    print(f"[FAIL] {message}")
    return False


# ===================================================================
# START
# ===================================================================

print("=" * 70)
print("GREENGUARD END-TO-END QUALITY GATE")
print("=" * 70)

failures = 0


# ===================================================================
# 1. FINAL ANOMALY DATASET
# ===================================================================

print("\n" + "-" * 70)
print("1. FINAL ANOMALY DATASET")
print("-" * 70)

if not check(
    INPUT.exists(),
    "Final anomaly dataset exists"
):
    failures += 1
    raise SystemExit(1)

df = pd.read_parquet(INPUT)

if not check(
    len(df) > 0,
    f"Dataset contains {len(df):,} anomaly records"
):
    failures += 1


# ===================================================================
# 2. QUANTITATIVE EVIDENCE
# ===================================================================

print("\n" + "-" * 70)
print("2. QUANTITATIVE EVIDENCE")
print("-" * 70)

missing_columns = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if not check(
    len(missing_columns) == 0,
    "All required quantitative evidence columns are present"
):
    failures += 1
    print("Missing columns:", missing_columns)


if len(missing_columns) == 0:

    if not check(
        df["pue"].notna().all(),
        "PUE values are available"
    ):
        failures += 1

    if not check(
        df["it_power_kw"].notna().all(),
        "IT-load values are available"
    ):
        failures += 1

    if not check(
        df["likely_subsystem"].notna().all(),
        "Subsystem hypotheses are available"
    ):
        failures += 1

    if not check(
        df["impact_priority"].notna().all(),
        "Impact priorities are available"
    ):
        failures += 1


# ===================================================================
# 3. RAG INITIALIZATION
# ===================================================================

print("\n" + "-" * 70)
print("3. RAG INITIALIZATION")
print("-" * 70)

rag_ready = False

try:

    retriever = SustainabilityRetriever()

    rag_ready = (
        hasattr(retriever, "df")
        and len(retriever.df) > 0
        and hasattr(retriever, "index")
    )

except Exception as exc:

    print("[FAIL] RAG initialization failed")
    print("Error:", exc)

if not check(
    rag_ready,
    "RAG retriever initialized successfully"
):
    failures += 1


# ===================================================================
# 4. EVIDENCE → RAG → EXPLANATION
# ===================================================================

print("\n" + "-" * 70)
print("4. EVIDENCE → RAG → EXPLANATION")
print("-" * 70)

PIPELINE_CASES = min(10, len(df))

pipeline_passes = 0


for position, (_, row) in enumerate(
    df.head(PIPELINE_CASES).iterrows(),
    start=1
):

    print(f"\nCase {position}/{PIPELINE_CASES}")

    print(f"Timestamp: {row['ts']}")
    print(f"Subsystem: {row['likely_subsystem']}")
    print(f"Priority : {row['impact_priority']}")

    case_pass = True

    # ---------------------------------------------------------------
    # A. Evidence-aware query
    # ---------------------------------------------------------------

    query = ""

    try:

        query = build_evidence_query(row)

        if not check(
            isinstance(query, str)
            and len(query.strip()) > 0,
            "Evidence-aware query generated"
        ):
            case_pass = False

    except Exception as exc:

        print("[FAIL] Evidence query generation failed")
        print("Error:", exc)

        case_pass = False


    # ---------------------------------------------------------------
    # B. RAG retrieval
    # ---------------------------------------------------------------

    retrieved = []

    if query and rag_ready:

        try:

            retrieved = retriever.retrieve(
                query,
                top_k=3,
                subsystem=row["likely_subsystem"]
            )

            if not check(
                len(retrieved) > 0,
                "Relevant sustainability knowledge retrieved"
            ):
                case_pass = False

        except Exception as exc:

            print("[FAIL] RAG retrieval failed")
            print("Error:", exc)

            case_pass = False

    else:

        print("[FAIL] RAG stage skipped")
        case_pass = False


    # ---------------------------------------------------------------
    # C. Grounded explanation
    # ---------------------------------------------------------------

    grounded = None

    if retrieved:

        try:

            grounded = build_grounded_explanation(
                row,
                retrieved
            )

            required_sections = [
                "summary",
                "investigation_actions",
                "caveats",
            ]

            sections_present = all(
                section in grounded
                for section in required_sections
            )

            if not check(
                sections_present,
                "Grounded explanation contains required sections"
            ):
                case_pass = False


            if not check(
                bool(
                    str(
                        grounded.get("summary", "")
                    ).strip()
                ),
                "Grounded summary is non-empty"
            ):
                case_pass = False


            if not check(
                len(
                    grounded.get(
                        "investigation_actions",
                        []
                    )
                ) > 0,
                "Safe investigation actions are present"
            ):
                case_pass = False


            if not check(
                len(
                    grounded.get(
                        "caveats",
                        []
                    )
                ) > 0,
                "Caveats are present"
            ):
                case_pass = False


            # -------------------------------------------------------
            # D. Unsupported claim detection
            #
            # IMPORTANT:
            # We inspect summary + actions only.
            #
            # Caveats are intentionally allowed to contain phrases
            # such as "confirmed equipment fault" because the system
            # explicitly says that the fault is NOT confirmed.
            # -------------------------------------------------------

            claim_text = (
                str(
                    grounded.get(
                        "summary",
                        ""
                    )
                )
                + " "
                + " ".join(
                    str(action)
                    for action in grounded.get(
                        "investigation_actions",
                        []
                    )
                )
            ).lower()


            forbidden_found = [
                phrase
                for phrase in FORBIDDEN_PHRASES
                if phrase in claim_text
            ]


            if not check(
                len(forbidden_found) == 0,
                "No unsupported certainty/savings claims detected"
            ):

                print(
                    "Forbidden phrases:",
                    forbidden_found
                )

                case_pass = False


            # -------------------------------------------------------
            # E. Safety caveat validation
            # -------------------------------------------------------

            caveat_text = " ".join(
                str(caveat)
                for caveat in grounded.get(
                    "caveats",
                    []
                )
            ).lower()


            required_safety_phrases = [
    		"not a confirmed equipment fault",
    		"modeled estimate",
    		"reviewed by an operator",
	    ]


            missing_safety = [
                phrase
                for phrase in required_safety_phrases
                if phrase not in caveat_text
            ]


            if not check(
                len(missing_safety) == 0,
                "Required safety caveats are present"
            ):

                print(
                    "Missing safety phrases:",
                    missing_safety
                )

                case_pass = False


        except Exception as exc:

            print("[FAIL] Grounded explanation failed")
            print("Error:", exc)

            case_pass = False


    # ---------------------------------------------------------------
    # Case result
    # ---------------------------------------------------------------

    if case_pass:

        pipeline_passes += 1
        print("Case result: PASS")

    else:

        print("Case result: FAIL")


# ===================================================================
# 5. FINAL QUALITY GATE
# ===================================================================

print("\n" + "=" * 70)
print("QUALITY GATE RESULTS")
print("=" * 70)

print(
    f"Anomaly records available : {len(df):,}"
)

print(
    f"Cases evaluated           : {PIPELINE_CASES}"
)

print(
    f"Cases passed              : {pipeline_passes}"
)


pipeline_success = (
    PIPELINE_CASES > 0
    and pipeline_passes == PIPELINE_CASES
)


if not check(
    pipeline_success,
    "All selected anomalies passed the end-to-end pipeline"
):
    failures += 1


# ===================================================================
# FINAL STATUS
# ===================================================================

print("\n" + "=" * 70)

if failures == 0:

    print("FINAL RESULT: PASS")
    print(
        "GREENGUARD END-TO-END QUALITY GATE PASSED"
    )

else:

    print("FINAL RESULT: FAIL")
    print(
        f"Quality-gate failures: {failures}"
    )

print("=" * 70)


if failures > 0:
    raise SystemExit(1)