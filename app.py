import sys
from pathlib import Path

import streamlit as st
import pandas as pd


# ==========================================================
# PROJECT PATH
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# GREEN GUARD MODULES
# ==========================================================

from rag.retriever import SustainabilityRetriever
from ai.evidence_rag_pipeline import build_evidence_query
from ai.grounded_explainer import build_grounded_explanation


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Data-Center GreenGuard",
    page_icon="🌱",
    layout="wide"
)


# ==========================================================
# PATHS
# ==========================================================

DASHBOARD_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard_daily_summary.parquet"
)

FINAL_ANOMALY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "greenguard_final_anomalies.parquet"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_dashboard_summary():

    return pd.read_parquet(
        DASHBOARD_SUMMARY_PATH
    )


@st.cache_data
def load_final_anomalies():

    return pd.read_parquet(
        FINAL_ANOMALY_PATH
    )


@st.cache_resource
def load_rag():

    return SustainabilityRetriever()


# ==========================================================
# SAFE HELPERS
# ==========================================================

def safe_float(value, default=None):

    try:

        if pd.isna(value):
            return default

        return float(value)

    except Exception:

        return default


def format_kw(value, decimals=2):

    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"{number:,.{decimals}f} kW"


def format_number(value, decimals=2):

    number = safe_float(value)

    if number is None:
        return "N/A"

    return f"{number:,.{decimals}f}"


def sort_events(df):

    """
    Sort events safely.

    Priority:
    1. impact_score
    2. estimated_excess_power_kw
    3. anomaly_score
    4. timestamp
    """

    result = df.copy()

    if "impact_score" in result.columns:

        result = result.sort_values(
            "impact_score",
            ascending=False
        )

    elif "estimated_excess_power_kw" in result.columns:

        result = result.sort_values(
            "estimated_excess_power_kw",
            ascending=False
        )

    elif "anomaly_score" in result.columns:

        result = result.sort_values(
            "anomaly_score",
            ascending=False
        )

    elif "ts" in result.columns:

        result = result.sort_values(
            "ts",
            ascending=False
        )

    return result.reset_index(drop=True)


# ==========================================================
# HEADER
# ==========================================================

st.title(
    "🌱 Data-Center GreenGuard"
)

st.markdown(
    """
    ### AI-powered sustainability intelligence for data centers

    GreenGuard analyzes real operational telemetry to detect
    unusual behavior, estimate potential sustainability impact,
    identify likely contributing subsystems, retrieve relevant
    sustainability knowledge, and generate grounded investigation
    guidance.
    """
)

st.info(
    "Decision-support system: recommendations are investigation "
    "steps and require human/operator review. GreenGuard does not "
    "automatically control data-center equipment."
)


# ==========================================================
# LOAD SYSTEM
# ==========================================================

try:

    dashboard = load_dashboard_summary()

    anomalies = load_final_anomalies()

    retriever = load_rag()

except FileNotFoundError as e:

    st.error(
        "Required GreenGuard data file was not found."
    )

    st.code(
        str(e)
    )

    st.info(
        "Make sure dashboard_daily_summary.parquet and "
        "greenguard_final_anomalies.parquet exist."
    )

    st.stop()

except Exception as e:

    st.error(
        "Unable to initialize GreenGuard."
    )

    st.code(
        str(e)
    )

    st.stop()


# ==========================================================
# NORMALIZE TIMESTAMPS
# ==========================================================

if "date" in dashboard.columns:

    dashboard["date"] = pd.to_datetime(
        dashboard["date"],
        errors="coerce"
    )


if "ts" in anomalies.columns:

    anomalies["ts"] = pd.to_datetime(
        anomalies["ts"],
        errors="coerce"
    )


# ==========================================================
# BASIC VALIDATION
# ==========================================================

required_anomaly_columns = [
    "ts",
    "pue",
    "it_power_kw",
    "impact_priority",
    "likely_subsystem"
]

missing_columns = [
    column
    for column in required_anomaly_columns
    if column not in anomalies.columns
]

if missing_columns:

    st.error(
        "Final anomaly dataset is missing required columns."
    )

    st.code(
        ", ".join(missing_columns)
    )

    st.stop()


# ==========================================================
# DATASET COUNTS
# ==========================================================

total_anomalies = len(anomalies)


critical_count = int(
    (
        anomalies["impact_priority"]
        == "Critical"
    ).sum()
)


high_count = int(
    (
        anomalies["impact_priority"]
        == "High"
    ).sum()
)


medium_count = int(
    (
        anomalies["impact_priority"]
        == "Medium"
    ).sum()
)


low_count = int(
    (
        anomalies["impact_priority"]
        == "Low"
    ).sum()
)


# ==========================================================
# TELEMETRY RECORD COUNT
# ==========================================================

if "records" in dashboard.columns:

    total_records = int(
        pd.to_numeric(
            dashboard["records"],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

else:

    # Validated telemetry size from the GreenGuard pipeline.
    total_records = 4_432_570


# ==========================================================
# BASELINE PUE
# ==========================================================

if "baseline_pue" in anomalies.columns:

    baseline_values = pd.to_numeric(
        anomalies["baseline_pue"],
        errors="coerce"
    ).dropna()

    if len(baseline_values) > 0:

        baseline_pue = float(
            baseline_values.median()
        )

    else:

        baseline_pue = None

else:

    baseline_pue = None


if baseline_pue is None:

    if "pue_median" in dashboard.columns:

        dashboard_pue = pd.to_numeric(
            dashboard["pue_median"],
            errors="coerce"
        ).dropna()

        if len(dashboard_pue) > 0:

            baseline_pue = float(
                dashboard_pue.median()
            )

        else:

            baseline_pue = 1.0665

    else:

        baseline_pue = 1.0665


# ==========================================================
# EXCESS POWER METRICS
# ==========================================================

if "estimated_excess_power_kw" in anomalies.columns:

    excess_values = pd.to_numeric(
        anomalies["estimated_excess_power_kw"],
        errors="coerce"
    )

    positive_excess_values = (
        excess_values[
            excess_values > 0
        ]
        .dropna()
    )

    positive_excess_count = len(
        positive_excess_values
    )

    if positive_excess_count > 0:

        avg_excess_power = float(
            positive_excess_values.mean()
        )

        max_excess_power = float(
            positive_excess_values.max()
        )

    else:

        avg_excess_power = 0.0
        max_excess_power = 0.0

else:

    positive_excess_count = 0
    avg_excess_power = 0.0
    max_excess_power = 0.0


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header(
    "GreenGuard Controls"
)

st.sidebar.caption(
    "Filters operate only on the 174 validated "
    "sustainability events — not the 4.4M telemetry rows."
)


# ----------------------------------------------------------
# Minimum PUE
# ----------------------------------------------------------

min_pue = st.sidebar.number_input(
    "Minimum PUE",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.01
)


# ----------------------------------------------------------
# Impact Priority
# ----------------------------------------------------------

priority_options = [
    "All",
    "Critical",
    "High",
    "Medium",
    "Low"
]

selected_priority = st.sidebar.selectbox(
    "Impact Priority",
    priority_options
)


# ----------------------------------------------------------
# Subsystem
# ----------------------------------------------------------

subsystem_values = (
    anomalies["likely_subsystem"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

subsystem_options = (
    ["All"]
    + sorted(subsystem_values)
)

selected_subsystem = st.sidebar.selectbox(
    "Subsystem",
    subsystem_options
)


# ==========================================================
# FILTER VALIDATED EVENTS ONLY
# ==========================================================

filtered = anomalies.copy()


# ----------------------------------------------------------
# PUE FILTER
# ----------------------------------------------------------

filtered = filtered[
    pd.to_numeric(
        filtered["pue"],
        errors="coerce"
    ) >= min_pue
]


# ----------------------------------------------------------
# PRIORITY FILTER
# ----------------------------------------------------------

if selected_priority != "All":

    filtered = filtered[
        filtered["impact_priority"]
        == selected_priority
    ]


# ----------------------------------------------------------
# SUBSYSTEM FILTER
# ----------------------------------------------------------

if selected_subsystem != "All":

    filtered = filtered[
        filtered["likely_subsystem"]
        == selected_subsystem
    ]


# ----------------------------------------------------------
# Stable ordering
# ----------------------------------------------------------

filtered = sort_events(
    filtered
)


# ==========================================================
# SUSTAINABILITY OVERVIEW
# ==========================================================

st.subheader(
    "📊 Sustainability Overview"
)


col1, col2, col3, col4, col5, col6 = st.columns(6)


col1.metric(
    "Telemetry Records",
    f"{total_records:,}"
)


col2.metric(
    "Validated Events",
    f"{total_anomalies:,}"
)


col3.metric(
    "Critical",
    f"{critical_count:,}"
)


col4.metric(
    "High",
    f"{high_count:,}"
)


col5.metric(
    "Baseline PUE",
    f"{baseline_pue:.4f}"
)


col6.metric(
    "Avg Positive Excess",
    f"{avg_excess_power:,.2f} kW"
)


# ==========================================================
# FILTER STATUS
# ==========================================================

st.caption(
    f"Showing {len(filtered):,} of "
    f"{total_anomalies:,} validated sustainability events "
    f"after applying the selected filters."
)


# ==========================================================
# PRIORITY DISTRIBUTION
# ==========================================================

st.divider()

st.subheader(
    "🚦 Event Priority Distribution"
)


priority_counts = (
    anomalies[
        "impact_priority"
    ]
    .value_counts()
    .reindex(
        [
            "Critical",
            "High",
            "Medium",
            "Low"
        ],
        fill_value=0
    )
)


st.bar_chart(
    priority_counts
)


# ==========================================================
# REAL OPERATIONAL DATASET
# ==========================================================

st.divider()

st.subheader(
    "📡 Real Operational Dataset"
)

st.write(
    "Source: NLR ESIF public data-center "
    "power/PUE operational dataset."
)

st.write(
    f"Telemetry records: "
    f"{total_records:,}"
)


if (
    "date" in dashboard.columns
    and dashboard["date"].notna().any()
):

    min_date = dashboard["date"].min()

    max_date = dashboard["date"].max()

    st.write(
        f"Time range: "
        f"{min_date} → {max_date}"
    )


# ==========================================================
# PUE TREND
# ==========================================================

st.divider()

st.subheader(
    "📈 PUE Trend"
)


if {
    "date",
    "pue_median"
}.issubset(
    dashboard.columns
):

    trend = (
        dashboard[
            [
                "date",
                "pue_median"
            ]
        ]
        .dropna()
        .sort_values("date")
        .set_index("date")
    )

    st.line_chart(
        trend
    )

else:

    st.warning(
        "PUE trend data is unavailable."
    )


# ==========================================================
# POWER & COOLING
# ==========================================================

st.subheader(
    "⚡ Power & Cooling"
)


required_power_columns = {
    "date",
    "it_power_mean",
    "cooling_power_mean"
}


if required_power_columns.issubset(
    dashboard.columns
):

    power_df = (
        dashboard[
            [
                "date",
                "it_power_mean",
                "cooling_power_mean"
            ]
        ]
        .dropna()
        .sort_values("date")
        .set_index("date")
    )

    st.line_chart(
        power_df
    )

else:

    st.warning(
        "Power and cooling trend data is unavailable."
    )


# ==========================================================
# FILTERED SUSTAINABILITY EVENTS
# ==========================================================

st.divider()

st.subheader(
    "🚨 Sustainability Events"
)


if len(filtered) == 0:

    st.warning(
        "No validated sustainability events match "
        "the selected filters."
    )

else:

    event_columns = [
        "ts",
        "impact_priority",
        "likely_subsystem",
        "pue",
        "it_power_kw",
        "estimated_excess_power_kw",
        "evidence_class"
    ]

    available_event_columns = [
        column
        for column in event_columns
        if column in filtered.columns
    ]

    event_table = filtered[
        available_event_columns
    ].copy()


    # ------------------------------------------------------
    # User-friendly column names
    # ------------------------------------------------------

    event_table = event_table.rename(
        columns={
            "ts": "Timestamp",
            "impact_priority": "Priority",
            "likely_subsystem": "Subsystem",
            "pue": "PUE",
            "it_power_kw": "IT Load (kW)",
            "estimated_excess_power_kw":
                "Estimated Excess Power (kW)",
            "evidence_class": "Evidence Class"
        }
    )


    st.dataframe(
        event_table.head(100),
        use_container_width=True,
        height=450
    )


# ==========================================================
# EVENT EXPLORER
# ==========================================================

st.divider()

st.subheader(
    "🔎 Sustainability Event Explorer"
)


if len(filtered) == 0:

    st.info(
        "Change the sidebar filters to select "
        "a validated sustainability event."
    )

else:

    # ------------------------------------------------------
    # Create event labels
    # ------------------------------------------------------

    event_labels = []


    for i, row in filtered.iterrows():

        timestamp = str(
            row.get(
                "ts",
                "Unknown"
            )
        )

        priority = str(
            row.get(
                "impact_priority",
                "Unknown"
            )
        )

        subsystem = str(
            row.get(
                "likely_subsystem",
                "Unknown"
            )
        )

        pue_value = safe_float(
            row.get(
                "pue",
                None
            )
        )


        if pue_value is not None:

            pue_text = (
                f"{pue_value:.3f}"
            )

        else:

            pue_text = "N/A"


        event_labels.append(
            f"{i + 1}. "
            f"{timestamp} | "
            f"{priority} | "
            f"{subsystem} | "
            f"PUE {pue_text}"
        )


    # ------------------------------------------------------
    # Event selection
    # ------------------------------------------------------

    selected_event_index = st.selectbox(
        "Select an event to investigate",
        range(
            len(event_labels)
        ),
        format_func=lambda x:
            event_labels[x]
    )


    # ------------------------------------------------------
    # SELECTED ROW
    # ------------------------------------------------------

    selected_row = filtered.iloc[
        selected_event_index
    ]


    # ======================================================
    # BUILD EVIDENCE → RAG → EXPLANATION
    # ======================================================

    try:

        evidence_query = build_evidence_query(
            selected_row
        )


        retrieved_knowledge = retriever.retrieve(
            evidence_query,
            top_k=3,
            subsystem=selected_row.get(
                "likely_subsystem",
                None
            )
        )


        explanation = build_grounded_explanation(
            selected_row,
            retrieved_knowledge
        )


    except Exception as e:

        st.error(
            "Unable to generate grounded analysis "
            "for the selected event."
        )

        st.code(
            str(e)
        )

        explanation = {
            "summary": (
                "Grounded analysis could not be generated "
                "for this event."
            ),
            "observed_evidence": {},
            "retrieved_knowledge": [],
            "investigation_actions": [],
            "caveats": [
                "RAG analysis was unavailable for this event.",
                "Operator review is required."
            ]
        }

        evidence_query = ""


    # ======================================================
    # OBSERVED EVIDENCE
    # ======================================================

    st.subheader(
        "📋 Observed Evidence"
    )


    evidence_cols = st.columns(4)


    # ------------------------------------------------------
    # PUE
    # ------------------------------------------------------

    pue_value = safe_float(
        selected_row.get(
            "pue",
            None
        )
    )


    evidence_cols[0].metric(
        "PUE",
        (
            f"{pue_value:.4f}"
            if pue_value is not None
            else "N/A"
        )
    )


    # ------------------------------------------------------
    # IT LOAD
    # ------------------------------------------------------

    it_value = safe_float(
        selected_row.get(
            "it_power_kw",
            None
        )
    )


    evidence_cols[1].metric(
        "IT Load",
        (
            f"{it_value:,.2f} kW"
            if it_value is not None
            else "N/A"
        )
    )


    # ------------------------------------------------------
    # ESTIMATED EXCESS POWER
    # ------------------------------------------------------

    excess_value = safe_float(
        selected_row.get(
            "estimated_excess_power_kw",
            None
        )
    )


    evidence_cols[2].metric(
        "Estimated Excess Power",
        (
            f"{excess_value:,.2f} kW"
            if excess_value is not None
            else "N/A"
        )
    )


    # ------------------------------------------------------
    # PRIORITY
    # ------------------------------------------------------

    priority_text = str(
        selected_row.get(
            "impact_priority",
            "N/A"
        )
    )


    evidence_cols[3].metric(
        "Impact Priority",
        priority_text
    )


    # ======================================================
    # SUBSYSTEM HYPOTHESIS
    # ======================================================

    subsystem_text = str(
        selected_row.get(
            "likely_subsystem",
            "No dominant subsystem"
        )
    )


    st.write(
        "**Likely contributing subsystem:** "
        f"{subsystem_text}"
    )


    st.caption(
        "This is an investigation hypothesis, "
        "not a confirmed equipment fault."
    )


    # ======================================================
    # QUANTITATIVE EVIDENCE
    # ======================================================

    st.subheader(
        "📐 Quantitative Evidence"
    )


    quantitative_columns = [
        "pue_deviation",
        "hvac_deviation_ratio",
        "pump_deviation_ratio",
        "cooling_deviation_ratio",
        "estimated_excess_power_kw"
    ]


    available_quantitative = [
        column
        for column in quantitative_columns
        if column in selected_row.index
    ]


    if available_quantitative:

        quantitative_data = {
            column: [
                selected_row.get(
                    column,
                    None
                )
            ]
            for column in available_quantitative
        }


        quantitative_df = pd.DataFrame(
            quantitative_data
        )


        st.dataframe(
            quantitative_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Quantitative evidence fields are unavailable "
            "for this event."
        )


#======================================================
    # GROUNDED AI ANALYSIS
#======================================================

    st.divider()

    st.subheader(
        "🧠 Grounded AI Sustainability Analysis"
    )


  # ------------------------------------------------------
    # Grounded Explanation
    # ------------------------------------------------------

    st.markdown(
        "### 📝 Grounded Explanation"
    )


    st.write(
        explanation.get(
            "summary",
            "No grounded explanation available."
        )
    )


    # ======================================================
    # RETRIEVED KNOWLEDGE
    # ======================================================

    st.markdown(
        "### 📚 Retrieved Sustainability Knowledge"
    )


    knowledge_items = explanation.get(
        "retrieved_knowledge",
        []
    )


    if knowledge_items:

        for i, item in enumerate(
            knowledge_items,
            start=1
        ):

            topic = item.get(
                "topic",
                "Unknown"
            )

            similarity = item.get(
                "similarity",
                None
            )

            condition = item.get(
                "condition",
                ""
            )

            knowledge_explanation = item.get(
                "explanation",
                ""
            )

            recommended_action = item.get(
                "recommended_action",
                ""
            )

            sustainability_area = item.get(
                "sustainability_area",
                ""
            )


            with st.expander(
                f"{i}. {topic}"
            ):

                if similarity is not None:

                    st.write(
                        f"**Retrieval similarity:** "
                        f"{similarity:.4f}"
                    )


                if sustainability_area:

                    st.write(
                        f"**Sustainability area:** "
                        f"{sustainability_area}"
                    )


                if condition:

                    st.write(
                        f"**Relevant condition:** "
                        f"{condition}"
                    )


                if knowledge_explanation:

                    st.write(
                        f"**Explanation:** "
                        f"{knowledge_explanation}"
                    )


                if recommended_action:

                    st.write(
                        f"**Recommended investigation:** "
                        f"{recommended_action}"
                    )

    else:

        st.info(
            "No sustainability knowledge was retrieved."
        )


    # ======================================================
    # INVESTIGATION ACTIONS
    # ======================================================

    st.markdown(
        "### 🔧 Recommended Investigation"
    )


    actions = explanation.get(
        "investigation_actions",
        []
    )


    if actions:

        for action in actions:

            st.write(
                f"• {action}"
            )

    else:

        st.info(
            "No investigation actions were generated."
        )


    # ======================================================
    # SAFETY BOUNDARY
    # ======================================================

    st.warning(
        "Human review required. GreenGuard provides "
        "investigation guidance, not automatic "
        "equipment-control commands."
    )


    # ======================================================
    # EVIDENCE CAVEATS
    # ======================================================

    st.markdown(
        "### ⚠️ Evidence Caveats"
    )


    caveats = explanation.get(
        "caveats",
        []
    )


    if caveats:

        for caveat in caveats:

            st.write(
                f"• {caveat}"
            )

    else:

        st.write(
            "• Results should be reviewed by a "
            "data-center operator before action."
        )


    # ======================================================
    # EVIDENCE QUERY
    # ======================================================

    with st.expander(
        "🔍 View Evidence-Aware Retrieval Query"
    ):

        if evidence_query:

            st.code(
                evidence_query
            )

        else:

            st.info(
                "Evidence query was unavailable."
            )


# ==========================================================
# TRACEABILITY
# ==========================================================

st.divider()

st.subheader(
    "🔗 GreenGuard Decision Traceability"
)


trace = [
    "Real operational telemetry",
    "ML anomaly detection",
    "Quantitative sustainability evidence",
    "Potential sustainability impact estimation",
    "Subsystem hypothesis",
    "Sustainability knowledge retrieval",
    "Grounded explanation",
    "Human investigation"
]


trace_cols = st.columns(4)


for i, step in enumerate(
    trace,
    start=1
):

    column = trace_cols[
        (i - 1) % 4
    ]

    column.write(
        f"**{i}.** {step}"
    )


# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    "Data-Center GreenGuard | "
    "Real public operational data + ML + RAG + "
    "grounded sustainability decision support"
)