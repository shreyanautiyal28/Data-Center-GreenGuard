# Data-Center GreenGuard

**AI-powered sustainability intelligence and decision support for data-center operations**

GreenGuard is an academic prototype that analyzes data-center energy and cooling telemetry, flags unusual operating patterns, estimates potential excess power, and retrieves relevant operational guidance to help a human investigate.

> **Project status:** Offline research prototype. It does not connect to live data-center systems, confirm equipment faults, or automatically control equipment.

## The problem

Data centers use substantial energy for IT equipment and supporting infrastructure such as cooling. Unusual power or cooling patterns can indicate opportunities for investigation, but operators need context before deciding what action—if any—is appropriate.

GreenGuard explores how anomaly detection and retrieval-augmented generation (RAG) can support that investigation using public telemetry and controlled tests.

## What it does

* Loads and prepares public data-center telemetry.
* Engineers sustainability-related features, including PUE and subsystem power ratios.
* Uses a rolling historical baseline and an Isolation Forest model to flag unusual observations.
* Estimates potential excess power relative to a modeled baseline.
* Retrieves relevant guidance from a curated knowledge base.
* Presents findings and suggested investigation steps through a Streamlit interface.
* Includes evaluation scripts and controlled anomaly tests.

Recommendations are decision support for a human operator—not instructions to automatically change equipment settings.

## Architecture

```text
Public telemetry
      ↓
Data cleaning and validation
      ↓
Sustainability feature engineering
      ↓
Leakage-safe rolling baseline
      ↓
Anomaly detection and modeled impact
      ↓
RAG evidence retrieval and explanation
      ↓
Streamlit dashboard
      ↓
Human review and decision
```

## Data

The project uses public data from the National Renewable Energy Laboratory (NREL), now part of the National Laboratory of the Rockies (NLR), including the ESIF PUE dataset.

* Dataset: [ESIF PUE data submission](https://data.nlr.gov/submissions/300)
* Background: [Measuring data-center efficiency using PUE](https://www.nlr.gov/computational-science/measuring-efficiency-pue)

The raw dataset is not included in this repository. Download it from the source and place the file at:

```text
data/raw/public/esif.influx.buildingData.PUE.combined.parquet
```

The data contains missing values and operating variation. The project applies cleaning, plausibility checks, and feature engineering before analysis. Review the source dataset and project scripts for the exact processing assumptions.

## Technology

* Python
* Pandas and PyArrow for data processing
* scikit-learn for anomaly detection
* Sentence Transformers and FAISS for retrieval
* Streamlit for the user interface

See `requirements.txt` for the project dependencies.

## Setup

Run these commands from the project directory in PowerShell.

```powershell
cd F:\GreenGuard_AI

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use the project’s approved Python environment setup instructions or activate the environment from a terminal that permits it.

## Run the application

```powershell
streamlit run app.py
```

If the application reports that a dataset or generated artifact is missing, confirm that you downloaded the required public data and ran the relevant preparation scripts. Check the script paths and instructions in the repository before running them.

## Evaluation and results

The project includes scripts for baseline checks, model comparisons, controlled anomaly tests, RAG evaluation, and end-to-end validation.

Selected prototype results:

* The V2 Isolation Forest flagged **174 of 8,698 modeling rows (2.00%)** in the evaluated rolling-baseline sample.
* In controlled injected-anomaly tests, detection varied by subsystem; cooling anomalies were notably harder to detect than HVAC and pump anomalies.
* The RAG knowledge base contains a small curated set of topics. Its evaluation results apply to those test cases, not to all possible data-center questions.

These are prototype evaluation results—not production accuracy or measured operational savings. Controlled injections are simulated tests and do not establish real-world fault-detection performance.

## Important limitations

* The prototype analyzes prepared public data; it is not connected to live telemetry.
* An anomaly is a flagged observation, not a confirmed fault.
* Modeled excess power is an estimate relative to a baseline, not measured energy savings.
* Root-cause labels are hypotheses to guide investigation, not verified diagnoses.
* Controlled anomaly tests do not replace evaluation against labeled real-world incidents.
* The current RAG evaluation is limited by the size and coverage of its curated knowledge base.
* Water-efficiency and security-anomaly monitoring are not claimed as validated capabilities unless supported by data and evaluation in the project.

## Repository structure

```text
ai/               Explanation and RAG pipeline components
analysis/          Analysis and dataset-building scripts
api/               API-related code
cleaning/          Data-cleaning scripts
evaluation/        Model and pipeline evaluation scripts
explanation/       Explanation engine
impact/            Modeled impact calculations
ingestion/         Data-loading components
models/            Anomaly-detection models
preprocessing/     Feature preparation
rag/               Retrieval components
recommendations/   Recommendation logic
risk/              Severity and risk logic
root_cause/        Root-cause hypothesis logic
sustainability/    Sustainability features and baselines
tests/             Tests
validation/        Data plausibility checks
app.py             Streamlit application entry point
requirements.txt   Python dependencies
```

## Responsible use

GreenGuard is intended for learning, research, and human-reviewed investigation. Do not use its outputs as the sole basis for operational changes or safety-critical decisions.

## Author

**Shreya Nautiyal**
Veer Madho Singh Bhandari Uttarakhand Technical University, Dehradun, India

## License

No license has been specified yet. Unless a license is added, reuse is subject to the applicable copyright rules.
