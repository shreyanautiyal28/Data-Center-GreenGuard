import pandas as pd


def load_datacenter_data(
    path="data/raw/datacenter_operations.csv"
):
    """
    Load and validate Data-Center GreenGuard operational data.
    """

    df = pd.read_csv(path)

    required_columns = [
        "timestamp",
        "facility_id",
        "it_load_kw",
        "cooling_load_kw",
        "total_energy_kwh",
        "water_usage_l",
        "temperature_c",
        "humidity_pct",
        "server_utilization_pct",
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    numeric_columns = [
        "it_load_kw",
        "cooling_load_kw",
        "total_energy_kwh",
        "water_usage_l",
        "temperature_c",
        "humidity_pct",
        "server_utilization_pct",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.sort_values("timestamp")

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    return df