from ingestion.loaders import load_datacenter_data
from preprocessing.features import create_sustainability_features


# Load raw operational data
df = load_datacenter_data()

# Create sustainability features
df = create_sustainability_features(df)


print("\nFeature engineering successful!")

print("\nNew columns:")
print(df.columns.tolist())

print("\nSelected sustainability metrics:")
print(
    df[
        [
            "timestamp",
            "it_load_kw",
            "cooling_load_kw",
            "cooling_it_ratio",
            "pue",
            "water_per_kwh",
            "temperature_stress",
            "high_utilization",
            "cooling_stress",
        ]
    ].head(10)
)