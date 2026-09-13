from ingestion.loaders import load_datacenter_data


df = load_datacenter_data()

print("\nData loaded successfully!")
print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)