import pandas as pd

# Load dataset
df = pd.read_csv("data/dataset.csv", encoding="latin-1")

# Keep only first two columns
df = df[['v1','v2']]

# Rename columns
df.columns = ['label','text']

# Show first rows
print("Dataset Preview:")
print(df.head())

# Save cleaned dataset
df.to_csv("data/clean_dataset.csv", index=False)

print("\nClean dataset saved successfully!")