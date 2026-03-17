import pandas as pd

# Load dataset (fix encoding issue)
df = pd.read_csv("data/dataset.csv", encoding="latin-1")

# Print dataset size
print("Dataset Shape:")
print(df.shape)

# Show first 5 rows
print("\nFirst 5 rows:")
print(df.head())