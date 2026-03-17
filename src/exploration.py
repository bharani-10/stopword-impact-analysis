import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/clean_dataset.csv")

# Dataset size
print("Dataset shape:", df.shape)

# Show first rows
print("\nSample rows:")
print(df.head())

# Class distribution
print("\nClass Distribution:")
print(df['label'].value_counts())

# Visualization
sns.countplot(x='label', data=df)
plt.title("Spam vs Ham Distribution")
plt.show()