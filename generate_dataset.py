from sklearn.datasets import load_breast_cancer
import pandas as pd
import os

# Load full dataset (569 rows)
data = load_breast_cancer()

# Convert to DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# Create data folder if not exists
os.makedirs("data", exist_ok=True)

# Save as CSV
df.to_csv("data/breast_cancer.csv", index=False)

print("Full dataset generated successfully!")
print("Shape:", df.shape)
print(df["target"].value_counts())
