import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your data
df = pd.read_csv('data/sentiment_data.csv')

# 2. Basic exploration
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# 3. Check sentiment distribution
sentiment_counts = df['sentiment'].value_counts()
print(f"\nSentiment distribution:\n{sentiment_counts}")
print(f"Positive percentage: {sentiment_counts['positive']/len(df)*100:.1f}%")