"""
Convert cleaned text to TF-IDF features.
Run: python src/extract_features.py
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("1. Loading cleaned data...")
df = pd.read_csv('data/cleaned_reviews.csv')
print(f"   Loaded {len(df)} cleaned reviews")

print("2. Creating TF-IDF features...")
# Initialize TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,      # Keep top 5000 words (controls model size)
    stop_words='english',   # Remove common words
    ngram_range=(1, 2)      # Use single words + word pairs
)

# Fit on all text, then transform
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment'].map({'positive': 1, 'negative': 0})  # Convert to 1/0

print(f"   Created {X.shape[1]} features (unique words/word-pairs)")

print("3. Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

print("4. Saving features and vectorizer...")
# Save the vectorizer to reuse later
joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')

# Save the split data
joblib.dump((X_train, X_test, y_train, y_test), 'data/features.joblib')

print("✅ Feature extraction complete!")
print(f"   Vectorizer saved: models/tfidf_vectorizer.joblib")
print(f"   Features saved: data/features.joblib")