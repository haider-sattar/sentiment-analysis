"""
Optimize TF-IDF features to boost model accuracy.
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("🔄 OPTIMIZING TF-IDF FEATURES")
print("=" * 50)

# Load cleaned data
df = pd.read_csv('data/cleaned_reviews.csv')

# Different TF-IDF configurations to test
configs = [
    {'name': 'Baseline', 'max_features': 5000, 'ngram_range': (1, 1), 'min_df': 1, 'max_df': 1.0},
    {'name': '+ Bigrams', 'max_features': 5000, 'ngram_range': (1, 2), 'min_df': 1, 'max_df': 1.0},
    {'name': '+ Trigrams', 'max_features': 5000, 'ngram_range': (1, 3), 'min_df': 1, 'max_df': 1.0},
    {'name': 'Filter Rare', 'max_features': 8000, 'ngram_range': (1, 2), 'min_df': 5, 'max_df': 0.7},
    {'name': 'Aggressive Filter', 'max_features': 10000, 'ngram_range': (1, 3), 'min_df': 10, 'max_df': 0.5},
]

results = []

for config in configs:
    print(f"\n🔧 Testing: {config['name']}")
    print(f"   Params: {config}")
    
    # Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=config['max_features'],
        ngram_range=config['ngram_range'],
        min_df=config['min_df'],
        max_df=config['max_df'],
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df['cleaned_review'])
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    results.append({
        'Config': config['name'],
        'Features': X.shape[1],
        'Accuracy': f"{acc:.2%}",
        'Params': f"ngrams={config['ngram_range']}, min_df={config['min_df']}, max_df={config['max_df']}"
    })
    
    print(f"   ✓ Features: {X.shape[1]}")
    print(f"   ✓ Accuracy: {acc:.2%}")

# Display results
print("\n" + "=" * 50)
print("📈 OPTIMIZATION RESULTS")
print("=" * 50)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Save best configuration
best_idx = results_df['Accuracy'].str.rstrip('%').astype(float).idxmax()
best_config = configs[best_idx]

print(f"\n🏆 Best configuration: {best_config['name']}")
print(f"   Accuracy: {results_df.iloc[best_idx]['Accuracy']}")

# Retrain with best config and save
print("\n💾 Saving optimized features...")
best_vectorizer = TfidfVectorizer(
    max_features=best_config['max_features'],
    ngram_range=best_config['ngram_range'],
    min_df=best_config['min_df'],
    max_df=best_config['max_df'],
    stop_words='english'
)

X_best = best_vectorizer.fit_transform(df['cleaned_review'])
X_train_best, X_test_best, y_train_best, y_test_best = train_test_split(
    X_best, y, test_size=0.2, random_state=42, stratify=y
)

# Train final model
final_model = LogisticRegression(max_iter=1000, random_state=42)
final_model.fit(X_train_best, y_train_best)

# Save everything
joblib.dump(best_vectorizer, 'models/optimized_vectorizer.joblib')
joblib.dump(final_model, 'models/optimized_model.joblib')
joblib.dump((X_train_best, X_test_best, y_train_best, y_test_best), 'data/optimized_features.joblib')

print("✅ Optimization complete!")
print(f"   Saved: models/optimized_vectorizer.joblib")
print(f"   Saved: models/optimized_model.joblib")