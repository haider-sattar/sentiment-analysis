"""
Systematic model comparison for sentiment analysis.
Run: python src/compare_models.py
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

print("📊 MODEL COMPARISON FOR SENTIMENT ANALYSIS")
print("=" * 50)

# 1. Load features
print("\n1. Loading preprocessed features...")
X_train, X_test, y_train, y_test = joblib.load('data/features.joblib')
print(f"   Training: {X_train.shape[0]} samples, {X_train.shape[1]} features")
print(f"   Testing:  {X_test.shape[0]} samples")

# 2. Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Linear SVM': LinearSVC(random_state=42, max_iter=5000),
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

# 3. Train and evaluate each model
results = []

for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    
    # Time the training
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': f"{acc:.2%}",
        'F1-Score': f"{f1:.3f}",
        'Train Time (s)': f"{train_time:.1f}",
        'Parameters': str(model.get_params())[:80] + "..."
    })
    
    print(f"   ✓ Accuracy: {acc:.2%}")
    print(f"   ✓ F1-Score: {f1:.3f}")
    print(f"   ✓ Time: {train_time:.1f}s")

# 4. Display comparison table
print("\n" + "=" * 50)
print("🏆 MODEL COMPARISON RESULTS")
print("=" * 50)

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))

# 5. Save the best model
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n💾 Saving best model: {best_model_name}")
joblib.dump(best_model, f'models/best_model.joblib')
print(f"   Saved to: models/best_model.joblib")

# 6. Detailed report for best model
print(f"\n📈 Detailed report for {best_model_name}:")
y_pred_best = best_model.predict(X_test)
print(classification_report(y_test, y_pred_best, 
                           target_names=['Negative', 'Positive']))

print("✅ Model comparison complete!")