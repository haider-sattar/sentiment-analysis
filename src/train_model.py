"""
Train a sentiment classifier.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

print("1. Loading features...")
X_train, X_test, y_train, y_test = joblib.load('data/features.joblib')

print("2. Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print("3. Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Model Accuracy: {accuracy:.1%}")

print("4. Saving model...")
joblib.dump(model, 'models/sentiment_model.joblib')
print("✅ Model saved: models/sentiment_model.joblib")