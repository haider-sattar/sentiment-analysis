import joblib
import os

print("Current directory:", os.getcwd())
print("Files in models/:", os.listdir('models') if os.path.exists('models') else "NO models folder")

try:
    v = joblib.load('models/optimized_vectorizer.joblib')
    m = joblib.load('models/optimized_model.joblib')
    print("✅ SUCCESS: Models loaded!")
    print(f"Vectorizer has {len(v.vocabulary_)} features")
except Exception as e:
    print(f"❌ ERROR: {e}")