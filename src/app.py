import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd

# ==========================================================
#  PAGE CONFIG & STYLING
# ==========================================================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="📊",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: #1E3A8A;
        font-size: 2.4rem;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }
    .subtext {
        text-align: center;
        color: #475569;
        font-size: 1.0rem;
        margin-bottom: 1rem;
    }
    .feature-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #1E40AF;
        padding: 5px 14px;
        border-radius: 20px;
        margin: 3px;
        font-size: 0.9rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
#  LOAD OPTIMIZED MODEL
# ==========================================================
@st.cache_resource
def load_model():
    try:
        vectorizer = joblib.load("src/optimized_vectorizer.joblib")
        model = joblib.load("src/optimized_model.joblib")
        return vectorizer, model, True
    except Exception as e:
        st.warning(f"⚠️ Could not load optimized model. Using fallback model. ({e})")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        X = vectorizer.fit_transform(["good", "bad"])
        model.fit(X, [1, 0])
        return vectorizer, model, False

vectorizer, model, is_optimized = load_model()

# ==========================================================
#  HEADER
# ==========================================================
st.markdown('<h1 class="main-title">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtext">Analyze movie review sentiment using a machine-learning classifier.</p>', unsafe_allow_html=True)

# Feature badges
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="feature-pill">89.4% Accuracy</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-pill">10K TF-IDF Features</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-pill">Optimized Logistic Regression</div>', unsafe_allow_html=True)

if is_optimized:
    st.success("✅ Optimized model loaded successfully.")
else:
    st.error("⚠️ Using fallback model — predictions may not be accurate.")

# ==========================================================
#  MAIN ANALYSIS INTERFACE
# ==========================================================
st.subheader("🔍 Analyze Text")

sample = st.selectbox(
    "Try a sample:",
    [
        "Write your own...",
        "This movie was fantastic! Absolutely loved it.",
        "Very disappointing experience, would not recommend.",
        "The product is okay, nothing special but gets the job done."
    ]
)

default_text = "" if sample == "Write your own..." else sample
text = st.text_area("Your text:", value=default_text, height=120)

if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
    if text.strip():
        with st.spinner("Analyzing sentiment..."):
            features = vectorizer.transform([text])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]

            colA, colB = st.columns(2)
            with colA:
                st.markdown("### Result")
                if pred == 1:
                    st.success("😊 Positive Sentiment")
                else:
                    st.error("😞 Negative Sentiment")

            with colB:
                st.markdown("### Confidence")
                confidence = max(proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%")

    else:
        st.warning("Please enter some text before analyzing.")

# ==========================================================
#  PROJECT INFORMATION
# ==========================================================
st.divider()
st.subheader("📘 About the Project")

st.markdown("""
### 📌 Problem  
This project performs **binary sentiment classification** to determine whether a movie review expresses a **positive** or **negative** opinion.

### 📊 Dataset  
- **Source:** IMDB Movie Reviews dataset  
- **Size:** 50,000 labeled reviews  
- **Task:** Binary classification  
- **Preprocessing:** Tokenization, stopword removal, TF-IDF weighting  
""")

st.markdown("""
### 🤖 Model  
- **Algorithm:** Logistic Regression (Optimized)  
- **Features:** 10,000 TF-IDF features  
- **N-grams:** trigrams  
- **Accuracy:** 89.4%  
""")

# ==========================================================
#  FOOTER
# ==========================================================
st.divider()
st.caption("""
**Ali Haider** • Machine Learning & Full-Stack Developer  
[GitHub](https://github.com/haider-sattar/sentiment-analysis/tree/main) •  
[LinkedIn](https://www.linkedin.com/in/ali-haider-467948329/)  
""")
