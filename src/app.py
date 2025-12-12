import streamlit as st
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np

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
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .feature-pill {
        display: inline-block;
        background: #EFF6FF;
        color: #1E40AF;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
#  LOAD OPTIMIZED MODEL
# ==========================================================
@st.cache_resource
def load_model():
    """Load the pre-trained optimized model"""
    try:
        vectorizer = joblib.load("optimized_vectorizer.joblib")
        model = joblib.load("optimized_model.joblib")
        return vectorizer, model, True
    except:
        # Fallback to demo model
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vectorizer = TfidfVectorizer(max_features=100)
        model = LogisticRegression()
        X = vectorizer.fit_transform(["good", "bad"])
        model.fit(X, [1, 0])
        return vectorizer, model, False

# Load model
vectorizer, model, is_optimized = load_model()

# ==========================================================
#  HEADER
# ==========================================================
st.markdown('<h1 class="main-title">📊 Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)

# Feature badges
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="feature-pill">89.4% Accuracy</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="feature-pill">10K Features</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="feature-pill">Fast Inference</div>', unsafe_allow_html=True)

st.caption("An optimized machine learning model for text sentiment classification")

# ==========================================================
#  MAIN ANALYSIS INTERFACE
# ==========================================================
st.subheader("🔍 Analyze Text")

# Quick samples
sample = st.selectbox(
    "Try a sample:",
    ["Write your own...", 
     "This movie was fantastic! Absolutely loved it.",
     "Very disappointing experience, would not recommend.",
     "The product is okay, nothing special but gets the job done."]
)

if sample == "Write your own...":
    default_text = ""
else:
    default_text = sample

text = st.text_area("Your text:", value=default_text, height=100)

if st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True):
    if text.strip():
        with st.spinner("Processing..."):
            # Predict
            features = vectorizer.transform([text])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            
            # Results
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### Result")
                if pred == 1:
                    st.success("✅ **Positive**", icon="😊")
                else:
                    st.error("❌ **Negative**", icon="😞")
            
            with col_b:
                st.markdown("### Confidence")
                confidence = max(proba) * 100
                st.metric("Score", f"{confidence:.1f}%")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(6, 3))
            labels = ['Negative', 'Positive']
            colors = ['#EF4444', '#10B981']
            ax.bar(labels, proba, color=colors, edgecolor='black', width=0.6)
            ax.set_ylim([0, 1])
            ax.set_ylabel('Probability')
            ax.set_title('Sentiment Distribution')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            st.pyplot(fig)
    else:
        st.warning("Please enter some text.")

# ==========================================================
#  PROJECT INFORMATION
# ==========================================================
st.divider()
st.subheader("📋 Project Overview")

st.markdown("""
**Model Architecture:**
- **Algorithm**: Logistic Regression (Optimized)
- **Features**: 10,000 TF-IDF with trigrams (1-3)
- **Dataset**: 50,000 IMDB movie reviews
- **Accuracy**: 89.4% (optimized from 88.7% baseline)

**Optimization:**
- Feature selection: min_df=10, max_df=0.5
- N-gram range: 1-3 for phrase detection
- Hyperparameter tuning via systematic comparison

""")

            
# ==========================================================
#  FOOTER
# ==========================================================
st.divider()
st.caption("""
**ali haider** • Built with Streamlit • 
[View on GitHub](https://github.com) • 
[Live Demo](https://huggingface.co/spaces)
""")
