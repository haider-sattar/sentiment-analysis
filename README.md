# 📊 Sentiment Analysis Dashboard

A production-ready machine learning application that classifies text sentiment as **Positive** or **Negative** with **89.4% accuracy**, built for showcasing end-to-end ML skills.

## 🚀 Live Demo
**[👉 Try the Live App ](https://sentiment-analysis-dpa7pjlnj6nimmdbara7ej.streamlit.app)**

## 🎯 Project Overview
This is a complete sentiment analysis pipeline built to demonstrate practical machine learning skills for freelance and professional opportunities. The application uses an optimized Logistic Regression model to analyze text sentiment in real-time with 89.4% accuracy.

## 📊 Model Performance & Optimization
| Metric | Score | Details |
|--------|-------|---------|
| **Final Accuracy** | **89.4%** | Optimized from 88.7% baseline (+0.7% gain) |
| **Best Model** | Logistic Regression | After comparing 5 algorithms |
| **Features** | 10,000 TF-IDF | With trigrams (1-3 word combinations) |
| **Training Data** | 50,000 IMDB Reviews | Balanced positive/negative samples |
| **Inference Speed** | < 0.1 seconds | Real-time analysis |

### Model Comparison Results
| Model | Accuracy | Training Time | Best For |
|-------|----------|---------------|----------|
| **Logistic Regression** | **89.4%** | 0.3s | **Production use** |
| Linear SVM | 87.5% | 0.7s | High-dimension data |
| Naive Bayes | 85.5% | 0.0s | Quick baselines |
| Random Forest | 84.9% | 22.4s | Complex patterns |
| Gradient Boosting | 80.1% | 77.5s | Non-linear data |

## 🔧 Technical Implementation

### 1. Data Pipeline
- **Dataset**: 50,000 IMDB movie reviews (balanced)
- **Preprocessing**: Text cleaning, HTML removal, stopword elimination, lemmatization
- **Feature Engineering**: TF-IDF vectorization with optimized parameters:
  - `max_features=10000`
  - `ngram_range=(1,3)` (captures phrases like "not good")
  - `min_df=10, max_df=0.5` (filters rare/common words)

### 2. Model Development
- **Algorithm Selection**: Systematic comparison of 5 ML algorithms
- **Optimization**: Hyperparameter tuning for TF-IDF and model parameters
- **Evaluation**: 80/20 train-test split with stratified sampling

### 3. Application Layer
- **Framework**: Streamlit for interactive web interface
- **Features**:
  - Real-time text sentiment analysis
  - Confidence scoring for predictions
  - Interactive visualizations
  - Sample texts for quick testing
- **Deployment**: Hugging Face Spaces (serverless hosting)

## 🛠️ Tech Stack
| Category | Technologies |
|----------|--------------|
| **Machine Learning** | Scikit-learn, Logistic Regression, TF-IDF |
| **Web Application** | Streamlit, Matplotlib |
| **Data Processing** | Pandas, NLTK |
| **Deployment** | Hugging Face Spaces, Git |
| **Development** | Python, Virtual Environments, Jupyter |

## 🚀 Quick Start

### Local Development
```bash
# 1. Clone repository
git clone https://github.com/haider-sattar/sentiment-analysis.git
cd sentiment-analysis

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
