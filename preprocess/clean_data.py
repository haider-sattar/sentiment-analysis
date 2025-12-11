"""
Clean and prepare IMDB sentiment dataset.
This script loads the IMDB reviews dataset, cleans the text data by removing HTML tags,
punctuation, and stopwords, and saves the cleaned data to a new CSV file.
"""

import pandas as pd
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords once
nltk.download('stopwords')

def clean_text(text):
    """Cleans the input text by removing HTML tags, punctuation, and stopwords."""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)  # Remove HTML
    text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(words)

def main():
    
    print("1. Loading data...")
    df = pd.read_csv('data/sentiment_data.csv')
    print(f"   Loaded {len(df)} reviews")
    
    print("2. Cleaning text...")
    global stop_words
    stop_words = set(stopwords.words('english'))
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    print("3. Saving cleaned data...")
    df[['cleaned_review', 'sentiment']].to_csv('data/cleaned_reviews.csv', index=False)
    print(f"   Saved to data/cleaned_reviews.csv")
    
    print("4. Quick stats:")
    print(f"   Positive: {(df['sentiment'] == 'positive').sum()}")
    print(f"   Negative: {(df['sentiment'] == 'negative').sum()}")
    
if __name__ == "__main__":
    main()