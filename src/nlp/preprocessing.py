"""
Text Preprocessing and Feature Engineering

Handles cleaning, tokenization, and feature extraction from text data.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import NLP_CONFIG, FAKE_NEWS_INDICATORS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Preprocesses text data for NLP models."""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Keep some important words for fake news detection
        self.stop_words -= {'not', 'no', 'never', 'nothing', 'nobody'}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> list:
        """Tokenize and lemmatize text."""
        tokens = word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token not in self.stop_words and len(token) > 2]
        return lemmatized
    
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        cleaned = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned)
        return ' '.join(tokens)


class FeatureEngineer:
    """Extracts linguistic and statistical features from text."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def extract_linguistic_features(self, text: str) -> dict:
        """Extract linguistic features that indicate fake news."""
        if pd.isna(text):
            return self._empty_features()
        
        features = {}
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Sentence statistics
        sentences = text.split('.')
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['capitalized_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0
        
        # Fake news indicator presence
        text_lower = text.lower()
        features['fake_indicator_count'] = sum(1 for indicator in FAKE_NEWS_INDICATORS 
                                              if indicator in text_lower)
        features['has_fake_indicators'] = 1 if features['fake_indicator_count'] > 0 else 0
        
        # Sensationalism indicators
        sensational_words = ['shocking', 'unbelievable', 'explosive', 'bombshell', 
                           'scandal', 'exposed', 'secret', 'urgent', 'breaking']
        features['sensational_word_count'] = sum(1 for word in sensational_words 
                                                if word in text_lower)
        
        # Anonymity indicators
        anonymous_terms = ['anonymous', 'insider', 'sources say', 'leaked', 'whistleblower']
        features['anonymity_score'] = sum(1 for term in anonymous_terms if term in text_lower)
        
        # Readability (simple Flesch-Kincaid approximation)
        words_per_sentence = features['avg_sentence_length']
        syllables_per_word = features['avg_word_length'] / 3  # Rough approximation
        features['readability_score'] = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word
        
        return features
    
    def _empty_features(self) -> dict:
        """Return empty feature dict for null text."""
        return {
            'char_count': 0, 'word_count': 0, 'avg_word_length': 0,
            'sentence_count': 0, 'avg_sentence_length': 0,
            'exclamation_count': 0, 'question_count': 0, 'capitalized_ratio': 0,
            'fake_indicator_count': 0, 'has_fake_indicators': 0,
            'sensational_word_count': 0, 'anonymity_score': 0, 'readability_score': 0
        }
    
    def add_features_to_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Add linguistic features to dataframe."""
        print("Extracting linguistic features...")
        feature_dicts = df[text_column].apply(self.extract_linguistic_features)
        feature_df = pd.DataFrame(feature_dicts.tolist())
        
        # Combine with original dataframe
        result = pd.concat([df, feature_df], axis=1)
        return result


class TfidfFeatureExtractor:
    """Extracts TF-IDF features from text."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=NLP_CONFIG['tfidf_max_features'],
            ngram_range=NLP_CONFIG['tfidf_ngram_range'],
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
        self.preprocessor = TextPreprocessor()
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """Fit and transform texts to TF-IDF features."""
        print("Fitting TF-IDF vectorizer...")
        processed_texts = texts.apply(self.preprocessor.preprocess)
        features = self.vectorizer.fit_transform(processed_texts)
        print(f"TF-IDF features shape: {features.shape}")
        return features
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """Transform texts to TF-IDF features using fitted vectorizer."""
        processed_texts = texts.apply(self.preprocessor.preprocess)
        return self.vectorizer.transform(processed_texts)
    
    def get_feature_names(self) -> list:
        """Get feature names from vectorizer."""
        return self.vectorizer.get_feature_names_out()
    
    def get_top_features_by_class(self, X, y, top_n=20):
        """Get top TF-IDF features for each class."""
        feature_names = self.get_feature_names()
        
        # Calculate mean TF-IDF for each class
        fake_mean = np.asarray(X[y == 'fake'].mean(axis=0)).ravel()
        real_mean = np.asarray(X[y == 'real'].mean(axis=0)).ravel()
        
        # Get top features
        fake_top_indices = fake_mean.argsort()[-top_n:][::-1]
        real_top_indices = real_mean.argsort()[-top_n:][::-1]
        
        fake_top_features = [(feature_names[i], fake_mean[i]) for i in fake_top_indices]
        real_top_features = [(feature_names[i], real_mean[i]) for i in real_top_indices]
        
        return {
            'fake': fake_top_features,
            'real': real_top_features
        }


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "BREAKING: You won't believe what AAPL is hiding! Anonymous sources reveal shocking details!!!"
    
    preprocessor = TextPreprocessor()
    print("Original:", sample_text)
    print("Cleaned:", preprocessor.clean_text(sample_text))
    print("Preprocessed:", preprocessor.preprocess(sample_text))
    
    engineer = FeatureEngineer()
    features = engineer.extract_linguistic_features(sample_text)
    print("\nLinguistic features:")
    for key, value in features.items():
        print(f"  {key}: {value}")
