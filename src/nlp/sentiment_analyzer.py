"""
Sentiment Analysis Module

Analyzes sentiment of financial news and tracks shifts over time.
Uses FinBERT for financial domain-specific sentiment.
"""

import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import timedelta
from scipy import stats
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import NLP_CONFIG


class SentimentAnalyzer:
    """Analyzes sentiment using FinBERT (financial sentiment model)."""
    
    def __init__(self):
        print("Loading FinBERT sentiment model...")
        self.tokenizer = AutoTokenizer.from_pretrained(NLP_CONFIG['sentiment_model'])
        self.model = AutoModelForSequenceClassification.from_pretrained(NLP_CONFIG['sentiment_model'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Sentiment labels for FinBERT
        self.labels = ['positive', 'negative', 'neutral']
    
    def analyze(self, text: str) -> dict:
        """Analyze sentiment of a single text."""
        if pd.isna(text) or text == "":
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34, 'score': 0.0}
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        # Create sentiment dict
        sentiment_dict = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2])
        }
        
        # Calculate compound score (-1 to 1)
        sentiment_dict['score'] = sentiment_dict['positive'] - sentiment_dict['negative']
        
        return sentiment_dict
    
    def analyze_batch(self, texts: pd.Series) -> pd.DataFrame:
        """Analyze sentiment for multiple texts."""
        print(f"Analyzing sentiment for {len(texts)} texts...")
        sentiments = []
        
        for i, text in enumerate(texts):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(texts)}...")
            sentiment = self.analyze(text)
            sentiments.append(sentiment)
        
        print(f"✅ Completed sentiment analysis")
        return pd.DataFrame(sentiments)


class SentimentTimeSeriesAnalyzer:
    """Analyzes sentiment trends and shifts over time."""
    
    def __init__(self):
        pass
    
    def calculate_time_series(self, df: pd.DataFrame, date_col='published_date', 
                            sentiment_col='score', freq='D') -> pd.DataFrame:
        """Calculate sentiment time series."""
        df_copy = df.copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        
        # Group by date and calculate mean sentiment
        time_series = df_copy.groupby(pd.Grouper(key=date_col, freq=freq)).agg({
            sentiment_col: ['mean', 'std', 'count']
        }).reset_index()
        
        time_series.columns = ['date', 'sentiment_mean', 'sentiment_std', 'article_count']
        
        return time_series
    
    def detect_sentiment_shifts(self, time_series: pd.DataFrame, 
                               window=7, threshold=0.3) -> pd.DataFrame:
        """Detect significant sentiment shifts using rolling statistics."""
        ts = time_series.copy()
        
        # Calculate rolling mean and std
        ts['rolling_mean'] = ts['sentiment_mean'].rolling(window=window, min_periods=1).mean()
        ts['rolling_std'] = ts['sentiment_mean'].rolling(window=window, min_periods=1).std()
        
        # Detect shifts (when current sentiment deviates significantly from rolling mean)
        ts['deviation'] = (ts['sentiment_mean'] - ts['rolling_mean']) / (ts['rolling_std'] + 1e-6)
        ts['shift_detected'] = abs(ts['deviation']) > threshold
        ts['shift_direction'] = np.where(ts['deviation'] > 0, 'positive', 'negative')
        
        return ts
    
    def compare_fake_vs_real_sentiment(self, df: pd.DataFrame, 
                                      label_col='label',
                                      sentiment_col='score') -> dict:
        """Compare sentiment patterns between fake and real news."""
        fake_sentiment = df[df[label_col] == 'fake'][sentiment_col]
        real_sentiment = df[df[label_col] == 'real'][sentiment_col]
        
        # Statistical test
        t_stat, p_value = stats.ttest_ind(fake_sentiment, real_sentiment)
        
        results = {
            'fake_mean': fake_sentiment.mean(),
            'fake_std': fake_sentiment.std(),
            'real_mean': real_sentiment.mean(),
            'real_std': real_sentiment.std(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        return results
    
    def analyze_ticker_sentiment(self, df: pd.DataFrame, ticker: str,
                                date_col='published_date',
                                sentiment_col='score') -> pd.DataFrame:
        """Analyze sentiment trends for a specific ticker."""
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Time series
        ts = self.calculate_time_series(ticker_df, date_col, sentiment_col)
        
        # Detect shifts
        ts = self.detect_sentiment_shifts(ts)
        
        return ts
    
    def calculate_sentiment_momentum(self, time_series: pd.DataFrame,
                                    window=5) -> pd.DataFrame:
        """Calculate sentiment momentum (rate of change)."""
        ts = time_series.copy()
        
        # Calculate momentum as the difference between current and past sentiment
        ts['momentum'] = ts['sentiment_mean'].diff(window)
        ts['momentum_direction'] = np.where(ts['momentum'] > 0, 'increasing', 'decreasing')
        
        # Acceleration (second derivative)
        ts['acceleration'] = ts['momentum'].diff()
        
        return ts


def analyze_news_sentiment(news_df: pd.DataFrame) -> pd.DataFrame:
    """Main function to analyze sentiment of news articles."""
    
    # Combine headline and body
    news_df['text'] = news_df['headline'] + ' ' + news_df['body']
    
    # Analyze sentiment
    analyzer = SentimentAnalyzer()
    sentiment_results = analyzer.analyze_batch(news_df['text'])
    
    # Combine with original dataframe
    result_df = pd.concat([news_df, sentiment_results], axis=1)
    
    # Time series analysis
    ts_analyzer = SentimentTimeSeriesAnalyzer()
    
    # Compare fake vs real
    comparison = ts_analyzer.compare_fake_vs_real_sentiment(result_df)
    
    print("\n" + "="*50)
    print("Sentiment Analysis Results")
    print("="*50)
    print(f"Fake news sentiment: {comparison['fake_mean']:.3f} ± {comparison['fake_std']:.3f}")
    print(f"Real news sentiment: {comparison['real_mean']:.3f} ± {comparison['real_std']:.3f}")
    print(f"Difference is statistically significant: {comparison['significant']}")
    print(f"P-value: {comparison['p_value']:.4f}")
    
    return result_df


if __name__ == "__main__":
    # Test sentiment analysis
    from config import PATHS
    
    print("Loading news articles...")
    news_df = pd.read_csv(PATHS['news_articles'])
    
    # Analyze sentiment
    news_with_sentiment = analyze_news_sentiment(news_df)
    
    # Save results
    output_path = PATHS['processed_features']
    news_with_sentiment.to_csv(output_path, index=False)
    print(f"\n✅ Saved sentiment analysis to {output_path}")
