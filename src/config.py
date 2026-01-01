"""
Central configuration file for the Fake News Detector project.
Contains all hyperparameters, paths, and settings.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, SIMULATED_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data Generation Settings
DATA_GENERATION = {
    "num_articles": 1000,           # Total number of news articles to generate
    "fake_news_ratio": 0.35,        # Proportion of fake news (35%)
    "num_social_posts": 5000,       # Number of social media posts
    "num_stocks": 20,               # Number of stocks to track
    "date_range_days": 180,         # Historical data range (6 months)
    "coordinated_groups": 5,        # Number of coordinated manipulation groups
    "accounts_per_group": 15,       # Accounts in each coordinated group
}

# Stock tickers to simulate
STOCK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "META", "NVDA", "JPM", "V", "WMT",
    "PG", "DIS", "NFLX", "PYPL", "INTC",
    "AMD", "CRM", "ADBE", "CSCO", "ORCL"
]

# NLP Model Settings
NLP_CONFIG = {
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 3),
    "bert_model_name": "distilbert-base-uncased",
    "bert_max_length": 512,
    "bert_batch_size": 16,
    "bert_epochs": 3,
    "bert_learning_rate": 2e-5,
    "sentiment_model": "ProsusAI/finbert",  # Financial sentiment model
    "test_size": 0.2,
    "random_state": 42,
}

# Network Analysis Settings
NETWORK_CONFIG = {
    "similarity_threshold": 0.7,     # Content similarity threshold for edge creation
    "temporal_window_hours": 2,      # Time window for coordinated activity
    "min_community_size": 3,         # Minimum size for suspicious communities
    "influence_threshold": 0.75,     # PageRank threshold for high influence
    "bot_score_threshold": 0.6,      # Threshold for bot-like behavior
}

# Correlation Analysis Settings
CORRELATION_CONFIG = {
    "event_window_days": 5,          # Days before/after event for analysis
    "significance_level": 0.05,      # Statistical significance threshold
    "min_news_spike": 10,            # Minimum articles to count as spike
    "rolling_window_days": 7,        # Rolling correlation window
}

# Dashboard Settings
DASHBOARD_CONFIG = {
    "update_interval_seconds": 30,   # Real-time update frequency
    "default_page": "Overview",
    "theme": "dark",
    "chart_height": 400,
    "network_graph_nodes_limit": 500,
    "news_feed_page_size": 20,
}

# Alert Thresholds
ALERT_THRESHOLDS = {
    "high_confidence_fake": 0.9,     # Probability threshold for high-confidence fake news
    "coordinated_activity": 0.8,     # Coordination score threshold
    "abnormal_return": 0.05,         # 5% abnormal return threshold
    "sentiment_shift": 0.3,          # Sentiment change threshold
}

# File Paths
PATHS = {
    "news_articles": SIMULATED_DATA_DIR / "news_articles.csv",
    "social_posts": SIMULATED_DATA_DIR / "social_posts.csv",
    "stock_prices": SIMULATED_DATA_DIR / "stock_prices.csv",
    "user_accounts": SIMULATED_DATA_DIR / "user_accounts.csv",
    "processed_features": PROCESSED_DATA_DIR / "features.csv",
    "tfidf_model": MODELS_DIR / "tfidf_classifier.pkl",
    "bert_model": MODELS_DIR / "bert_classifier",
    "network_graph": PROCESSED_DATA_DIR / "network.gpickle",
}

# Random seed for reproducibility
RANDOM_SEED = 42

# Fake news indicators (keywords and patterns)
FAKE_NEWS_INDICATORS = [
    "shocking", "unbelievable", "they don't want you to know",
    "secret", "exposed", "revealed", "bombshell", "scandal",
    "you won't believe", "insiders say", "anonymous sources",
    "mainstream media hides", "leaked documents"
]

# Credible news sources
CREDIBLE_SOURCES = [
    "Reuters", "Bloomberg", "Financial Times", "Wall Street Journal",
    "Associated Press", "CNBC", "MarketWatch", "The Economist",
    "Forbes", "Business Insider"
]

# Suspicious/fake sources
SUSPICIOUS_SOURCES = [
    "TotallyRealNews.com", "StockTipsNow.biz", "InvestorSecrets.net",
    "MarketInsider247.com", "FinanceLeaks.info", "WallStreetExposed.org",
    "InvestmentGuru.blog", "TruthAboutStocks.com"
]
