import streamlit as st
import pandas as pd
import networkx as nx
from config import PATHS

# Custom CSS
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff4444 0%, #ff8844 50%, #ffaa44 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 20px 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #888;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, rgba(255,68,68,0.1) 0%, rgba(68,68,255,0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.1);
    }
</style>
"""

def load_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_news_data():
    """Load news articles data."""
    if PATHS['news_articles'].exists():
        df = pd.read_csv(PATHS['news_articles'])
        # Load sentiment if available
        features_path = PATHS['processed_features']
        if features_path.exists():
            sentiment_df = pd.read_csv(features_path)
            if 'score' in sentiment_df.columns:
                df = df.merge(sentiment_df[['article_id', 'score', 'positive', 'negative', 'neutral']], 
                            on='article_id', how='left')
        return df
    return None

@st.cache_data
def load_social_data():
    """Load social media posts data."""
    if PATHS['social_posts'].exists():
        return pd.read_csv(PATHS['social_posts'])
    return None

@st.cache_data
def load_stock_data():
    """Load stock prices data."""
    if PATHS['stock_prices'].exists():
        return pd.read_csv(PATHS['stock_prices'])
    return None

@st.cache_data
def load_accounts_data():
    """Load user accounts data."""
    if PATHS['user_accounts'].exists():
        return pd.read_csv(PATHS['user_accounts'])
    return None

@st.cache_resource
def load_network_graph():
    """Load network graph."""
    if PATHS['network_graph'].exists():
        return nx.read_gpickle(PATHS['network_graph'])
    return None

def load_all_data():
    with st.spinner("Loading data..."):
        return {
            'news_df': load_news_data(),
            'social_df': load_social_data(),
            'stock_df': load_stock_data(),
            'accounts_df': load_accounts_data(),
            'network_graph': load_network_graph()
        }
