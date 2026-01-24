"""
AI-Powered Fake News and Market Manipulation Detector
Main Streamlit Dashboard Application

A master-level data science project combining NLP, network analysis, and ML.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import dashboard modules
from dashboard.utils import load_css, load_all_data
from dashboard.pages import (
    overview,
    news_feed,
    sentiment,
    network,
    market,
    alerts
)

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
load_css()

# Sidebar navigation
st.sidebar.markdown("## 🔍 Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["🏠 Overview", "📰 News Feed", "📊 Sentiment Analysis", 
     "🕸️ Network Analysis", "📈 Market Impact", "⚠️ Alerts"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
confidence_threshold = st.sidebar.slider("Alert confidence threshold", 0.5, 1.0, 0.85, 0.05)

# Main header
st.markdown('<h1 class="main-header">🔍 AI-Powered Fake News Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Detecting misinformation and market manipulation using advanced NLP and network analysis</p>', unsafe_allow_html=True)

# Load all data
data = load_all_data()

# Check if data exists
if data['news_df'] is None:
    st.error("⚠️ No data found. Please run data generation first:")
    st.code("python src/data_generation/data_simulator.py", language="bash")
    st.stop()

# Route to appropriate page
if page == "🏠 Overview":
    overview.render_overview(data)
elif page == "📰 News Feed":
    news_feed.render_news_feed(data)
elif page == "📊 Sentiment Analysis":
    sentiment.render_sentiment_analysis(data)
elif page == "🕸️ Network Analysis":
    network.render_network_analysis(data)
elif page == "📈 Market Impact":
    market.render_market_impact(data)
elif page == "⚠️ Alerts":
    alerts.render_alerts(data)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p><strong>AI-Powered Fake News & Market Manipulation Detector</strong></p>
    <p>Built with Python, NLP (BERT), NetworkX, and Streamlit</p>
    <p>⚠️ For educational and demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)
