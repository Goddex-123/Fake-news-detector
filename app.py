"""
AI-Powered Fake News and Market Manipulation Detector
Main Streamlit Dashboard Application

A master-level data science project combining NLP, network analysis, and ML.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import PATHS, ALERT_THRESHOLDS, STOCK_TICKERS
from dashboard.dashboard_components import *
import pickle
import networkx as nx

# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
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
""", unsafe_allow_html=True)

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

# Load data
with st.spinner("Loading data..."):
    news_df = load_news_data()
    social_df = load_social_data()
    stock_df = load_stock_data()
    accounts_df = load_accounts_data()
    network_graph = load_network_graph()

# Check if data exists
if news_df is None:
    st.error("⚠️ No data found. Please run data generation first:")
    st.code("python src/data_generation/data_simulator.py", language="bash")
    st.stop()

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.markdown("## 📊 System Overview")
    st.markdown("Real-time monitoring of fake news articles and their impact on financial markets.")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_articles = len(news_df)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        render_metric_card("Total Articles", f"{total_articles:,}", icon="📰")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        fake_count = len(news_df[news_df['label'] == 'fake'])
        fake_pct = (fake_count / total_articles * 100) if total_articles > 0 else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        render_metric_card("Fake News Detected", f"{fake_count:,}", f"{fake_pct:.1f}%", icon="🚨")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        if social_df is not None:
            coordinated_posts = social_df['coordinated'].sum()
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card("Coordinated Posts", f"{coordinated_posts:,}", icon="🕸️")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        monitored_stocks = len(STOCK_TICKERS)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        render_metric_card("Stocks Monitored", f"{monitored_stocks}", icon="📈")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📰 Recent Fake News Detection")
        recent_fake = news_df[news_df['label'] == 'fake'].sort_values('published_date', ascending=False).head(5)
        for _, article in recent_fake.iterrows():
            render_news_card(article.to_dict(), show_details=False)
    
    with col2:
        st.markdown("### 📊 Detection Statistics")
        
        # Fake news by source
        fake_by_source = news_df[news_df['label'] == 'fake']['source'].value_counts().head(5)
        
        fig = go.Figure(data=[go.Bar(
            x=fake_by_source.values,
            y=fake_by_source.index,
            orientation='h',
            marker=dict(color='#ff4444')
        )])
        
        fig.update_layout(
            title="Top Sources of Fake News",
            xaxis_title="Count",
            yaxis_title="Source",
            template='plotly_dark',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline
    st.markdown("### 📅 Fake News Timeline")
    
    news_df_time = news_df.copy()
    news_df_time['published_date'] = pd.to_datetime(news_df_time['published_date'])
    news_df_time['date'] = news_df_time['published_date'].dt.date
    
    timeline = news_df_time.groupby(['date', 'label']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    if 'fake' in timeline.columns:
        fig.add_trace(go.Scatter(
            x=timeline.index,
            y=timeline['fake'],
            name='Fake News',
            fill='tozeroy',
            line=dict(color='#ff4444', width=2)
        ))
    
    if 'real' in timeline.columns:
        fig.add_trace(go.Scatter(
            x=timeline.index,
            y=timeline['real'],
            name='Real News',
            fill='tozeroy',
            line=dict(color='#44ff44', width=2)
        ))
    
    fig.update_layout(
        title="News Articles Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Articles",
        template='plotly_dark',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: NEWS FEED
# ============================================================================
elif page == "📰 News Feed":
    st.markdown("## 📰 News Feed")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_label = st.selectbox("Filter by Type", ["All", "Fake Only", "Real Only"])
    
    with col2:
        filter_ticker = st.selectbox("Filter by Ticker", ["All"] + STOCK_TICKERS)
    
    with col3:
        sort_by = st.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Credibility (Low)", "Credibility (High)"])
    
    # Apply filters
    filtered_df = news_df.copy()
    
    if filter_label == "Fake Only":
        filtered_df = filtered_df[filtered_df['label'] == 'fake']
    elif filter_label == "Real Only":
        filtered_df = filtered_df[filtered_df['label'] == 'real']
    
    if filter_ticker != "All":
        filtered_df = filtered_df[filtered_df['ticker'] == filter_ticker]
    
    # Apply sorting
    if sort_by == "Date (Newest)":
        filtered_df = filtered_df.sort_values('published_date', ascending=False)
    elif sort_by == "Date (Oldest)":
        filtered_df = filtered_df.sort_values('published_date', ascending=True)
    elif sort_by == "Credibility (Low)":
        filtered_df = filtered_df.sort_values('credibility_score', ascending=True)
    elif sort_by == "Credibility (High)":
        filtered_df = filtered_df.sort_values('credibility_score', ascending=False)
    
    st.markdown(f"### Showing {len(filtered_df)} articles")
    
    # Pagination
    page_size = 20
    total_pages = (len(filtered_df) - 1) // page_size + 1
    page_num = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1) - 1
    
    start_idx = page_num * page_size
    end_idx = min(start_idx + page_size, len(filtered_df))
    
    # Display articles
    for _, article in filtered_df.iloc[start_idx:end_idx].iterrows():
        render_news_card(article.to_dict(), show_details=True)
    
    st.markdown(f"Page {page_num + 1} of {total_pages}")

# ============================================================================
# PAGE: SENTIMENT ANALYSIS
# ============================================================================
elif page == "📊 Sentiment Analysis":
    st.markdown("## 📊 Sentiment Analysis")
    
    if 'score' not in news_df.columns:
        st.warning("Sentiment analysis not yet performed. Run the sentiment analyzer:")
        st.code("python src/nlp/sentiment_analyzer.py", language="bash")
    else:
        # Sentiment timeline
        st.markdown("### Sentiment Over Time")
        fig = create_sentiment_timeline(news_df)
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            st.markdown("### Sentiment Distribution")
            fig = create_distribution_plot(news_df, 'score', 'label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Average sentiment by label
            st.markdown("### Average Sentiment by Type")
            fake_sentiment = news_df[news_df['label'] == 'fake']['score'].mean()
            real_sentiment = news_df[news_df['label'] == 'real']['score'].mean()
            
            fig = go.Figure(data=[go.Bar(
                x=['Fake News', 'Real News'],
                y=[fake_sentiment, real_sentiment],
                marker_color=['#ff4444', '#44ff44']
            )])
            
            fig.update_layout(
                title="Average Sentiment Score",
                yaxis_title="Sentiment",
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Ticker-specific analysis
        st.markdown("### Sentiment by Ticker")
        selected_ticker = st.selectbox("Select Ticker", STOCK_TICKERS)
        
        ticker_df = news_df[news_df['ticker'] == selected_ticker]
        if len(ticker_df) > 0:
            fig = create_sentiment_timeline(ticker_df)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: NETWORK ANALYSIS
# ============================================================================
elif page == "🕸️ Network Analysis":
    st.markdown("## 🕸️ Network Analysis")
    
    if network_graph is None:
        st.warning("Network graph not yet built. Run the network analyzer:")
        st.code("python src/network_analysis/network_detector.py", language="bash")
    else:
        st.markdown(f"**Network Statistics:** {network_graph.number_of_nodes()} nodes, {network_graph.number_of_edges()} edges")
        
        # Network visualization
        st.markdown("### Network Graph")
        st.info("🔴 Red nodes = Suspicious/Coordinated accounts | 🔵 Blue nodes = Regular accounts")
        
        # For demonstration, mark coordinated accounts as suspicious
        if accounts_df is not None:
            coordinated_ids = accounts_df[accounts_df['coordinated'] == True]['account_id'].tolist()
            suspicious_groups = [{'members': coordinated_ids}]
        else:
            suspicious_groups = None
        
        fig = create_network_graph(network_graph, suspicious_groups=suspicious_groups, max_nodes=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Community statistics
        if social_df is not None and accounts_df is not None:
            st.markdown("### 🚨 Suspicious Activity Detected")
            
            coordinated_accounts = accounts_df[accounts_df['coordinated'] == True]
            coordinated_posts = social_df[social_df['coordinated'] == True]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Coordinated Accounts", len(coordinated_accounts))
            
            with col2:
                st.metric("Coordinated Posts", len(coordinated_posts))
            
            with col3:
                num_groups = len(accounts_df[accounts_df['coordinated'] == True]['group_id'].unique())
                st.metric("Detected Groups", num_groups)
            
            # Group details
            st.markdown("### Coordination Patterns")
            
            for group_id in sorted(accounts_df[accounts_df['coordinated'] == True]['group_id'].unique()):
                with st.expander(f"Group {group_id}"):
                    group_accounts = accounts_df[accounts_df['group_id'] == group_id]
                    group_posts = social_df[social_df['account_id'].isin(group_accounts['account_id'])]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Size:** {len(group_accounts)} accounts")
                        st.write(f"**Posts:** {len(group_posts)} total posts")
                        st.write(f"**Avg Account Age:** {group_accounts['account_age_days'].mean():.0f} days")
                    
                    with col2:
                        st.write(f"**Verified:** {group_accounts['verified'].sum()} accounts")
                        st.write(f"**Avg Followers:** {group_accounts['followers'].mean():.0f}")

# ============================================================================
# PAGE: MARKET IMPACT
# ============================================================================
elif page == "📈 Market Impact":
    st.markdown("## 📈 Market Impact Analysis")
    
    if stock_df is None:
        st.error("Stock data not available.")
    else:
        # Load correlation results if available
        corr_path = PATHS['processed_features'].parent / 'time_series_analysis.csv'
        event_path = PATHS['processed_features'].parent / 'event_study_results.csv'
        
        st.markdown("### Correlation Analysis")
        st.info("Analysis of correlation between fake news events and stock price movements")
        
        # Select ticker
        selected_ticker = st.selectbox("Select Ticker for Analysis", STOCK_TICKERS)
        
        # Price chart with fake news overlay
        ticker_prices = stock_df[stock_df['ticker'] == selected_ticker].copy()
        ticker_prices['date'] = pd.to_datetime(ticker_prices['date'])
        ticker_news = news_df[news_df['ticker'] == selected_ticker].copy()
        ticker_news['date'] = pd.to_datetime(ticker_news['published_date']).dt.date
        
        fake_events = ticker_news[ticker_news['label'] == 'fake'].groupby('date').size().reset_index(name='count')
        
        # Create figure with secondary y-axis
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=ticker_prices['date'], y=ticker_prices['close'],
                      name="Stock Price", line=dict(color='#4444ff', width=2)),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Bar(x=pd.to_datetime(fake_events['date']), y=fake_events['count'],
                   name="Fake News Count", marker_color='#ff4444', opacity=0.6),
            secondary_y=True
        )
        
        fig.update_xaxis_title("Date")
        fig.update_yaxis_title("Stock Price ($)", secondary_y=False)
        fig.update_yaxis_title("Fake News Count", secondary_y=True)
        fig.update_layout(
            title=f"{selected_ticker} Price vs Fake News Volume",
            template='plotly_dark',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Event study results if available
        if event_path.exists():
            event_df = pd.read_csv(event_path)
            ticker_events = event_df[event_df['ticker'] == selected_ticker]
            
            if len(ticker_events) > 0:
                st.markdown("### Event Study Analysis")
                st.write(f"**Number of Events:** {len(ticker_events)}")
                st.write(f"**Average Abnormal Return:** {ticker_events['event_day_abnormal_return'].mean()*100:.2f}%")
                st.write(f"**Significant Events:** {ticker_events['significant'].sum()}")
                
                fig = create_event_study_chart(ticker_events, selected_ticker)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: ALERTS
# ============================================================================
elif page == "⚠️ Alerts":
    st.markdown("## ⚠️ Active Alerts")
    
    # High-confidence fake news
    st.markdown("### 🚨 High-Confidence Fake News")
    high_conf_fake = news_df[
        (news_df['label'] == 'fake') & 
        (news_df['credibility_score'] < (1 - ALERT_THRESHOLDS['high_confidence_fake']))
    ].sort_values('published_date', ascending=False).head(10)
    
    if len(high_conf_fake) > 0:
        for _, article in high_conf_fake.iterrows():
            render_alert_badge(
                "fake_news",
                f"{article['ticker']}: {article['headline'][:100]}...",
                "high"
            )
    else:
        st.success("No high-confidence fake news alerts")
    
    # Coordinated campaigns
    if social_df is not None:
        st.markdown("### 🕸️ Coordinated Campaigns")
        coordinated_count = social_df['coordinated'].sum()
        
        if coordinated_count > 100:
            render_alert_badge(
                "coordinated",
                f"{coordinated_count} coordinated posts detected across multiple groups",
                "high"
            )
        else:
            st.success("No significant coordinated activity detected")
    
    # Market anomalies
    st.markdown("### 📉 Market Anomalies")
    st.info("Advanced anomaly detection would appear here in production version")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; padding: 20px;">
    <p><strong>AI-Powered Fake News & Market Manipulation Detector</strong></p>
    <p>Built with Python, NLP (BERT), NetworkX, and Streamlit</p>
    <p>⚠️ For educational and demonstration purposes only</p>
</div>
""", unsafe_allow_html=True)
