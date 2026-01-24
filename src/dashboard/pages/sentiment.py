import streamlit as st
import plotly.graph_objects as go
from dashboard.dashboard_components import create_sentiment_timeline, create_distribution_plot
from config import STOCK_TICKERS

def render_sentiment_analysis(data):
    news_df = data['news_df']
    
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
