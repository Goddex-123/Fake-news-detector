import streamlit as st
import plotly.graph_objects as go
from dashboard.dashboard_components import render_metric_card, render_news_card
from config import STOCK_TICKERS
import pandas as pd

def render_overview(data):
    news_df = data['news_df']
    social_df = data['social_df']
    
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
