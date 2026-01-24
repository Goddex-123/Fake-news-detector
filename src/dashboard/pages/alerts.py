import streamlit as st
from dashboard.dashboard_components import render_alert_badge
from config import ALERT_THRESHOLDS

def render_alerts(data):
    news_df = data['news_df']
    social_df = data['social_df']
    
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
