import streamlit as st
from dashboard.dashboard_components import render_news_card
from config import STOCK_TICKERS

def render_news_feed(data):
    news_df = data['news_df']
    
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
    if len(filtered_df) > 0:
        total_pages = (len(filtered_df) - 1) // page_size + 1
        page_num = st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1) - 1
        
        start_idx = page_num * page_size
        end_idx = min(start_idx + page_size, len(filtered_df))
        
        # Display articles
        for _, article in filtered_df.iloc[start_idx:end_idx].iterrows():
            render_news_card(article.to_dict(), show_details=True)
        
        st.markdown(f"Page {page_num + 1} of {total_pages}")
    else:
        st.info("No articles found matching filters")
