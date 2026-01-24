import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard.dashboard_components import create_event_study_chart
from config import STOCK_TICKERS, PATHS

def render_market_impact(data):
    stock_df = data['stock_df']
    news_df = data['news_df']
    
    st.markdown("## 📈 Market Impact Analysis")
    
    if stock_df is None:
        st.error("Stock data not available.")
    else:
        # Load correlation results if available
        # Ensure paths are correct relative to project root or use config
        processed_dir = PATHS['processed_features'].parent
        if not processed_dir.exists():
            processed_dir = PATHS['data_dir'] / 'processed'
            
        corr_path = processed_dir / 'time_series_analysis.csv'
        event_path = processed_dir / 'event_study_results.csv'
        
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
