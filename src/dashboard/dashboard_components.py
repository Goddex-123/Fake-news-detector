"""
Dashboard Components

Reusable visualization components for the Streamlit dashboard.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime


def render_metric_card(label: str, value: str, delta: str = None, icon: str = "📊"):
    """Render a metric card."""
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown(f"## {icon}")
    with col2:
        if delta:
            st.metric(label=label, value=value, delta=delta)
        else:
            st.metric(label=label, value=value)


def render_news_card(article: dict, show_details: bool = True):
    """Render a single news article card."""
    
    # Determine badge color based on classification
    if article.get('predicted_label') == 'fake' or article.get('label') == 'fake':
        badge_color = "🔴"
        badge_text = "FAKE NEWS"
        confidence = article.get('fake_probability', article.get('credibility_score', 0.5))
    else:
        badge_color = "🟢"
        badge_text = "VERIFIED"
        confidence = 1 - article.get('fake_probability', 1 - article.get('credibility_score', 0.5))
    
    # Create card
    with st.container():
        st.markdown(f"""
        <div style="padding: 15px; border-left: 4px solid {'#ff4444' if badge_text == 'FAKE NEWS' else '#44ff44'}; 
                    background-color: rgba(255,255,255,0.05); border-radius: 5px; margin-bottom: 15px;">
        <h4 style="margin: 0;">{article.get('headline', 'No headline')}</h4>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
        with col1:
            st.caption(f"{badge_color} **{badge_text}**")
        with col2:
            st.caption(f"📰 {article.get('source', 'Unknown')}")
        with col3:
            st.caption(f"💹 {article.get('ticker', 'N/A')}")
        with col4:
            st.caption(f"📅 {article.get('published_date', 'N/A')}")
        
        if show_details:
            with st.expander("View Details"):
                st.write(f"**Body:** {article.get('body', 'No content')[:300]}...")
                st.progress(confidence, text=f"Confidence: {confidence*100:.1f}%")
                if 'score' in article:
                    sentiment = article['score']
                    sentiment_text = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
                    st.write(f"**Sentiment:** {sentiment_text} ({sentiment:.2f})")


def create_sentiment_timeline(df: pd.DataFrame):
    """Create sentiment timeline chart."""
    
    df = df.copy()
    df['published_date'] = pd.to_datetime(df['published_date'])
    df = df.sort_values('published_date')
    
    # Aggregate by date and label
    daily_sentiment = df.groupby([pd.Grouper(key='published_date', freq='D'), 'label'])['score'].mean().reset_index()
    
    fig = go.Figure()
    
    # Fake news sentiment line
    fake_data = daily_sentiment[daily_sentiment['label'] == 'fake']
    fig.add_trace(go.Scatter(
        x=fake_data['published_date'],
        y=fake_data['score'],
        mode='lines+markers',
        name='Fake News',
        line=dict(color='#ff4444', width=2),
        marker=dict(size=6)
    ))
    
    # Real news sentiment line
    real_data = daily_sentiment[daily_sentiment['label'] == 'real']
    fig.add_trace(go.Scatter(
        x=real_data['published_date'],
        y=real_data['score'],
        mode='lines+markers',
        name='Real News',
        line=dict(color='#44ff44', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Sentiment Timeline: Fake vs Real News",
        xaxis_title="Date",
        yaxis_title="Sentiment Score",
        hovermode='x unified',
        template='plotly_dark',
        height=400
    )
    
    return fig


def create_network_graph(graph: nx.Graph, communities: dict = None, 
                        suspicious_groups: list = None, max_nodes: int = 500):
    """Create interactive network visualization."""
    
    # Limit nodes for performance
    if graph.number_of_nodes() > max_nodes:
        # Get most connected nodes
        node_degrees = dict(graph.degree())
        top_nodes = sorted(node_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        subgraph = graph.subgraph([n[0] for n in top_nodes])
    else:
        subgraph = graph
    
    # Calculate layout
    pos = nx.spring_layout(subgraph, k=0.5, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    
    # Get suspicious node IDs
    suspicious_nodes = set()
    if suspicious_groups:
        for group in suspicious_groups:
            suspicious_nodes.update(group['members'])
    
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Node text
        username = subgraph.nodes[node].get('username', 'Unknown')
        degree = subgraph.degree(node)
        node_text.append(f"{username}<br>Connections: {degree}")
        
        # Node color (suspicious vs normal)
        if node in suspicious_nodes:
            node_color.append('#ff4444')  # Red for suspicious
        else:
            node_color.append('#4444ff')  # Blue for normal
        
        # Node size based on degree
        node_size.append(5 + degree * 2)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=0.5, color='white')
        ))
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=f"Social Network Graph ({subgraph.number_of_nodes()} nodes)",
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0, l=0, r=0, t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       template='plotly_dark',
                       height=600
                   ))
    
    return fig


def create_correlation_heatmap(correlations: dict):
    """Create correlation heatmap."""
    
    # Prepare data for heatmap
    tickers = list(correlations.keys())
    metrics = ['return_correlation', 'volatility_correlation', 'volume_correlation']
    metric_labels = ['Returns', 'Volatility', 'Volume']
    
    data = []
    for metric in metrics:
        row = [correlations[ticker][metric] for ticker in tickers]
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=tickers,
        y=metric_labels,
        colorscale='RdBu',
        zmid=0,
        text=np.round(data, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Fake News Correlation with Stock Metrics",
        xaxis_title="Stock Ticker",
        yaxis_title="Metric",
        template='plotly_dark',
        height=400
    )
    
    return fig


def create_event_study_chart(event_data: pd.DataFrame, ticker: str):
    """Create event study visualization showing abnormal returns."""
    
    ticker_events = event_data[event_data['ticker'] == ticker]
    
    if len(ticker_events) == 0:
        return None
    
    fig = go.Figure()
    
    # Plot each event
    for idx, event in ticker_events.iterrows():
        fig.add_trace(go.Bar(
            x=[event['event_date']],
            y=[event['event_day_abnormal_return'] * 100],
            name=f"Event {idx}",
            marker_color='red' if event['event_day_abnormal_return'] < 0 else 'green'
        ))
    
    fig.update_layout(
        title=f"Abnormal Returns on Fake News Event Days: {ticker}",
        xaxis_title="Event Date",
        yaxis_title="Abnormal Return (%)",
        showlegend=False,
        template='plotly_dark',
        height=400
    )
    
    return fig


def create_volume_chart(time_series_df: pd.DataFrame, ticker: str = None):
    """Create fake news volume over time chart."""
    
    if ticker:
        df = time_series_df[time_series_df['ticker'] == ticker].copy()
        title = f"Fake News Volume Over Time: {ticker}"
    else:
        df = time_series_df.copy()
        title = "Total Fake News Volume Over Time"
    
    df['date'] = pd.to_datetime(df['date'])
    
    if not ticker:
        # Aggregate across all tickers
        df = df.groupby('date')['fake_count'].sum().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['fake_count'],
        mode='lines',
        name='Fake News Count',
        fill='tozeroy',
        line=dict(color='#ff4444', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Number of Fake Articles",
        template='plotly_dark',
        height=400
    )
    
    return fig


def render_alert_badge(alert_type: str, message: str, severity: str = "high"):
    """Render an alert badge."""
    
    colors = {
        "high": "#ff4444",
        "medium": "#ffaa44",
        "low": "#ffff44"
    }
    
    icons = {
        "fake_news": "🚨",
        "coordinated": "🕸️",
        "market_impact": "📉",
        "sentiment_shift": "📊"
    }
    
    color = colors.get(severity, "#ff4444")
    icon = icons.get(alert_type, "⚠️")
    
    st.markdown(f"""
    <div style="padding: 10px; background-color: {color}22; border-left: 4px solid {color}; 
                border-radius: 5px; margin-bottom: 10px;">
    <strong>{icon} {alert_type.replace('_', ' ').upper()}</strong><br>
    {message}
    </div>
    """, unsafe_allow_html=True)


def create_distribution_plot(df: pd.DataFrame, column: str, label_column: str = 'label'):
    """Create distribution comparison plot."""
    
    fig = go.Figure()
    
    # Fake news distribution
    fake_data = df[df[label_column] == 'fake'][column]
    fig.add_trace(go.Histogram(
        x=fake_data,
        name='Fake News',
        opacity=0.7,
        marker_color='#ff4444',
        nbinsx=30
    ))
    
    # Real news distribution
    real_data = df[df[label_column] == 'real'][column]
    fig.add_trace(go.Histogram(
        x=real_data,
        name='Real News',
        opacity=0.7,
        marker_color='#44ff44',
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f"Distribution of {column.replace('_', ' ').title()}",
        xaxis_title=column.replace('_', ' ').title(),
        yaxis_title="Frequency",
        barmode='overlay',
        template='plotly_dark',
        height=400
    )
    
    return fig
