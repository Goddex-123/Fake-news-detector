"""
Impact Analyzer Module

Measures correlation between fake news events and stock price movements.
Includes event study analysis, time-series correlation, and statistical testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import CORRELATION_CONFIG, STOCK_TICKERS


class EventStudyAnalyzer:
    """Performs event study analysis around fake news events."""
    
    def __init__(self):
        self.window_days = CORRELATION_CONFIG['event_window_days']
    
    def identify_fake_news_events(self, news_df: pd.DataFrame, 
                                  min_spike=None) -> pd.DataFrame:
        """Identify days with fake news spikes."""
        if min_spike is None:
            min_spike = CORRELATION_CONFIG['min_news_spike']
        
        # Filter fake news
        fake_news = news_df[news_df['label'] == 'fake'].copy()
        fake_news['date'] = pd.to_datetime(fake_news['published_date']).dt.date
        
        # Count fake news per ticker per day
        events = fake_news.groupby(['ticker', 'date']).size().reset_index(name='fake_count')
        
        # Filter for spikes
        events = events[events['fake_count'] >= min_spike]
        
        print(f"Identified {len(events)} fake news spike events")
        
        return events
    
    def calculate_abnormal_returns(self, prices_df: pd.DataFrame, 
                                   event_date, ticker) -> dict:
        """Calculate abnormal returns around an event."""
        
        # Filter for ticker
        ticker_prices = prices_df[prices_df['ticker'] == ticker].copy()
        ticker_prices['date'] = pd.to_datetime(ticker_prices['date']).dt.date
        ticker_prices = ticker_prices.sort_values('date')
        
        # Calculate daily returns
        ticker_prices['return'] = ticker_prices['close'].pct_change()
        
        # Get event window
        event_date = pd.to_datetime(event_date).date()
        start_date = event_date - timedelta(days=self.window_days)
        end_date = event_date + timedelta(days=self.window_days)
        
        window_data = ticker_prices[
            (ticker_prices['date'] >= start_date) & 
            (ticker_prices['date'] <= end_date)
        ].copy()
        
        if len(window_data) < 3:
            return None
        
        # Calculate normal returns (using pre-event period)
        pre_event = ticker_prices[ticker_prices['date'] < event_date].tail(30)
        normal_return = pre_event['return'].mean()
        normal_std = pre_event['return'].std()
        
        # Calculate abnormal returns
        window_data['abnormal_return'] = window_data['return'] - normal_return
        window_data['cumulative_abnormal_return'] = window_data['abnormal_return'].cumsum()
        
        # Calculate metrics
        event_day_return = window_data[window_data['date'] == event_date]['return'].iloc[0] if len(window_data[window_data['date'] == event_date]) > 0 else 0
        event_day_abnormal = event_day_return - normal_return
        
        # Post-event cumulative abnormal return
        post_event = window_data[window_data['date'] > event_date]
        post_event_car = post_event['abnormal_return'].sum() if len(post_event) > 0 else 0
        
        # Statistical significance
        t_stat = event_day_abnormal / (normal_std + 1e-6)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(pre_event)-1))
        
        return {
            'event_date': event_date,
            'ticker': ticker,
            'event_day_return': event_day_return,
            'event_day_abnormal_return': event_day_abnormal,
            'post_event_car': post_event_car,
            'normal_return': normal_return,
            'normal_std': normal_std,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < CORRELATION_CONFIG['significance_level'],
            'window_data': window_data
        }
    
    def analyze_all_events(self, news_df: pd.DataFrame, 
                          prices_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze all fake news events."""
        print("\n📊 Performing event study analysis...")
        
        # Identify events
        events = self.identify_fake_news_events(news_df)
        
        # Analyze each event
        results = []
        for _, event in events.iterrows():
            result = self.calculate_abnormal_returns(
                prices_df, event['date'], event['ticker']
            )
            if result:
                result['fake_count'] = event['fake_count']
                results.append(result)
        
        results_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != 'window_data'} 
            for r in results
        ])
        
        print(f"\n✅ Analyzed {len(results_df)} events")
        
        # Summary statistics
        if len(results_df) > 0:
            print(f"\nEvent Study Results:")
            print(f"  Average abnormal return on event day: {results_df['event_day_abnormal_return'].mean()*100:.2f}%")
            print(f"  Average post-event CAR: {results_df['post_event_car'].mean()*100:.2f}%")
            print(f"  Significant events: {results_df['significant'].sum()} ({results_df['significant'].sum()/len(results_df)*100:.1f}%)")
            
            # Split by direction
            positive_events = results_df[results_df['event_day_abnormal_return'] > 0]
            negative_events = results_df[results_df['event_day_abnormal_return'] < 0]
            print(f"  Positive abnormal returns: {len(positive_events)} events")
            print(f"  Negative abnormal returns: {len(negative_events)} events")
        
        return results_df


class TimeSeriesCorrelationAnalyzer:
    """Analyzes time-series correlation between news and prices."""
    
    def __init__(self):
        self.rolling_window = CORRELATION_CONFIG['rolling_window_days']
    
    def prepare_time_series(self, news_df: pd.DataFrame, 
                           prices_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare aligned time series data."""
        
        # Aggregate news by date and ticker
        news_df = news_df.copy()
        news_df['date'] = pd.to_datetime(news_df['published_date']).dt.date
        
        news_agg = news_df.groupby(['ticker', 'date']).agg({
            'label': lambda x: (x == 'fake').sum(),
            'article_id': 'count'
        }).rename(columns={'label': 'fake_count', 'article_id': 'total_count'}).reset_index()
        
        # Prepare price data
        prices_df = prices_df.copy()
        prices_df['date'] = pd.to_datetime(prices_df['date']).dt.date
        prices_df['return'] = prices_df.groupby('ticker')['close'].pct_change()
        prices_df['volatility'] = prices_df.groupby('ticker')['return'].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        
        # Merge
        merged = pd.merge(
            prices_df[['ticker', 'date', 'close', 'return', 'volatility', 'volume']],
            news_agg,
            on=['ticker', 'date'],
            how='left'
        )
        
        # Fill missing news counts with 0
        merged['fake_count'] = merged['fake_count'].fillna(0)
        merged['total_count'] = merged['total_count'].fillna(0)
        
        return merged
    
    def calculate_correlations(self, time_series_df: pd.DataFrame) -> dict:
        """Calculate various correlations."""
        
        print("\n📈 Calculating time-series correlations...")
        
        correlations = {}
        
        for ticker in STOCK_TICKERS:
            ticker_data = time_series_df[time_series_df['ticker'] == ticker].copy()
            
            if len(ticker_data) < 10:
                continue
            
            # Remove rows with NaN in return
            ticker_data = ticker_data.dropna(subset=['return'])
            
            if len(ticker_data) < 10:
                continue
            
            # Correlation between fake news count and return
            if ticker_data['fake_count'].sum() > 0:
                corr_return, p_return = pearsonr(
                    ticker_data['fake_count'], 
                    ticker_data['return']
                )
            else:
                corr_return, p_return = 0, 1
            
            # Correlation between fake news and volatility
            if ticker_data['fake_count'].sum() > 0:
                corr_vol, p_vol = pearsonr(
                    ticker_data['fake_count'], 
                    ticker_data['volatility']
                )
            else:
                corr_vol, p_vol = 0, 1
            
            # Correlation between fake news and volume
            if ticker_data['fake_count'].sum() > 0:
                corr_volume, p_volume = pearsonr(
                    ticker_data['fake_count'], 
                    ticker_data['volume']
                )
            else:
                corr_volume, p_volume = 0, 1
            
            correlations[ticker] = {
                'return_correlation': corr_return,
                'return_p_value': p_return,
                'volatility_correlation': corr_vol,
                'volatility_p_value': p_vol,
                'volume_correlation': corr_volume,
                'volume_p_value': p_volume
            }
        
        # Average correlations
        avg_corrs = {
            'avg_return_corr': np.mean([v['return_correlation'] for v in correlations.values()]),
            'avg_volatility_corr': np.mean([v['volatility_correlation'] for v in correlations.values()]),
            'avg_volume_corr': np.mean([v['volume_correlation'] for v in correlations.values()])
        }
        
        print(f"\nAverage Correlations:")
        print(f"  Fake news ↔ Returns: {avg_corrs['avg_return_corr']:.3f}")
        print(f"  Fake news ↔ Volatility: {avg_corrs['avg_volatility_corr']:.3f}")
        print(f"  Fake news ↔ Volume: {avg_corrs['avg_volume_corr']:.3f}")
        
        return correlations, avg_corrs
    
    def rolling_correlation(self, time_series_df: pd.DataFrame, 
                           ticker: str) -> pd.DataFrame:
        """Calculate rolling correlation for a specific ticker."""
        
        ticker_data = time_series_df[time_series_df['ticker'] == ticker].copy()
        ticker_data = ticker_data.sort_values('date').dropna(subset=['return'])
        
        # Rolling correlation
        rolling_corr = ticker_data['fake_count'].rolling(
            window=self.rolling_window
        ).corr(ticker_data['return'])
        
        ticker_data['rolling_correlation'] = rolling_corr
        
        return ticker_data


class RegressionAnalyzer:
    """Regression analysis for fake news impact."""
    
    def __init__(self):
        pass
    
    def build_regression_model(self, time_series_df: pd.DataFrame) -> dict:
        """Build regression model: Return ~ Fake_Count + controls."""
        
        print("\n🔬 Building regression model...")
        
        # Prepare data
        df = time_series_df.dropna(subset=['return']).copy()
        
        # Add lagged fake news (previous day's fake news might affect today's price)
        df['fake_count_lag1'] = df.groupby('ticker')['fake_count'].shift(1).fillna(0)
        
        # Add market trend control (overall market return)
        market_return = df.groupby('date')['return'].mean().rename('market_return')
        df = df.merge(market_return, on='date', how='left')
        
        # Features and target
        X = df[['fake_count', 'fake_count_lag1', 'total_count', 'market_return']].fillna(0)
        y = df['return']
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Coefficients
        coefficients = {
            'fake_count': model.coef_[0],
            'fake_count_lag1': model.coef_[1],
            'total_count': model.coef_[2],
            'market_return': model.coef_[3],
            'intercept': model.intercept_
        }
        
        # R-squared
        r_squared = model.score(X, y)
        
        print(f"\nRegression Results:")
        print(f"  Fake news coefficient: {coefficients['fake_count']:.6f}")
        print(f"  Lagged fake news coefficient: {coefficients['fake_count_lag1']:.6f}")
        print(f"  R-squared: {r_squared:.4f}")
        
        # Interpretation
        if abs(coefficients['fake_count']) > 0.001:
            direction = "increases" if coefficients['fake_count'] > 0 else "decreases"
            print(f"\n  📌 Interpretation: Each additional fake news article {direction}")
            print(f"     stock return by {abs(coefficients['fake_count'])*100:.3f}% on average.")
        
        return {
            'model': model,
            'coefficients': coefficients,
            'r_squared': r_squared,
            'X': X,
            'y': y
        }


def analyze_impact(news_df: pd.DataFrame, prices_df: pd.DataFrame) -> dict:
    """Main function to analyze fake news impact on stock prices."""
    
    print("\n" + "="*60)
    print("💹 ANALYZING FAKE NEWS IMPACT ON STOCK PRICES")
    print("="*60)
    
    results = {}
    
    # 1. Event study analysis
    event_analyzer = EventStudyAnalyzer()
    event_results = event_analyzer.analyze_all_events(news_df, prices_df)
    results['event_study'] = event_results
    
    # 2. Time-series correlation
    ts_analyzer = TimeSeriesCorrelationAnalyzer()
    time_series = ts_analyzer.prepare_time_series(news_df, prices_df)
    correlations, avg_corrs = ts_analyzer.calculate_correlations(time_series)
    results['correlations'] = correlations
    results['avg_correlations'] = avg_corrs
    results['time_series'] = time_series
    
    # 3. Regression analysis
    reg_analyzer = RegressionAnalyzer()
    regression_results = reg_analyzer.build_regression_model(time_series)
    results['regression'] = regression_results
    
    print("\n" + "="*60)
    print("✅ IMPACT ANALYSIS COMPLETE")
    print("="*60)
    
    return results


if __name__ == "__main__":
    from config import PATHS
    
    # Load data
    print("Loading data...")
    news_df = pd.read_csv(PATHS['news_articles'])
    prices_df = pd.read_csv(PATHS['stock_prices'])
    
    # Analyze impact
    results = analyze_impact(news_df, prices_df)
    
    # Save results
    results['event_study'].to_csv(PATHS['processed_features'].parent / 'event_study_results.csv', index=False)
    results['time_series'].to_csv(PATHS['processed_features'].parent / 'time_series_analysis.csv', index=False)
    
    print(f"\n✅ Results saved to {PATHS['processed_features'].parent}")
