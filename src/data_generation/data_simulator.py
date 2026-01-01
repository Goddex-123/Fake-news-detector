"""
Data Simulator for Financial News and Market Data

Generates realistic financial news articles, social media posts, and stock price data
with coordinated manipulation patterns for demonstration and testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    DATA_GENERATION, STOCK_TICKERS, FAKE_NEWS_INDICATORS,
    CREDIBLE_SOURCES, SUSPICIOUS_SOURCES, PATHS, RANDOM_SEED
)

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)


class FinancialNewsGenerator:
    """Generates realistic financial news articles (both fake and real)."""
    
    def __init__(self):
        self.fake = Faker()
        
        # Real news templates
        self.real_templates = [
            "{company} reports Q{quarter} earnings of ${earnings}B, {direction} expectations",
            "{company} announces new {product} initiative, stock {movement}",
            "{company} CEO discusses {topic} in quarterly call",
            "Analysts {action} {company} stock to {rating} rating",
            "{company} partners with {partner} for {initiative}",
            "Market analysis: {company} shows strong {metric} growth",
            "{company} faces regulatory review over {issue}",
            "Industry report: {sector} sector sees {trend} in Q{quarter}",
        ]
        
        # Fake news templates (sensational, misleading)
        self.fake_templates = [
            "BREAKING: {company} insider reveals {shocking_claim}!",
            "You won't believe what {company} is hiding from investors",
            "Anonymous sources: {company} about to {dramatic_event}",
            "{company} SCANDAL: {shocking_claim} exposed by leaked documents",
            "Shocking truth about {company} that Wall Street doesn't want you to know",
            "URGENT: {company} stock about to {dramatic_prediction} - Act NOW!",
            "Secret meeting reveals {company} plan to {conspiracy}",
            "{company} whistleblower exposes {shocking_claim}",
        ]
        
        self.shocking_claims = [
            "massive accounting fraud", "insider trading scheme",
            "product safety coverup", "bankruptcy filing next week",
            "CEO resignation imminent", "hostile takeover attempt",
            "criminal investigation", "major data breach"
        ]
        
        self.dramatic_events = [
            "declare bankruptcy", "lay off 50% of workforce",
            "face criminal charges", "shut down operations",
            "be acquired at fire-sale price"
        ]
        
        self.dramatic_predictions = [
            "crash 80%", "skyrocket 500%",
            "be delisted", "triple in 24 hours"
        ]
        
        self.conspiracies = [
            "manipulate stock prices", "deceive investors",
            "hide massive losses", "inflate earnings"
        ]
    
    def generate_real_article(self, ticker: str, date: datetime) -> Dict:
        """Generate a credible real news article."""
        template = random.choice(self.real_templates)
        
        # Fill in template with realistic data
        article_data = {
            'company': ticker,
            'quarter': random.choice(['1', '2', '3', '4']),
            'earnings': round(random.uniform(0.5, 10.0), 2),
            'direction': random.choice(['beating', 'meeting', 'missing']),
            'product': random.choice(['AI', 'cloud', 'sustainability', 'innovation']),
            'movement': random.choice(['rises 3%', 'falls 2%', 'remains stable']),
            'topic': random.choice(['growth strategy', 'market conditions', 'future outlook']),
            'action': random.choice(['upgrade', 'downgrade', 'maintain']),
            'rating': random.choice(['buy', 'hold', 'sell']),
            'partner': random.choice(['tech giant', 'industry leader', 'startup']),
            'initiative': random.choice(['digital transformation', 'market expansion', 'R&D']),
            'metric': random.choice(['revenue', 'profit', 'user']),
            'issue': random.choice(['data privacy', 'antitrust concerns', 'compliance']),
            'sector': random.choice(['tech', 'finance', 'healthcare', 'retail']),
            'trend': random.choice(['growth', 'consolidation', 'innovation']),
        }
        
        headline = template.format(**article_data)
        
        # Generate article body
        body = self._generate_credible_body(ticker, article_data)
        
        return {
            'article_id': self.fake.uuid4(),
            'headline': headline,
            'body': body,
            'source': random.choice(CREDIBLE_SOURCES),
            'author': self.fake.name(),
            'published_date': date,
            'ticker': ticker,
            'label': 'real',
            'credibility_score': round(random.uniform(0.85, 1.0), 2)
        }
    
    def generate_fake_article(self, ticker: str, date: datetime) -> Dict:
        """Generate a sensational fake news article."""
        template = random.choice(self.fake_templates)
        
        article_data = {
            'company': ticker,
            'shocking_claim': random.choice(self.shocking_claims),
            'dramatic_event': random.choice(self.dramatic_events),
            'dramatic_prediction': random.choice(self.dramatic_predictions),
            'conspiracy': random.choice(self.conspiracies),
        }
        
        headline = template.format(**article_data)
        
        # Generate sensational body with fake news indicators
        body = self._generate_fake_body(ticker, article_data)
        
        return {
            'article_id': self.fake.uuid4(),
            'headline': headline,
            'body': body,
            'source': random.choice(SUSPICIOUS_SOURCES),
            'author': random.choice([self.fake.name(), 'Anonymous Insider', 'Industry Whistleblower']),
            'published_date': date,
            'ticker': ticker,
            'label': 'fake',
            'credibility_score': round(random.uniform(0.1, 0.4), 2)
        }
    
    def _generate_credible_body(self, ticker: str, data: Dict) -> str:
        """Generate realistic article body."""
        paragraphs = [
            f"{ticker} published its quarterly results today, showing {random.choice(['strong', 'moderate', 'weak'])} performance across key metrics.",
            f"The company reported revenue of ${random.uniform(1, 50):.2f}B, representing a {random.uniform(-10, 30):.1f}% change year-over-year.",
            f"Analysts from major firms have been {random.choice(['closely monitoring', 'evaluating', 'tracking'])} the company's performance in this sector.",
            f"Looking ahead, the company maintains {random.choice(['optimistic', 'cautious', 'confident'])} guidance for the coming quarters."
        ]
        return " ".join(random.sample(paragraphs, k=3))
    
    def _generate_fake_body(self, ticker: str, data: Dict) -> str:
        """Generate sensational fake article body."""
        paragraphs = [
            f"According to anonymous sources, {ticker} is involved in {data.get('shocking_claim', 'suspicious activities')}.",
            "Mainstream media refuses to report this shocking development that could impact millions of investors!",
            f"Insider information suggests the company will {data.get('dramatic_event', 'face serious consequences')} within days.",
            "This is the kind of story they don't want you to see. Share before it gets taken down!",
            f"Expert analysts predict the stock could {data.get('dramatic_prediction', 'face major volatility')}."
        ]
        return " ".join(random.sample(paragraphs, k=4))


class SocialMediaGenerator:
    """Generates social media posts including coordinated campaigns."""
    
    def __init__(self, num_groups: int, accounts_per_group: int):
        self.fake = Faker()
        self.coordinated_groups = self._create_coordinated_groups(num_groups, accounts_per_group)
        self.regular_accounts = [self._create_account(coordinated=False) for _ in range(500)]
    
    def _create_account(self, coordinated: bool = False, group_id: int = None) -> Dict:
        """Create a social media account."""
        account_id = self.fake.uuid4()
        
        if coordinated:
            # Bot-like patterns for coordinated accounts
            username = f"investor_{random.randint(1000, 9999)}"
            followers = random.randint(50, 500)  # Lower followers
            verified = False
            account_age_days = random.randint(30, 180)  # Newer accounts
        else:
            username = self.fake.user_name()
            followers = random.randint(100, 10000)
            verified = random.random() < 0.1
            account_age_days = random.randint(365, 3650)
        
        return {
            'account_id': account_id,
            'username': username,
            'followers': followers,
            'verified': verified,
            'account_age_days': account_age_days,
            'coordinated': coordinated,
            'group_id': group_id if coordinated else None
        }
    
    def _create_coordinated_groups(self, num_groups: int, accounts_per_group: int) -> List[List[Dict]]:
        """Create coordinated manipulation groups."""
        groups = []
        for group_id in range(num_groups):
            group = [self._create_account(coordinated=True, group_id=group_id) 
                    for _ in range(accounts_per_group)]
            groups.append(group)
        return groups
    
    def generate_post(self, article: Dict, account: Dict, date: datetime, 
                     coordinated: bool = False) -> Dict:
        """Generate a social media post about an article."""
        
        if coordinated and article['label'] == 'fake':
            # Coordinated posts have similar patterns
            templates = [
                f"🚨 MUST READ: {article['headline'][:80]}...",
                f"Everyone needs to see this about {article['ticker']} 👇",
                f"Why isn't mainstream media covering this?? {article['ticker']}",
                f"BREAKING NEWS on {article['ticker']} - link in bio",
            ]
            hashtags = f"#{article['ticker']} #StockAlert #InvestorAlert #MarketNews"
        else:
            # Regular posts are more varied
            templates = [
                f"Interesting article on {article['ticker']}: {article['headline'][:60]}...",
                f"Thoughts on {article['ticker']} news?",
                f"Analysis: {article['headline'][:70]}",
                f"Latest on {article['ticker']}",
            ]
            hashtags = f"#{article['ticker']} #investing #{random.choice(['stocks', 'finance', 'trading'])}"
        
        text = random.choice(templates) + " " + hashtags
        
        # Add slight time variation for coordinated posts (within 2 hours)
        if coordinated:
            time_offset = timedelta(minutes=random.randint(0, 120))
        else:
            time_offset = timedelta(hours=random.randint(0, 48))
        
        post_date = date + time_offset
        
        return {
            'post_id': self.fake.uuid4(),
            'account_id': account['account_id'],
            'username': account['username'],
            'text': text,
            'article_id': article['article_id'],
            'posted_date': post_date,
            'likes': random.randint(0, 1000) if not coordinated else random.randint(100, 500),
            'retweets': random.randint(0, 500) if not coordinated else random.randint(50, 200),
            'ticker': article['ticker'],
            'coordinated': coordinated,
            'group_id': account.get('group_id')
        }


class StockPriceGenerator:
    """Generates stock price data with volatility correlated to news events."""
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.base_prices = {ticker: random.uniform(50, 500) for ticker in tickers}
    
    def generate_prices(self, start_date: datetime, end_date: datetime,
                       news_events: pd.DataFrame) -> pd.DataFrame:
        """Generate stock prices with correlation to news events."""
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        all_prices = []
        
        for ticker in self.tickers:
            price = self.base_prices[ticker]
            
            # Get news events for this ticker
            ticker_news = news_events[news_events['ticker'] == ticker].copy()
            ticker_news['date'] = pd.to_datetime(ticker_news['published_date']).dt.date
            news_by_date = ticker_news.groupby('date').agg({
                'label': lambda x: (x == 'fake').sum(),  # Count of fake news
                'article_id': 'count'  # Total news count
            }).rename(columns={'label': 'fake_count', 'article_id': 'total_count'})
            
            for date in date_range:
                # Normal daily return
                daily_return = np.random.normal(0.0005, 0.02)  # 0.05% mean, 2% std
                
                # Check for news events
                date_only = date.date()
                if date_only in news_by_date.index:
                    news_data = news_by_date.loc[date_only]
                    fake_count = news_data['fake_count']
                    total_count = news_data['total_count']
                    
                    # Fake news creates abnormal volatility
                    if fake_count > 0:
                        # Add extra volatility and potential manipulation
                        direction = random.choice([-1, 1])
                        manipulation_effect = direction * random.uniform(0.03, 0.08)  # 3-8% move
                        daily_return += manipulation_effect
                        
                        # Increase volatility
                        daily_return += np.random.normal(0, 0.03)
                    
                    # Real news has moderate impact
                    elif total_count > 3:
                        daily_return += np.random.normal(0, 0.015)
                
                # Update price
                price *= (1 + daily_return)
                
                # Calculate technical indicators
                volume = int(random.uniform(1_000_000, 10_000_000))
                if date_only in news_by_date.index:
                    volume *= random.uniform(1.5, 3.0)  # Higher volume on news days
                
                all_prices.append({
                    'ticker': ticker,
                    'date': date,
                    'open': price * random.uniform(0.98, 1.02),
                    'high': price * random.uniform(1.00, 1.03),
                    'low': price * random.uniform(0.97, 1.00),
                    'close': price,
                    'volume': volume
                })
        
        return pd.DataFrame(all_prices)


def generate_all_data():
    """Main function to generate all simulated data."""
    
    print("🚀 Starting data generation...")
    print(f"Configuration: {DATA_GENERATION['num_articles']} articles, "
          f"{DATA_GENERATION['num_social_posts']} posts, "
          f"{DATA_GENERATION['date_range_days']} days")
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DATA_GENERATION['date_range_days'])
    
    # Initialize generators
    news_gen = FinancialNewsGenerator()
    social_gen = SocialMediaGenerator(
        DATA_GENERATION['coordinated_groups'],
        DATA_GENERATION['accounts_per_group']
    )
    stock_gen = StockPriceGenerator(STOCK_TICKERS)
    
    # Generate news articles
    print("\n📰 Generating news articles...")
    articles = []
    num_fake = int(DATA_GENERATION['num_articles'] * DATA_GENERATION['fake_news_ratio'])
    num_real = DATA_GENERATION['num_articles'] - num_fake
    
    for i in range(num_real):
        ticker = random.choice(STOCK_TICKERS)
        date = fake.date_time_between(start_date=start_date, end_date=end_date)
        articles.append(news_gen.generate_real_article(ticker, date))
    
    for i in range(num_fake):
        ticker = random.choice(STOCK_TICKERS)
        date = fake.date_time_between(start_date=start_date, end_date=end_date)
        articles.append(news_gen.generate_fake_article(ticker, date))
    
    articles_df = pd.DataFrame(articles)
    print(f"✅ Generated {len(articles_df)} articles ({num_fake} fake, {num_real} real)")
    
    # Generate social media posts
    print("\n💬 Generating social media posts...")
    posts = []
    
    # Coordinated posts for fake news
    fake_articles = articles_df[articles_df['label'] == 'fake'].to_dict('records')
    for article in fake_articles[:50]:  # Top 50 fake articles get coordinated campaigns
        group = random.choice(social_gen.coordinated_groups)
        for account in group:
            post = social_gen.generate_post(
                article, account,
                article['published_date'],
                coordinated=True
            )
            posts.append(post)
    
    # Regular posts
    remaining_posts = DATA_GENERATION['num_social_posts'] - len(posts)
    for _ in range(remaining_posts):
        article = random.choice(articles)
        account = random.choice(social_gen.regular_accounts)
        post = social_gen.generate_post(article, account, article['published_date'])
        posts.append(post)
    
    posts_df = pd.DataFrame(posts)
    coordinated_count = posts_df['coordinated'].sum()
    print(f"✅ Generated {len(posts_df)} posts ({coordinated_count} coordinated)")
    
    # Generate stock prices
    print("\n📈 Generating stock prices...")
    prices_df = stock_gen.generate_prices(start_date, end_date, articles_df)
    print(f"✅ Generated {len(prices_df)} price records for {len(STOCK_TICKERS)} stocks")
    
    # Create user accounts dataframe
    all_accounts = social_gen.regular_accounts + [acc for group in social_gen.coordinated_groups for acc in group]
    accounts_df = pd.DataFrame(all_accounts)
    
    # Save all data
    print("\n💾 Saving data...")
    articles_df.to_csv(PATHS['news_articles'], index=False)
    print(f"✅ Saved articles to {PATHS['news_articles']}")
    
    posts_df.to_csv(PATHS['social_posts'], index=False)
    print(f"✅ Saved posts to {PATHS['social_posts']}")
    
    prices_df.to_csv(PATHS['stock_prices'], index=False)
    print(f"✅ Saved prices to {PATHS['stock_prices']}")
    
    accounts_df.to_csv(PATHS['user_accounts'], index=False)
    print(f"✅ Saved accounts to {PATHS['user_accounts']}")
    
    # Print summary statistics
    print("\n📊 Data Generation Summary:")
    print("=" * 50)
    print(f"Articles: {len(articles_df)} total")
    print(f"  - Real: {len(articles_df[articles_df['label']=='real'])} ({len(articles_df[articles_df['label']=='real'])/len(articles_df)*100:.1f}%)")
    print(f"  - Fake: {len(articles_df[articles_df['label']=='fake'])} ({len(articles_df[articles_df['label']=='fake'])/len(articles_df)*100:.1f}%)")
    print(f"\nSocial Posts: {len(posts_df)} total")
    print(f"  - Coordinated: {coordinated_count} ({coordinated_count/len(posts_df)*100:.1f}%)")
    print(f"  - Regular: {len(posts_df)-coordinated_count}")
    print(f"\nUser Accounts: {len(accounts_df)} total")
    print(f"  - Coordinated: {len(accounts_df[accounts_df['coordinated']==True])}")
    print(f"  - Regular: {len(accounts_df[accounts_df['coordinated']==False])}")
    print(f"\nStock Prices: {len(prices_df)} records")
    print(f"  - Date range: {prices_df['date'].min()} to {prices_df['date'].max()}")
    print(f"  - Tickers: {len(STOCK_TICKERS)}")
    print("=" * 50)
    print("\n✨ Data generation complete!")
    
    return articles_df, posts_df, prices_df, accounts_df


if __name__ == "__main__":
    generate_all_data()
