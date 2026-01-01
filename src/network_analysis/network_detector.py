"""
Network Detection Module

Builds social network graphs and detects coordinated misinformation campaigns.
Uses community detection, influence metrics, and coordination patterns.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import community as community_louvain  # python-louvain
from datetime import timedelta
from collections import defaultdict
import pickle
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config import NETWORK_CONFIG, PATHS


class NetworkBuilder:
    """Builds social network graph from user interactions and content."""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.directed_graph = nx.DiGraph()  # For retweet/share relationships
    
    def build_from_posts(self, posts_df: pd.DataFrame, accounts_df: pd.DataFrame) -> nx.Graph:
        """Build network graph from social media posts."""
        
        print("🕸️ Building social network graph...")
        
        # Add nodes (accounts)
        print(f"Adding {len(accounts_df)} account nodes...")
        for _, account in accounts_df.iterrows():
            self.graph.add_node(
                account['account_id'],
                username=account['username'],
                followers=account['followers'],
                verified=account['verified'],
                account_age=account['account_age_days'],
                coordinated=account['coordinated'],
                group_id=account.get('group_id')
            )
        
        # Add edges based on:
        # 1. Posting about same articles within time window
        # 2. Content similarity
        
        print("Analyzing article co-posting patterns...")
        article_groups = posts_df.groupby('article_id')['account_id'].apply(list).to_dict()
        
        edge_weights = defaultdict(int)
        
        for article_id, account_ids in article_groups.items():
            # Get timestamps for these posts
            article_posts = posts_df[posts_df['article_id'] == article_id]
            
            # Connect accounts that posted about same article
            for i, acc1 in enumerate(account_ids):
                for acc2 in account_ids[i+1:]:
                    # Check temporal proximity
                    time1 = pd.to_datetime(article_posts[article_posts['account_id'] == acc1]['posted_date'].iloc[0])
                    time2 = pd.to_datetime(article_posts[article_posts['account_id'] == acc2]['posted_date'].iloc[0])
                    
                    time_diff_hours = abs((time1 - time2).total_seconds() / 3600)
                    
                    # If posted within temporal window, strengthen connection
                    if time_diff_hours <= NETWORK_CONFIG['temporal_window_hours']:
                        weight = 1.0 + (1.0 / (1.0 + time_diff_hours))  # Closer in time = higher weight
                        edge_weights[(acc1, acc2)] += weight
        
        print("Analyzing content similarity...")
        # Content similarity edges
        account_texts = posts_df.groupby('account_id')['text'].apply(' '.join).to_dict()
        account_ids = list(account_texts.keys())
        texts = [account_texts[aid] for aid in account_ids]
        
        if len(texts) > 1:
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Add similarity edges
            for i, acc1 in enumerate(account_ids):
                for j, acc2 in enumerate(account_ids[i+1:], start=i+1):
                    sim = similarity_matrix[i, j]
                    if sim > NETWORK_CONFIG['similarity_threshold']:
                        edge_weights[(acc1, acc2)] += sim
        
        # Add all edges to graph
        print(f"Adding {len(edge_weights)} weighted edges...")
        for (acc1, acc2), weight in edge_weights.items():
            self.graph.add_edge(acc1, acc2, weight=weight)
        
        print(f"✅ Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def save_graph(self, path):
        """Save graph to file."""
        nx.write_gpickle(self.graph, path)
        print(f"✅ Graph saved to {path}")
    
    @staticmethod
    def load_graph(path):
        """Load graph from file."""
        return nx.read_gpickle(path)


class CoordinationDetector:
    """Detects coordinated manipulation campaigns in the network."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.communities = None
        self.influence_scores = None
    
    def detect_communities(self):
        """Detect communities using Louvain algorithm."""
        print("\n🔍 Detecting communities...")
        
        # Louvain community detection
        self.communities = community_louvain.best_partition(self.graph)
        
        # Get community sizes
        community_sizes = defaultdict(int)
        for node, comm_id in self.communities.items():
            community_sizes[comm_id] += 1
        
        print(f"Found {len(community_sizes)} communities")
        print(f"Largest community: {max(community_sizes.values())} nodes")
        print(f"Smallest community: {min(community_sizes.values())} nodes")
        
        return self.communities
    
    def calculate_influence_metrics(self):
        """Calculate influence metrics for each node."""
        print("\n📊 Calculating influence metrics...")
        
        self.influence_scores = {}
        
        # PageRank - measures overall influence
        pagerank = nx.pagerank(self.graph, weight='weight')
        
        # Degree centrality - number of connections
        degree_centrality = nx.degree_centrality(self.graph)
        
        # Betweenness centrality - bridging different groups
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        
        # Clustering coefficient - how clustered their neighbors are
        clustering = nx.clustering(self.graph, weight='weight')
        
        for node in self.graph.nodes():
            self.influence_scores[node] = {
                'pagerank': pagerank[node],
                'degree_centrality': degree_centrality[node],
                'betweenness': betweenness[node],
                'clustering': clustering[node],
                'degree': self.graph.degree(node)
            }
        
        print("✅ Influence metrics calculated")
        
        return self.influence_scores
    
    def identify_coordinated_groups(self, accounts_df: pd.DataFrame, posts_df: pd.DataFrame):
        """Identify suspicious coordinated groups."""
        print("\n🚨 Identifying coordinated manipulation groups...")
        
        if self.communities is None:
            self.detect_communities()
        
        if self.influence_scores is None:
            self.calculate_influence_metrics()
        
        # Group nodes by community
        community_groups = defaultdict(list)
        for node, comm_id in self.communities.items():
            community_groups[comm_id].append(node)
        
        suspicious_groups = []
        
        for comm_id, members in community_groups.items():
            if len(members) < NETWORK_CONFIG['min_community_size']:
                continue
            
            # Get account details for this community
            member_accounts = accounts_df[accounts_df['account_id'].isin(members)]
            
            # Calculate suspicion indicators
            metrics = self._calculate_group_suspicion_metrics(
                members, member_accounts, posts_df
            )
            
            if metrics['suspicion_score'] > 0.5:  # Threshold for suspicious
                suspicious_groups.append({
                    'community_id': comm_id,
                    'members': members,
                    'size': len(members),
                    **metrics
                })
        
        print(f"✅ Found {len(suspicious_groups)} suspicious coordinated groups")
        
        # Sort by suspicion score
        suspicious_groups.sort(key=lambda x: x['suspicion_score'], reverse=True)
        
        return suspicious_groups
    
    def _calculate_group_suspicion_metrics(self, members: list, 
                                          member_accounts: pd.DataFrame,
                                          posts_df: pd.DataFrame) -> dict:
        """Calculate metrics indicating coordination."""
        
        metrics = {}
        
        # 1. Account age - newer accounts are more suspicious
        avg_account_age = member_accounts['account_age_days'].mean()
        metrics['avg_account_age_days'] = avg_account_age
        metrics['new_account_ratio'] = (member_accounts['account_age_days'] < 180).sum() / len(members)
        
        # 2. Verified ratio - fewer verified accounts is more suspicious
        metrics['verified_ratio'] = member_accounts['verified'].sum() / len(members)
        
        # 3. Follower distribution - similar follower counts suggests bots
        follower_std = member_accounts['followers'].std()
        follower_mean = member_accounts['followers'].mean()
        metrics['follower_cv'] = follower_std / (follower_mean + 1)  # Coefficient of variation
        
        # 4. Posting synchronization - posts at similar times
        member_posts = posts_df[posts_df['account_id'].isin(members)]
        member_posts['posted_date'] = pd.to_datetime(member_posts['posted_date'])
        
        # Calculate time variance for posts about same articles
        time_variances = []
        for article_id in member_posts['article_id'].unique():
            article_posts = member_posts[member_posts['article_id'] == article_id]
            if len(article_posts) >= 2:
                times = article_posts['posted_date']
                time_range_hours = (times.max() - times.min()).total_seconds() / 3600
                time_variances.append(time_range_hours)
        
        metrics['avg_post_time_variance_hours'] = np.mean(time_variances) if time_variances else 24
        
        # 5. Content similarity - very similar posts suggest coordination
        if len(member_posts) > 1:
            vectorizer = TfidfVectorizer(max_features=500)
            try:
                tfidf = vectorizer.fit_transform(member_posts['text'])
                similarity = cosine_similarity(tfidf).mean()
                metrics['content_similarity'] = similarity
            except:
                metrics['content_similarity'] = 0.5
        else:
            metrics['content_similarity'] = 0.5
        
        # 6. Calculate overall suspicion score
        suspicion_score = 0.0
        
        # New accounts indicator
        if metrics['new_account_ratio'] > 0.7:
            suspicion_score += 0.3
        
        # Low verification
        if metrics['verified_ratio'] < 0.1:
            suspicion_score += 0.2
        
        # Similar follower counts (bot-like)
        if metrics['follower_cv'] < 0.3:
            suspicion_score += 0.2
        
        # Synchronized posting
        if metrics['avg_post_time_variance_hours'] < 2:
            suspicion_score += 0.2
        
        # High content similarity
        if metrics['content_similarity'] > 0.7:
            suspicion_score += 0.3
        
        # Check if actually labeled as coordinated (for ground truth comparison)
        actual_coordinated_ratio = member_accounts['coordinated'].sum() / len(members)
        metrics['actual_coordinated_ratio'] = actual_coordinated_ratio
        
        metrics['suspicion_score'] = min(suspicion_score, 1.0)
        
        return metrics
    
    def get_influential_nodes(self, top_n=20):
        """Get top influential nodes by PageRank."""
        if self.influence_scores is None:
            self.calculate_influence_metrics()
        
        # Sort by PageRank
        sorted_nodes = sorted(
            self.influence_scores.items(),
            key=lambda x: x[1]['pagerank'],
            reverse=True
        )
        
        return sorted_nodes[:top_n]


def analyze_network(posts_df: pd.DataFrame, accounts_df: pd.DataFrame):
    """Main function to build and analyze network."""
    
    # Build network
    builder = NetworkBuilder()
    graph = builder.build_from_posts(posts_df, accounts_df)
    
    # Save graph
    builder.save_graph(PATHS['network_graph'])
    
    # Detect coordination
    detector = CoordinationDetector(graph)
    communities = detector.detect_communities()
    influence = detector.calculate_influence_metrics()
    suspicious_groups = detector.identify_coordinated_groups(accounts_df, posts_df)
    
    # Print results
    print("\n" + "="*50)
    print("🎯 Top Suspicious Groups")
    print("="*50)
    
    for i, group in enumerate(suspicious_groups[:5], 1):
        print(f"\nGroup {i} (Community {group['community_id']}):")
        print(f"  Size: {group['size']} accounts")
        print(f"  Suspicion Score: {group['suspicion_score']:.2f}")
        print(f"  Actually Coordinated: {group['actual_coordinated_ratio']*100:.1f}%")
        print(f"  Avg Account Age: {group['avg_account_age_days']:.0f} days")
        print(f"  Content Similarity: {group['content_similarity']:.2f}")
        print(f"  Post Time Variance: {group['avg_post_time_variance_hours']:.1f} hours")
    
    # Top influencers
    top_influencers = detector.get_influential_nodes(top_n=10)
    print("\n" + "="*50)
    print("👑 Top Influential Accounts")
    print("="*50)
    
   for i, (node_id, metrics) in enumerate(top_influencers, 1):
        username = graph.nodes[node_id].get('username', 'Unknown')
        print(f"{i}. {username}")
        print(f"   PageRank: {metrics['pagerank']:.4f}")
        print(f"   Connections: {metrics['degree']}")
    
    return graph, communities, suspicious_groups


if __name__ == "__main__":
    # Load data
    print("Loading data...")
    posts_df = pd.read_csv(PATHS['social_posts'])
    accounts_df = pd.read_csv(PATHS['user_accounts'])
    
    # Analyze network
    graph, communities, suspicious_groups = analyze_network(posts_df, accounts_df)
