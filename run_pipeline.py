"""
Main Execution Script
Run the complete pipeline: data generation → analysis → dashboard

Usage:
    python run_pipeline.py --all              # Run complete pipeline
    python run_pipeline.py --generate         # Only generate data
    python run_pipeline.py --analyze          # Only run analysis
    python run_pipeline.py --dashboard        # Only launch dashboard
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))


def print_banner():
    """Print project banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   🔍 AI-POWERED FAKE NEWS & MARKET MANIPULATION DETECTOR 🔍   ║
    ║                                                               ║
    ║   Advanced NLP + Network Analysis + Machine Learning         ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_data_generation():
    """Generate simulated data."""
    print("\n" + "="*60)
    print("📊 STEP 1: DATA GENERATION")
    print("="*60)
    
    from src.data_generation.data_simulator import generate_all_data
    
    try:
        generate_all_data()
        print("\n✅ Data generation complete!")
        return True
    except Exception as e:
        print(f"\n❌ Data generation failed: {e}")
        return False


def run_sentiment_analysis():
    """Run sentiment analysis on news articles."""
    print("\n" + "="*60)
    print("🎭 STEP 2: SENTIMENT ANALYSIS")
    print("="*60)
    
    from config import PATHS
    from src.nlp.sentiment_analyzer import analyze_news_sentiment
    import pandas as pd
    
    try:
        if not PATHS['news_articles'].exists():
            print("❌ News articles not found. Run data generation first.")
            return False
        
        news_df = pd.read_csv(PATHS['news_articles'])
        news_with_sentiment = analyze_news_sentiment(news_df)
        news_with_sentiment.to_csv(PATHS['processed_features'], index=False)
        
        print(f"\n✅ Sentiment analysis complete! Saved to {PATHS['processed_features']}")
        return True
    except Exception as e:
        print(f"\n❌ Sentiment analysis failed: {e}")
        return False


def run_network_analysis():
    """Run network analysis for coordinated campaigns."""
    print("\n" + "="*60)
    print("🕸️  STEP 3: NETWORK ANALYSIS")
    print("="*60)
    
    from config import PATHS
    from src.network_analysis.network_detector import analyze_network
    import pandas as pd
    
    try:
        if not PATHS['social_posts'].exists():
            print("❌ Social posts not found. Run data generation first.")
            return False
        
        posts_df = pd.read_csv(PATHS['social_posts'])
        accounts_df = pd.read_csv(PATHS['user_accounts'])
        
        graph, communities, suspicious_groups = analyze_network(posts_df, accounts_df)
        
        print(f"\n✅ Network analysis complete!")
        return True
    except Exception as e:
        print(f"\n❌ Network analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_impact_analysis():
    """Run market impact correlation analysis."""
    print("\n" + "="*60)
    print("📈 STEP 4: MARKET IMPACT ANALYSIS")
    print("="*60)
    
    from config import PATHS
    from src.correlation.impact_analyzer import analyze_impact
    import pandas as pd
    
    try:
        if not PATHS['news_articles'].exists() or not PATHS['stock_prices'].exists():
            print("❌ Required data not found. Run data generation first.")
            return False
        
        news_df = pd.read_csv(PATHS['news_articles'])
        prices_df = pd.read_csv(PATHS['stock_prices'])
        
        results = analyze_impact(news_df, prices_df)
        
        # Save results
        results['event_study'].to_csv(PATHS['processed_features'].parent / 'event_study_results.csv', index=False)
        results['time_series'].to_csv(PATHS['processed_features'].parent / 'time_series_analysis.csv', index=False)
        
        print(f"\n✅ Impact analysis complete!")
        return True
    except Exception as e:
        print(f"\n❌ Impact analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def launch_dashboard():
    """Launch Streamlit dashboard."""
    print("\n" + "="*60)
    print("🚀 LAUNCHING DASHBOARD")
    print("="*60)
    print("\nStarting Streamlit server...")
    print("Dashboard will open in your browser at http://localhost:8501")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run(["streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard closed. Goodbye!")
    except Exception as e:
        print(f"\n❌ Failed to launch dashboard: {e}")
        print("\nTry running manually:")
        print("  streamlit run app.py")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="AI-Powered Fake News Detector Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all              # Run complete pipeline
  python run_pipeline.py --generate         # Only generate data
  python run_pipeline.py --analyze          # Only run analysis
  python run_pipeline.py --dashboard        # Only launch dashboard
  python run_pipeline.py --quick            # Generate data + launch dashboard (skip analysis)
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--generate', action='store_true', help='Generate data only')
    parser.add_argument('--analyze', action='store_true', help='Run analysis only')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard only')
    parser.add_argument('--quick', action='store_true', help='Quick start (data + dashboard)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine what to run
    run_all = args.all or (not any([args.generate, args.analyze, args.dashboard, args.quick]))
    
    success = True
    
    # Data generation
    if run_all or args.generate or args.quick:
        if not run_data_generation():
            success = False
            if not args.dashboard:
                return
    
    # Analysis steps
    if run_all or args.analyze:
        if not run_sentiment_analysis():
            success = False
        
        if not run_network_analysis():
            success = False
        
        if not run_impact_analysis():
            success = False
    
    # Dashboard
    if run_all or args.dashboard or args.quick:
        if success or args.dashboard:
            launch_dashboard()
        else:
            print("\n⚠️  Some analysis steps failed. You can still launch the dashboard with:")
            print("  python run_pipeline.py --dashboard")
    
    # Final message
    if run_all and success:
        print("\n" + "="*60)
        print("🎉 PIPELINE COMPLETE!")
        print("="*60)
        print("\n✅ All steps completed successfully!")
        print("\nTo launch the dashboard later:")
        print("  streamlit run app.py")
        print("\nOr use:")
        print("  python run_pipeline.py --dashboard")


if __name__ == "__main__":
    main()
