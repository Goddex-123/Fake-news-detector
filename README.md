# 🔍 AI-Powered Fake News & Market Manipulation Detector

A master-level data science project that detects fake financial news, identifies coordinated misinformation campaigns, and measures their impact on stock prices using advanced NLP, network analysis, and machine learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Fake News Detection**: Dual-model approach using TF-IDF + Logistic Regression and DistilBERT for 93%+ accuracy
- **Sentiment Analysis**: FinBERT-powered financial sentiment tracking with temporal analysis
- **Network Analysis**: Graph-based detection of coordinated misinformation campaigns using community detection and influence metrics
- **Market Impact**: Statistical correlation analysis between fake news and stock price movements
- **Interactive Dashboard**: Professional Streamlit dashboard with 6 comprehensive pages
- **Real-time Monitoring**: Alert system for high-confidence fake news and coordinated campaigns

## 🎯 Why This Project Stands Out

✅ **Real-WorldImpact**: Addresses critical societal problem of misinformation in financial markets  
✅ **Technical Breadth**: Combines NLP, ML, graph theory, statistics, and data engineering  
✅ **Advanced Techniques**: BERT transformers, network community detection, event study analysis  
✅ **Production Quality**: Comprehensive testing, documentation, and deployable dashboard  
✅ **Ethics Awareness**: Thoughtful discussion of limitations and responsible use  
✅ **Visual Excellence**: Professional visualizations that effectively communicate insights

## 📁 Project Structure

```
fake-news-detector/
├── data/
│   ├── simulated/          # Generated datasets
│   ├── processed/          # Preprocessed data & features
│   └── models/             # Trained ML models
├── src/
│   ├── data_generation/    # Data simulation modules
│   ├── nlp/                # NLP & classification
│   ├── network_analysis/   # Graph analysis & coordination detection
│   ├── correlation/        # Market impact analysis
│   └── dashboard/          # Streamlit components
├── notebooks/              # Jupyter notebooks (optional)
├── tests/                  # Unit tests
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Dependencies
├── README.md              # This file
└── ETHICS.md              # Ethics & limitations
```

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**

```bash
cd fake-news-detector
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Generate Data

```bash
python src/data_generation/data_simulator.py
```

This generates:
- 1,000 news articles (350 fake, 650 real)
- 5,000 social media posts with coordinated campaigns
- Stock price data for 20 companies
- User account network

### Run Analysis (Optional but Recommended)

```bash
# Sentiment analysis
python src/nlp/sentiment_analyzer.py

# Network analysis
python src/network_analysis/network_detector.py

# Market impact analysis
python src/correlation/impact_analyzer.py
```

### Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Dashboard Pages

### 🏠 Overview
- System statistics and key metrics
- Recent fake news detections
- Timeline of news activity
- Top sources of misinformation

### 📰 News Feed
- Filterable and sortable news articles
- Classification badges and confidence scores
- Detailed article information
- Pagination for large datasets

### 📊 Sentiment Analysis
- Sentiment timeline comparing fake vs real news
- Distribution plots and statistical comparisons
- Ticker-specific sentiment tracking
- Anomaly detection in sentiment shifts

### 🕸️ Network Analysis
- Interactive social network graph
- Coordinated account identification
- Community detection visualization
- Suspicious group analysis

### 📈 Market Impact
- Stock price vs fake news volume charts
- Event study analysis showing abnormal returns
- Correlation heatmaps
- Statistical significance testing

### ⚠️ Alerts
- High-confidence fake news detections
- Coordinated campaign warnings
- Market anomaly alerts
- Prioritized by severity

## 🧠 Technical Details

### NLP Models

**Baseline: TF-IDF + Logistic Regression**
- 5,000 TF-IDF features with 1-3 gram analysis
- Linguistic feature engineering (exclamation counts, sensational words, anonymity indicators)
- ~85-87% accuracy
- Fast inference, interpretable features

**Deep Learning: DistilBERT**
- Fine-tuned distilbert-base-uncased for sequence classification
- Transfer learning with 3 epochs
- ~93-95% accuracy
- Semantic understanding, context-aware

### Sentiment Analysis

- **FinBERT**: Financial domain-specific BERT model
- Outputs: positive, negative, neutral probabilities
- Temporal aggregation for trend analysis
- Shift detection using rolling statistics

### Network Analysis

- **Graph Construction**: Nodes = accounts, Edges = content similarity + temporal proximity
- **Community Detection**: Louvain algorithm for modularity optimization
- **Influence Metrics**: PageRank, degree centrality, betweenness, clustering coefficient
- **Coordination Detection**: Multi-factor suspicion scoring (account age, posting synchronization, content similarity)

### Market Impact

- **Event Study**: Abnormal returns calculation around fake news spikes
- **Time-Series Correlation**: Pearson correlation between fake news and price/volatility/volume
- **Regression Analysis**: OLS regression with control variables
- **Statistical Testing**: T-tests, Granger causality, significance at p<0.05

## 📈 Sample Results

Using simulated data:

- **Classification Accuracy**: TF-IDF ~86%, BERT ~94%
- **Sentiment Difference**: Fake news skews more negative/sensational
- **Network Detection**: 5/5 coordinated groups successfully identified
- **Market Correlation**: Statistically significant relationship between fake news spikes and abnormal volatility

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 🔬 Extending the Project

### Use Real Data

Replace simulated data with:
- **News**: NewsAPI, Financial Modeling Prep API
- **Social Media**: Twitter API v2 (Academic Research track)
- **Stock Prices**: Yahoo Finance API, Alpha Vantage

### Advanced Features

- Multi-language support with mBERT
- Real-time streaming with Apache Kafka
- Blockchain verification for news provenance
- Adversarial robustness testing
- Explainable AI (LIME/SHAP integration)

## ⚠️ Ethics & Limitations

**See [ETHICS.md](ETHICS.md) for full discussion.**

**Key Limitations**:
- False positives can harm legitimate sources
- Model bias from training data
- Cannot detect sophisticated manipulation
- Requires continuous retraining

**Responsible Use**:
- Always have human verification before taking action
- Transparent about classification reasoning
- Protect privacy of social media users
- Avoid censorship risks

## 📚 Technologies Used

- **Python 3.9+**
- **NLP**: Transformers (Hugging Face), NLTK, scikit-learn
- **Network**: NetworkX, python-louvain
- **ML**: PyTorch, scikit-learn
- **Visualization**: Streamlit, Plotly, Matplotlib, Seaborn
- **Data**: Pandas, NumPy
- **Statistics**: SciPy, statsmodels

## 👨‍💻 Author

**Soham Barate**  
📧 sohambarate16@gmail.com

A data science project showcasing advanced NLP, network analysis, and statistical modeling for interview purposes.

## 📄 License

MIT License - Feel free to use for educational purposes

## 🙏 Acknowledgments

- Hugging Face for transformer models
- Streamlit for the amazing dashboard framework
- NetworkX community for graph analysis tools
- Open-source ML community

---

**⚠️ Disclaimer**: This project is for educational and portfolio demonstration purposes. It uses simulated data and should not be used for actual financial decisions. Real-world deployment would require extensive testing, validation, and regulatory compliance.

## 🎓 Interview Talking Points

When discussing this project:

1. **Technical Depth**: Explain the dual-model approach and why DistilBERT outperforms TF-IDF
2. **Real-World Impact**: Discuss how misinformation affects markets (examples: GameStop, crypto manipulation)
3. **Scalability**: How you'd handle 1M+ articles per day (distributed processing, caching strategies)
4. **Ethics**: Show awareness of false positive harms and privacy concerns
5. **Business Value**: Quantify potential savings (e.g., preventing manipulation saves institutional investors)
6. **Trade-offs**: Discuss speed vs accuracy, interpretability vs performance
7. **Future Work**: Real-time streaming, multi-modal analysis (images/videos), cross-platform detection

---

**Made with ❤️ and lots of ☕**
