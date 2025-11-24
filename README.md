# 🚀 Social Media Trends Analyzer

**AI-powered system for automatic trend detection in social networks using advanced ML and RAG architecture**

![Python](https://img.shields.io/badge/Python-3.11-blue)

## 📊 Project Overview

An intelligent system that automatically discovers and analyzes emerging trends across social media platforms using cutting-edge Machine Learning and Retrieval-Augmented Generation techniques.

### ✨ Key Features

- **📡 Multi-Source Data Collection** - Real-time data from VK, Telegram with smart rate limiting
- **🤖 Advanced ML Analysis** - Semantic clustering and theme extraction using transformer models
- **🧠 Custom RAG System** - Historical context enrichment through vector similarity search
- **⚡ High Performance** - Optimized for low-latency inference (<500ms)
- **📈 Engagement Analytics** - Trend velocity and growth rate calculations

## 🏗️ System Architecture
┌─────────────────┐ ┌──────────────────┐ ┌─────────────────┐
│ Data Collector │ ──> │ Trend Analyzer │ ──> │ RAG Engine │
│ • VK API │ │ • Embeddings │ │ • FAISS Vector │
│ │ • Clustering │ │ • Similarity │
└─────────────────┘ │ • Theme Extraction│ └─────────────────┘
└──────────────────┘
│
┌─────────────────┐
│ Trend Output │
│ • Confidence │
│ • Analytics │
└─────────────────┘



## 🛠️ Tech Stack

### Core ML & NLP
- **`sentence-transformers`** - Semantic text embeddings
- **`scikit-learn`** - Advanced clustering algorithms
- **`PyTorch`** - Deep learning backend
- **`numpy/pandas`** - Data processing and analysis

### Infrastructure
- **`aiohttp`** - Asynchronous HTTP requests
- **`pydantic`** - Data validation and configuration
- **`FAISS`** - High-performance vector search
- **`loguru`** - Structured logging

## 🚀 Quick Start

git clone 

pip install -r requirements.txt

#Add your VK_API_TOKEN to .env file

VK_API_TOKEN=your_vk_api_token_here

#Edit configs/config.py

data_sources=[
    DataSourceConfig(

        platform=SocialPlatform.VK,
        api_key="your_vk_token_here",
        sources=[
            "habr",              # IT community
            "tproger",           # Programming
            "tech",              # Technology
            "opennet",           # Open Source
            # Add more VK groups...
        ],
        enabled=True
    )
]


python main.py


## 📈 Settings
#Edit main.py

COLLECTION_HOURS = 24 #how many hours does it take to collect posts from social networks

MIN_CLUSTER_SIZE = 15 #минимальное количество постов для образования тренда


