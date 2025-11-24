import os
from .base_config import AppConfig, DataSourceConfig, TrendAnalysisConfig, SocialPlatform
from .rag_config import RAGConfig
from dotenv import load_dotenv


load_dotenv()

app_config = AppConfig(
    data_sources=[
        DataSourceConfig(
            platform=SocialPlatform.VK,
            api_key=os.getenv('VK_API_TOKEN'),
            sources=[
                "habr",
                "tproger",
                "tech",
                "opennet",
                "linux_org_ru",
                "python_django_programirovanie",
                "pythonboost",
                "php2all",
                "techno_yandex",
                "public200673827"
            ],
            enabled=True,
            rate_limit=5
        ),
        DataSourceConfig(
            platform=SocialPlatform.TELEGRAM,
            api_key="telegram_mock",
            sources=["@ai_journal", "@tech_feed", "@data_science_ru"],
            enabled=False,
            rate_limit=10
        )
    ],

    trend_analysis=TrendAnalysisConfig(
        min_trend_confidence=0.7,
        max_trends_per_day=10,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        clustering_algorithm="kmeans"
    ),

    rag_config=RAGConfig(
        vector_store_path="data/vector_store",
        similarity_threshold=0.3,
        max_similar_trends=10
    )
)