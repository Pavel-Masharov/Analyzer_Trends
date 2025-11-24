from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

from .rag_config import RAGConfig


class SocialPlatform(str, Enum):
    TELEGRAM = "telegram"
    VK = "vk"


class DataSourceConfig(BaseModel):
    platform: SocialPlatform
    api_key: Optional[str] = None
    sources: List[str]
    enabled: bool = True
    rate_limit: int = Field(10, ge=1)


class TrendAnalysisConfig(BaseModel):
    min_trend_confidence: float = Field(0.7, ge=0.0, le=1.0)
    max_trends_per_day: int = Field(5, ge=1, le=20)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clustering_algorithm: str = "kmeans"


class AppConfig(BaseModel):
    data_sources: List[DataSourceConfig]
    trend_analysis: TrendAnalysisConfig

    rag_config: RAGConfig = Field(default_factory=RAGConfig)
