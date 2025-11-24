from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional
from .models import SocialPost, TrendCluster


class HistoricalTrend(BaseModel):
    """Model for storing historical trends DB"""
    id: str = Field(..., description="Уникальный ID исторического тренда")
    theme: str = Field(..., description="Тема тренда")
    posts: List[SocialPost] = Field(..., description="Посты в этом тренде")
    confidence: float = Field(..., description="Уверенность в тренде")
    timestamp: datetime = Field(..., description="Когда был обнаружен тренд")
    embedding: List[float] = Field(..., description="Векторное представление темы")
    metadata: Dict = Field(
        default_factory=dict,
        description="Дополнительные метаданные: платформы, авторы, engagement"
    )


    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrendEnrichment(BaseModel):
    """Result of enriching trend through RAG"""
    current_trend: TrendCluster = Field(..., description="Исходный тренд")
    similar_historical_trends: List[HistoricalTrend] = Field(
        default_factory=list,
        description="Похожие исторические тренды (топ-5)"
    )
    enrichment_data: Dict = Field(
        default_factory=dict,
        description="Дополнительная аналитика: скорость роста, контекст и т.д."
    )


    def get_similar_themes(self) -> List[str]:
        """Возвращает список тем похожих трендов"""
        return [trend.theme for trend in self.similar_historical_trends]


    def get_historical_confidence_stats(self) -> Dict:
        """Статистика по уверенности похожих трендов"""
        if not self.similar_historical_trends:
            return {}

        confidences = [trend.confidence for trend in self.similar_historical_trends]
        return {
            "avg_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
            "count": len(confidences)
        }


class VectorSearchResult(BaseModel):
    """Result search DB"""
    trend: HistoricalTrend = Field(..., description="Найденный тренд")
    similarity_score: float = Field(..., description="Косинусная схожесть (0-1)")
    rank: int = Field(..., description="Ранг в результатах поиска")