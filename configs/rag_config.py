from pydantic import BaseModel, Field


class RAGConfig(BaseModel):
    """Config RAG system"""
    vector_store_path: str = Field("data/vector_store", description="Путь к векторной БД")
    similarity_threshold: float = Field(0.7, description="Порог схожести для поиска")
    max_similar_trends: int = Field(5, description="Максимум похожих трендов")
    min_trend_similarity: float = Field(0.3, description="Минимальная схожесть для учета")

    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Модель для эмбеддингов"
    )

    save_historical_trends: bool = Field(True, description="Сохранять ли исторические тренды")
    cleanup_old_trends: bool = Field(True, description="Удалять старые тренды")
    max_history_days: int = Field(30, description="Хранить тренды за N дней")
