from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

from src.common.rag_models import HistoricalTrend, VectorSearchResult


class BaseVectorStore(ABC):
    """Class for vector stores that support chunks"""

    @abstractmethod
    async def initialize(self):
        pass

    @abstractmethod
    async def add_trend(self, trend: HistoricalTrend) -> str:
        pass

    @abstractmethod
    async def add_trends_batch(self, trends: List[HistoricalTrend]) -> List[str]:
        pass

    @abstractmethod
    async def find_similar(
            self,
            embedding: List[float],
            k: int = 5,
            similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """Finds k similar trends"""
        pass

    @abstractmethod
    async def delete_trend(self, trend_id: str) -> bool:
        pass

    @abstractmethod
    async def get_trend_stats(self) -> dict:
        pass

    @abstractmethod
    async def add_chunk(self, chunk: Dict[str, Any], trend_id: str) -> str:
        """
        Adds a chunk to the vector store

        Args:
            chunk: Dictionary with chunk data {content, metadata}
            trend_id: ID of the parent trend

        Returns:
             ID of the created chunk
        """
        pass

    @abstractmethod
    async def add_trend_with_chunks(self, trend: HistoricalTrend, chunks: List[Dict[str, Any]]) -> str:
        """
        Adds a trend along with its chunks

        Args:
            trend: Historical trend
            chunks:  List of trend chunks

        Returns:
             ID of the trend
        """
        pass

    @abstractmethod
    async def find_similar_chunks(
            self,
            embedding: List[float],
            k: int = 5,
            similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
         Finds k similar chunks

        Args:
            embedding: Search vector
            k: Number of results
            similarity_threshold: Similarity threshold

        Returns:
             List of dictionaries with chunks and metadata
        """
        pass

    @abstractmethod
    async def cleanup_old_trends(self, max_days: int = 30):
        """
        Deletes trends older than max_days days

        Args:
            max_days: Maximum age of trends in days
        """
        pass
