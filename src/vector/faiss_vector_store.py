import faiss
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import asyncio
from loguru import logger

from .base_vector_store import BaseVectorStore
from src.common.rag_models import HistoricalTrend, VectorSearchResult
from configs.rag_config import RAGConfig


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector storage with chunk support"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.index = None
        self.trends_map = {}
        self.chunks_map = {}
        self.ids_list = []
        self.dimension = 384

        Path(self.config.vector_store_path).mkdir(parents=True, exist_ok=True)


    async def initialize(self):
        """Initializing the FAISS index"""

        try:
            await self._load_from_disk()
            logger.success("FAISS vector storage is loaded")
        except Exception as e:
            await self._create_new_index()
            logger.info("A new FAISS repository has been created")


    async def _create_new_index(self):
        """Creates a new FAISS index"""

        self.index = faiss.IndexFlatIP(self.dimension)
        self.trends_map = {}
        self.chunks_map = {}
        self.ids_list = []


    async def _load_from_disk(self):
        """Loads the index and data from the disk"""

        index_path = Path(self.config.vector_store_path) / "faiss.index"
        data_path = Path(self.config.vector_store_path) / "trends_data.pkl"

        if not index_path.exists() or not data_path.exists():
            raise FileNotFoundError("Файлы хранилища не найдены")

        self.index = faiss.read_index(str(index_path))

        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.trends_map = data.get('trends_map', {})
            self.chunks_map = data.get('chunks_map', {})
            self.ids_list = data.get('ids_list', [])

        logger.info(f"Load {len(self.trends_map)} historical trends and {len(self.chunks_map)} chunks")


    async def save_to_disk(self):
        """Saves the index and data to disk"""

        if self.index is None:
            logger.warning("There is no data to save")
            return

        index_path = Path(self.config.vector_store_path) / "faiss.index"
        data_path = Path(self.config.vector_store_path) / "trends_data.pkl"

        faiss.write_index(self.index, str(index_path))

        with open(data_path, 'wb') as f:
            pickle.dump({
                'trends_map': self.trends_map,
                'chunks_map': self.chunks_map,
                'ids_list': self.ids_list
            }, f)

        logger.info(f"Saved {len(self.trends_map)} trends and {len(self.chunks_map)} chanks")


    async def add_trend(self, trend: HistoricalTrend) -> str:
        """Adds a trend to the vector storage (for backward compatibility)"""

        if self.index is None:
            await self.initialize()

        if trend.id in self.trends_map:
            logger.warning(f"The trend {trend.id} it already exists, we are updating it")
            await self.delete_trend(trend.id)

        embedding_array = np.array([trend.embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_array)

        self.index.add(embedding_array)
        self.trends_map[trend.id] = trend
        self.ids_list.append(trend.id)

        logger.debug(f"Add trend: {trend.theme}")

        await self.save_to_disk()
        return trend.id


    async def add_trends_batch(self, trends: List[HistoricalTrend]) -> List[str]:
        """Adds several trends with a batch"""

        if not trends:
            return []

        if self.index is None:
            await self.initialize()

        new_trends = [t for t in trends if t.id not in self.trends_map]

        if not new_trends:
            return [t.id for t in trends]

        embeddings = np.array([t.embedding for t in new_trends], dtype=np.float32)
        faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

        trend_ids = []
        for trend in new_trends:
            self.trends_map[trend.id] = trend
            self.ids_list.append(trend.id)
            trend_ids.append(trend.id)

        logger.info(f"Added {len(new_trends)} trends batcch")

        await self.save_to_disk()

        return trend_ids

    async def add_chunk(self, chunk: Dict[str, Any], trend_id: str) -> str:
        """Adds a chunk to the vector storage"""

        if self.index is None:
            await self.initialize()

        chunk_id = f"{trend_id}_chunk_{len([c for c in self.chunks_map.values() if c['trend_id'] == trend_id])}"

        from src.trend_analyzer.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        chunk_embedding = embedding_service.encode_texts([chunk["content"]])[0].tolist()

        chunk_data = {
            "id": chunk_id,
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "trend_id": trend_id,
            "embedding": chunk_embedding
        }

        embedding_array = np.array([chunk_embedding], dtype=np.float32)
        faiss.normalize_L2(embedding_array)
        self.index.add(embedding_array)

        self.chunks_map[chunk_id] = chunk_data
        self.ids_list.append(chunk_id)

        logger.debug(f"Added chank: {chunk['metadata'].get('chunk_type', 'unknown')} for trend {trend_id}")

        await self.save_to_disk()
        return chunk_id


    async def add_trend_with_chunks(self, trend: HistoricalTrend, chunks: List[Dict[str, Any]]) -> str:
        """Adds the trend along with its chunks"""

        trend_id = await self.add_trend(trend)

        for chunk in chunks:
            await self.add_chunk(chunk, trend_id)

        logger.info(f"Added trend {trend.theme} with {len(chunks)} chank")
        return trend_id


    async def find_similar(
            self,
            embedding: List[float],
            k: int = 5,
            similarity_threshold: float = 0.7
    ) -> List[VectorSearchResult]:
        """Finds k similar trends (for backward compatibility)"""

        return await self._find_similar_impl(embedding, k, similarity_threshold, search_chunks=False)


    async def find_similar_chunks(
            self,
            embedding: List[float],
            k: int = 5,
            similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Finds k similar chunks"""

        results = await self._find_similar_impl(embedding, k, similarity_threshold, search_chunks=True)

        chunk_results = []
        for result in results:
            chunk_data = self.chunks_map.get(result.trend.id)
            if chunk_data:
                chunk_results.append({
                    "chunk": chunk_data,
                    "similarity_score": result.similarity_score,
                    "rank": result.rank
                })

        return chunk_results


    async def _find_similar_impl(
            self,
            embedding: List[float],
            k: int = 5,
            similarity_threshold: float = 0.7,
            search_chunks: bool = False
    ) -> List[VectorSearchResult]:
        """The basic implementation of the search for similar"""

        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(query_embedding)

        search_k = min(k * 3, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, search_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.ids_list) and score >= similarity_threshold:
                item_id = self.ids_list[idx]

                if search_chunks:
                    chunk_data = self.chunks_map.get(item_id)
                    if chunk_data:
                        temp_trend = HistoricalTrend(
                            id=chunk_data["id"],
                            theme=chunk_data["metadata"].get("chunk_type", "chunk"),
                            posts=[],
                            confidence=score,
                            timestamp=datetime.now(),
                            embedding=chunk_data["embedding"],
                            metadata=chunk_data["metadata"]
                        )
                        results.append(VectorSearchResult(
                            trend=temp_trend,
                            similarity_score=float(score),
                            rank=len(results) + 1
                        ))
                else:
                    trend = self.trends_map.get(item_id)
                    if trend:
                        results.append(VectorSearchResult(
                            trend=trend,
                            similarity_score=float(score),
                            rank=len(results) + 1
                        ))

                if len(results) >= k:
                    break

        logger.debug(
            f"Found {len(results)} similar {'chanks' if search_chunks else 'trends'} (score ≥ {similarity_threshold})")
        return results


    async def delete_trend(self, trend_id: str) -> bool:
        """Deletes the trend and all its chunks"""

        if trend_id not in self.trends_map:
            return False

        chunks_to_delete = [chunk_id for chunk_id, chunk_data in self.chunks_map.items()
                            if chunk_data["trend_id"] == trend_id]

        for chunk_id in chunks_to_delete:
            del self.chunks_map[chunk_id]
            if chunk_id in self.ids_list:
                self.ids_list.remove(chunk_id)

        deleted_trend = self.trends_map.pop(trend_id)
        if trend_id in self.ids_list:
            self.ids_list.remove(trend_id)

        logger.debug(f"Trend removed: {deleted_trend.theme} and {len(chunks_to_delete)} chamks")

        await self._rebuild_index()
        return True


    async def _rebuild_index(self):
        """Rebuilds the index without deleted elements"""

        if not self.trends_map and not self.chunks_map:
            await self._create_new_index()
            return

        new_index = faiss.IndexFlatIP(self.dimension)

        embeddings = []
        valid_ids = []

        for trend_id, trend in self.trends_map.items():
            embeddings.append(trend.embedding)
            valid_ids.append(trend_id)

        for chunk_id, chunk_data in self.chunks_map.items():
            embeddings.append(chunk_data["embedding"])
            valid_ids.append(chunk_id)

        if embeddings:
            embedding_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embedding_array)
            new_index.add(embedding_array)

        self.index = new_index
        self.ids_list = valid_ids

        await self.save_to_disk()


    async def cleanup_old_trends(self, max_days: int = 30):
        """Deletes trends older than max_days days"""

        cutoff_date = datetime.now() - timedelta(days=max_days)
        deleted_count = 0

        for trend_id, trend in list(self.trends_map.items()):
            if trend.timestamp < cutoff_date:
                await self.delete_trend(trend_id)
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Removed {deleted_count} outdated trends")


    async def get_trend_stats(self) -> dict:
        """Returns storage statistics"""

        if self.index is None:
            return {"total_trends": 0, "total_chunks": 0}

        return {
            "total_trends": len(self.trends_map),
            "total_chunks": len(self.chunks_map),
            "total_indexed": self.index.ntotal,
            "dimension": self.dimension,
            "storage_path": self.config.vector_store_path,
            "chunks_by_type": self._get_chunks_by_type_stats()
        }


    def _get_chunks_by_type_stats(self) -> dict:
        """Statistics of chunks by type"""

        type_counts = {}
        for chunk_data in self.chunks_map.values():
            chunk_type = chunk_data["metadata"].get("chunk_type", "unknown")
            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
        return type_counts


    def _get_trends_by_date_stats(self) -> dict:
        """Trend statistics by date"""

        date_counts = {}
        for trend in self.trends_map.values():
            date_str = trend.timestamp.strftime("%Y-%m-%d")
            date_counts[date_str] = date_counts.get(date_str, 0) + 1
        return date_counts
