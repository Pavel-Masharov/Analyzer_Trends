from datetime import datetime
from typing import List, Dict
from loguru import logger

from src.common.rag_models import HistoricalTrend, TrendEnrichment
from src.common.models import TrendCluster, SocialPost
from src.services.rag_data_loader import RAGDataLoader
from src.vector import create_vector_store
from src.trend_analyzer.embedding_service import EmbeddingService
from configs.rag_config import RAGConfig
from .chunking_service import ChunkingService


class RAGService:
    """A service for enriching trends through RAG"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.vector_store = create_vector_store(config)
        self.embedding_service = EmbeddingService(config.embedding_model)
        self.data_loader = RAGDataLoader(self, self.embedding_service)
        self.chunking_service = ChunkingService()
        self._initialized = False


    async def initialize(self):
        if self._initialized:
            return

        await self.vector_store.initialize()
        self._initialized = True
        logger.success("✅ RAG service initialized")


    async def load_external_data(self, data_path: str) -> Dict[str, int]:
        """Loads external data into the RAG system"""
        if not self._initialized:
            await self.initialize()

        return await self.data_loader.load_from_directory(data_path)


    async def add_custom_trend(self, theme: str, confidence: float = 0.8,
                               category: str = "general") -> str:
        """Adds a custom trend to RAG"""

        return await self.data_loader.create_sample_external_trend(
            theme, confidence, category
        )


    async def enrich_trend(self, trend: TrendCluster) -> TrendEnrichment:
        """Enriches the trend with historical context"""

        if not self._initialized:
            await self.initialize()

        logger.debug(f"We are starting to enrich the trend: {trend.theme}")

        try:
            trend_embedding = await self._create_trend_embedding(trend)

            similar_chunks = await self.vector_store.find_similar_chunks(
                embedding=trend_embedding,
                k=self.config.max_similar_trends,
                similarity_threshold=self.config.similarity_threshold
            )

            similar_trends = self._group_chunks_by_trend(similar_chunks)

            # if similar_trends:
            #     first_trend = similar_trends[0]
            #     # logger.debug(f"✅ Первый grouped trend: {type(first_trend)}, theme: {first_trend.theme}")
            #     # logger.debug(f"✅ Атрибуты: {[attr for attr in dir(first_trend) if not attr.startswith('_')]}")

            enrichment_data = await self._analyze_trend_context(trend, similar_trends)

            if self.config.save_historical_trends:
                await self._store_current_trend_as_chunks(trend, trend_embedding)

            result = TrendEnrichment(
                current_trend=trend,
                similar_historical_trends=similar_trends,
                enrichment_data=enrichment_data
            )
            logger.debug(f"Successfully created TrendEnrichment, similar_trends: {len(similar_trends)}")
            return result

        except Exception as e:
            logger.error(f"Error в enrich_trend для '{trend.theme}': {e}")

            return TrendEnrichment(
                current_trend=trend,
                similar_historical_trends=[],
                enrichment_data={"context": "Ошибка обогащения", "velocity": 1.0}
            )


    async def _create_trend_embedding(self, trend: TrendCluster) -> List[float]:
        """Creates a vector representation of the trend from chanks"""

        trend_for_chunking = HistoricalTrend(
            id=f"current_{hash(trend.theme)}",
            theme=trend.theme,
            posts=trend.posts,
            confidence=trend.confidence,
            timestamp=datetime.now(),
            embedding=[],
            metadata={
                "post_count": len(trend.posts),
                "platforms": list(set(post.platform.value for post in trend.posts)),
                "avg_engagement": sum(post.get_engagement_score() for post in trend.posts) / len(
                    trend.posts) if trend.posts else 0
            }
        )

        chunks = self.chunking_service.chunk_trend(trend_for_chunking)

        chunk_texts = [chunk["content"] for chunk in chunks]
        if chunk_texts:
            embeddings = self.embedding_service.encode_texts(chunk_texts)
            trend_embedding = embeddings.mean(axis=0).tolist()
        else:
            trend_embedding = self.embedding_service.encode_texts([trend.theme])[0].tolist()

        return trend_embedding


    async def _analyze_trend_context(
            self,
            trend: TrendCluster,
            similar_trends: List[HistoricalTrend]
    ) -> Dict:
        """Analyzes the context of a trend based on similar historical trends"""

        if not similar_trends:
            return {
                "context": "Новый тренд без исторических аналогов",
                "velocity": 1.0,
                "engagement_comparison": {"status": "new", "comparison": "Нет данных для сравнения"},
                "similar_trends_count": 0,
                "avg_similarity": 0.0
            }

        if similar_trends:
            avg_similarity = sum(trend.metadata.get("avg_similarity", 0) for trend in similar_trends) / len(
                similar_trends)
        else:
            avg_similarity = 0.0

        velocity = await self._calculate_trend_velocity(trend, similar_trends)

        engagement_stats = await self._analyze_engagement_comparison(trend, similar_trends)

        context = await self._generate_context_description(trend, similar_trends, velocity)

        return {
            "velocity": velocity,
            "engagement_comparison": engagement_stats,
            "context": context,
            "similar_trends_count": len(similar_trends),
            "avg_similarity": avg_similarity
        }


    async def _calculate_trend_velocity(self, current_trend: TrendCluster,
                                        similar_trends: List[HistoricalTrend]) -> float:
        """Calculates the growth rate of a trend relative to historical analogs"""

        if not similar_trends:
            return 1.0

        current_engagement = sum(post.get_engagement_score() for post in current_trend.posts)

        historical_engagements = []
        for historical_trend in similar_trends:
            historical_engagement = sum(post.get_engagement_score() for post in historical_trend.posts)
            historical_engagements.append(historical_engagement)

        avg_historical_engagement = sum(historical_engagements) / len(historical_engagements)

        if avg_historical_engagement > 0:
            return current_engagement / avg_historical_engagement
        else:
            return 2.0

    async def _analyze_engagement_comparison(self, current_trend: TrendCluster,
                                             similar_trends: List[HistoricalTrend]) -> Dict:
        """Compares engagement with historical trends"""

        current_score = current_trend.confidence * 100

        if not similar_trends:
            return {"status": "new", "comparison": "Нет данных для сравнения"}

        historical_scores = [trend.confidence * 100 for trend in similar_trends]
        avg_historical = sum(historical_scores) / len(historical_scores)

        difference = current_score - avg_historical

        if difference > 20:
            status = "growing_fast"
            comparison = f"На {difference:.1f}% выше исторических аналогов"
        elif difference > 0:
            status = "growing"
            comparison = f"На {difference:.1f}% выше среднего"
        else:
            status = "stable"
            comparison = "На уровне исторических трендов"

        return {
            "status": status,
            "comparison": comparison,
            "current_score": current_score,
            "historical_avg": avg_historical,
            "difference": difference
        }

    async def _generate_context_description(
            self,
            current_trend: TrendCluster,
            similar_trends: List[HistoricalTrend],
            velocity: float
    ) -> str:
        """Generates a text description of the trend context"""

        if not similar_trends:
            return "Новый тренд без исторических аналогов"

        similar_themes = [trend.theme for trend in similar_trends[:2]]
        themes_str = ", ".join(similar_themes)

        if velocity > 1.5:
            growth = "быстро растущий"
        elif velocity > 1.0:
            growth = "растущий"
        else:
            growth = "стабильный"

        return f"📈 {growth} тренд. Похож на: {themes_str}. Скорость роста: {velocity:.1f}x"

    async def _store_current_trend(self, trend: TrendCluster, embedding: List[float]):
        """Saves the current trend in the history"""

        historical_trend = HistoricalTrend(
            id=f"trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(trend.theme)}",
            theme=trend.theme,
            posts=trend.posts,
            confidence=trend.confidence,
            timestamp=datetime.now(),
            embedding=embedding,
            metadata={
                "post_count": len(trend.posts),
                "platforms": list(set(post.platform.value for post in trend.posts)),
                "avg_engagement": sum(post.get_engagement_score() for post in trend.posts) / len(trend.posts)
            }
        )

        await self.vector_store.add_trend(historical_trend)
        logger.debug(f"The historical trend has been preserved: {trend.theme}")


    async def cleanup_old_data(self):
        """Clears outdated data"""
        if self.config.cleanup_old_trends:
            await self.vector_store.cleanup_old_trends(self.config.max_history_days)


    async def get_stats(self) -> Dict:
        """Returns RAG service statistics"""

        if not self._initialized:
            await self.initialize()

        vector_stats = await self.vector_store.get_trend_stats()

        return {
            "rag_service": {
                "initialized": self._initialized,
                "config": self.config.dict()
            },
            "vector_store": vector_stats
        }


    def _group_chunks_by_trend(self, similar_chunks: List) -> List[HistoricalTrend]:
        """Groups the found chunks by source trends"""

        from collections import defaultdict, Counter

        trends_dict = defaultdict(list)

        for chunk_result in similar_chunks:
            chunk_data = chunk_result["chunk"]
            trend_id = chunk_data.get("trend_id")

            if trend_id:
                trends_dict[trend_id].append({
                    "chunk_data": chunk_data,
                    "similarity_score": chunk_result["similarity_score"],
                    "rank": chunk_result["rank"]
                })

        aggregated_trends = []

        for trend_id, chunks in trends_dict.items():
            if chunks:
                main_chunk_data = chunks[0]["chunk_data"]
                avg_similarity = sum(c["similarity_score"] for c in chunks) / len(chunks)
                max_similarity = max(c["similarity_score"] for c in chunks)

                original_trend = None
                if hasattr(self.vector_store, 'trends_map'):
                    original_trend = self.vector_store.trends_map.get(trend_id)

                if original_trend:
                    aggregated_trend = HistoricalTrend(
                        id=trend_id,
                        theme=original_trend.theme,
                        posts=original_trend.posts,
                        confidence=original_trend.confidence,
                        timestamp=original_trend.timestamp,
                        embedding=original_trend.embedding,
                        metadata={
                            "chunks_count": len(chunks),
                            "avg_similarity": avg_similarity,
                            "max_similarity": max_similarity,
                            "chunk_types": list(
                                set(c["chunk_data"].get("metadata", {}).get("chunk_type", "unknown") for c in chunks)),
                            "source": "chunk_aggregation"
                        }
                    )
                else:
                    theme_candidates = []

                    for chunk in chunks:
                        chunk_metadata = chunk["chunk_data"].get("metadata", {})

                        possible_theme_sources = [
                            chunk_metadata.get("theme"),
                            chunk_metadata.get("chunk_theme"),
                            self._extract_theme_from_content(chunk["chunk_data"].get("content", ""))
                        ]

                        for theme in possible_theme_sources:
                            if theme and theme not in ["Unknown Trend", "unknown", "chunk"]:
                                theme_candidates.append(theme)

                    final_theme = "Unknown Trend"
                    if theme_candidates:
                        theme_counter = Counter(theme_candidates)
                        final_theme = theme_counter.most_common(1)[0][0]
                    else:
                        main_theme = main_chunk_data.get("metadata", {}).get("theme")
                        if main_theme and main_theme != "Unknown Trend":
                            final_theme = main_theme
                        else:
                            chunk_types = list(
                                set(c["chunk_data"].get("metadata", {}).get("chunk_type", "unknown") for c in chunks))
                            if "theme" in chunk_types:
                                final_theme = "Aggregated Trend"
                            else:
                                final_theme = f"Trend from {len(chunks)} chunks"

                    aggregated_trend = HistoricalTrend(
                        id=trend_id,
                        theme=final_theme,
                        posts=[],
                        confidence=avg_similarity,
                        timestamp=datetime.now(),
                        embedding=[],
                        metadata={
                            "chunks_count": len(chunks),
                            "avg_similarity": avg_similarity,
                            "max_similarity": max_similarity,
                            "chunk_types": list(
                                set(c["chunk_data"].get("metadata", {}).get("chunk_type", "unknown") for c in chunks)),
                            "theme_candidates": theme_candidates,
                            "source": "chunk_reconstruction"
                        }
                    )

                aggregated_trends.append(aggregated_trend)

        aggregated_trends.sort(key=lambda x: x.metadata.get("max_similarity", 0), reverse=True)
        return aggregated_trends

    def _extract_theme_from_content(self, content: str) -> str:
        """Extracts a topic from the content of a chunk"""

        if not content:
            return ""

        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if len(line) < 10 or len(line) > 100:
                continue

            if line.startswith(('Тема тренда:', 'Тема:', 'Пост #', 'Аналитика тренда')):
                continue

            if any(keyword in line.lower() for keyword in
                   ['python', 'ai', 'искусственный', 'разработка', 'технологи', 'данн']):
                return line[:50] + "..." if len(line) > 50 else line

        return ""




    async def _store_current_trend_as_chunks(self, trend: TrendCluster, embedding: List[float]):
        """Preserves the current trend in history as a set of chunks"""

        historical_trend = HistoricalTrend(
            id=f"trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(trend.theme)}",
            theme=trend.theme,
            posts=trend.posts,
            confidence=trend.confidence,
            timestamp=datetime.now(),
            embedding=embedding,
            metadata={
                "post_count": len(trend.posts),
                "platforms": list(set(post.platform.value for post in trend.posts)),
                "avg_engagement": sum(post.get_engagement_score() for post in trend.posts) / len(
                    trend.posts) if trend.posts else 0
            }
        )

        chunks = self.chunking_service.chunk_trend(historical_trend)
        for chunk in chunks:
            await self.vector_store.add_chunk(chunk, historical_trend.id)

        logger.debug(f"Saved the trend as {len(chunks)} chunks: {trend.theme}")