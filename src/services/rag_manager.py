from typing import List
from loguru import logger

from .rag_service import RAGService
from src.common.models import TrendCluster
from configs.rag_config import RAGConfig
from ..common.rag_models import TrendEnrichment


class RAGManager:
    """A manager for working with the RAG service"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.rag_service = RAGService(config)
        self._enabled = True


    async def initialize(self):
        if self._enabled:
            await self.rag_service.initialize()
            logger.success("RAG manager initialized")
        else:
            logger.info("RAG manager disabled")


    async def enrich_trends(self, trends: List[TrendCluster]) -> List[TrendCluster]:
        """Enriches the list of trends via RAG"""

        if not self._enabled or not trends:
            return trends

        enriched_trends = []
        for trend in trends:
            try:
                logger.debug(f"Processing the trend: {trend.theme}")
                enriched = await self.rag_service.enrich_trend(trend)
                logger.debug(f"An enriched object of the type was obtained: {type(enriched)}")

                if hasattr(enriched, 'current_trend'):
                    logger.debug(f"current_trend: {type(enriched.current_trend)}")
                if hasattr(enriched, 'similar_historical_trends'):
                    logger.debug(
                        f"similar_historical_trends: {type(enriched.similar_historical_trends)}, count: {len(enriched.similar_historical_trends)}")
                    if enriched.similar_historical_trends:
                        first_trend = enriched.similar_historical_trends[0]
                        logger.debug(f"First historical_trend: {type(first_trend)}, attributes: {dir(first_trend)}")

                enriched_trend = self._apply_enrichment(enriched)
                enriched_trends.append(enriched_trend)

            except Exception as e:
                logger.error(f"Trend Enrichment Error '{trend.theme}': {e}")
                enriched_trends.append(trend)

        logger.success(f"Enriched {len(enriched_trends)} trends")
        return enriched_trends


    def _apply_enrichment(self, enriched: TrendEnrichment) -> TrendCluster:
        """Applies enrichment to the trend"""
        trend = enriched.current_trend

        trend.metadata.update({
            "rag_enriched": True,
            "similar_historical_trends": [
                {
                    "theme": historical_trend.theme,
                    "confidence": historical_trend.confidence,
                    "timestamp": historical_trend.timestamp.isoformat(),
                    "chunks_count": historical_trend.metadata.get("chunks_count", 0),
                    "avg_similarity": historical_trend.metadata.get("avg_similarity", 0),
                    "source": historical_trend.metadata.get("source", "unknown")
                }
                for historical_trend in enriched.similar_historical_trends[:3]
            ],
            "trend_velocity": enriched.enrichment_data.get("velocity", 1.0),
            "engagement_status": enriched.enrichment_data.get("engagement_comparison", {}).get("status", "unknown"),
            "historical_context": enriched.enrichment_data.get("context", ""),
            "similar_trends_count": len(enriched.similar_historical_trends),
            "avg_similarity_score": enriched.enrichment_data.get("avg_similarity", 0)
        })

        return trend


    def disable(self):
        self._enabled = False
        logger.info("RAG disabled")


    def enable(self):
        self._enabled = True
        logger.info("RAG enabled")


    async def cleanup(self):
        if self._enabled:
            await self.rag_service.cleanup_old_data()






