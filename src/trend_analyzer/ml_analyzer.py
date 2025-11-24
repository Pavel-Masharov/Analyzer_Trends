from typing import List, Dict
import pandas as pd
import numpy as np
from loguru import logger

from .base_analyzer import BaseTrendAnalyzer
from .embedding_service import EmbeddingService
from .clustering import TextClustering
from .theme_extractor import ThemeExtractor
from src.common.models import TrendCluster, SocialPost


class MLAnalyzer(BaseTrendAnalyzer):

    def __init__(
            self,
            min_cluster_size: int = 15,
            rag_manager=None,
            use_external_knowledge: bool = True
    ):

        logger.info("Initializing the ML analyzer with RAG and external knowledge")

        self.embedding_service = EmbeddingService()
        self.clustering = TextClustering(min_cluster_size=min_cluster_size)
        self.theme_extractor = ThemeExtractor()

        # RAG
        self.rag_manager = rag_manager
        self.use_external_knowledge = use_external_knowledge

        if self.rag_manager:
            logger.info("RAG integration is activated")
            if self.use_external_knowledge:
                logger.info("Using external knowledge: ON")
            else:
                logger.info("Using external knowledge: OFF")
        else:
            logger.info("RAG integration disabled")

        # To track the quality
        self.last_clustering_quality = 0.0
        self.last_analysis_stats = {}

        logger.success(f"The ML analyzer is ready. RAG: {bool(rag_manager)}, External: {use_external_knowledge}")


    async def analyze_trends(self, df: pd.DataFrame) -> List[TrendCluster]:
        """Анализирует тренды используя ML + RAG + внешние знания"""

        logger.info(f"🔍 Running a full analysis for {len(df)} posts")

        if df.empty:
            return []

        # preparation
        texts = df['text'].tolist()
        embeddings = self.embedding_service.encode_texts(texts)

        if len(embeddings) == 0:
            return []

        # Clusterization
        cluster_labels, n_clusters = self.clustering.cluster_texts(texts, embeddings)
        self.last_clustering_quality = self.clustering.last_quality_score

        logger.info(f"📊 Качество кластеризации: {self.last_clustering_quality:.3f}")

        # create base trends
        trends = await self._clusters_to_trends(df, cluster_labels, n_clusters)

        # Add RAG and external knowledge
        if self.rag_manager and self.use_external_knowledge:
            trends = await self._enhance_trends_complete(trends)
            logger.success(f"ML+RAG+External The analysis is completed! Found {len(trends)} trends")
        elif self.rag_manager:
            trends = await self.rag_manager.enrich_trends(trends)
            logger.success(f"ML+RAG The analysis is completed! Found {len(trends)} trends")
        else:
            logger.success(f"ML The analysis is completed! Found {len(trends)} trends")

        # save statistyc
        self.last_analysis_stats = self._calculate_analysis_stats(trends)

        return trends


    async def _enhance_trends_complete(self, trends: List[TrendCluster]) -> List[TrendCluster]:
        """Full enrichment of trends through RAG and external knowledge"""

        logger.info("🧠 Enrichment RAG and external knowledge")

        # 1. Enrichment RAG
        rag_enriched = await self.rag_manager.enrich_trends(trends)

        # 2. Additional analysis with an external context
        fully_enriched = []
        for trend in rag_enriched:
            enhanced_trend = await self._enhance_with_external_context(trend)
            fully_enriched.append(enhanced_trend)

        return fully_enriched


    async def _enhance_with_external_context(self, trend: TrendCluster) -> TrendCluster:
        """Enriches the trend with external context and analytics"""

        # Getting historical context from RAG
        historical_trends = trend.metadata.get("similar_historical_trends", [])

        # Analyzing with external knowledge
        external_analysis = {
            "historical_context_quality": self._analyze_historical_context_quality(historical_trends),
            "trend_novelty": self._calculate_trend_novelty(trend, historical_trends),
            "confidence_boost": self._calculate_confidence_boost(trend, historical_trends)
        }

        # We update our confidence based on external analysis
        original_confidence = trend.confidence
        boosted_confidence = self._apply_confidence_boost(original_confidence, external_analysis)

        # Updating metadata
        trend.metadata.update({
            "external_analysis": external_analysis,
            "original_confidence": original_confidence,
            "boosted_confidence": boosted_confidence,
            "enhancement_level": "complete"
        })

        # Updating basic confidence
        trend.confidence = boosted_confidence

        return trend


    def _analyze_historical_context_quality(self, historical_trends: List) -> float:
        """Analyzes the quality of the historical context"""

        if not historical_trends:
            return 0.0

        # We evaluate based on the number and confidence of historical trends
        count_score = min(len(historical_trends) / 5.0, 1.0)  # max 5 trends
        confidence_score = sum(t.get('confidence', 0.5) for t in historical_trends) / len(historical_trends)

        return (count_score + confidence_score) / 2.0

    def _calculate_trend_novelty(self, current_trend: TrendCluster, historical_trends: List) -> float:
        """Calculates the novelty of the trend"""

        if not historical_trends:
            return 1.0  # A completely new trend

        # The more historical analogues there are, the less novelty there is
        novelty = 1.0 - (len(historical_trends) / 10.0)  # max 10 trends
        return max(0.1, novelty)  # minimum 10% novelty


    def _calculate_confidence_boost(self, trend: TrendCluster, historical_trends: List) -> float:
        """Calculates a confidence boost based on external analysis"""

        base_confidence = trend.confidence

        if not historical_trends:
            # A new trend is a moderate boost for novelty
            return min(base_confidence * 1.1, 1.0)

        # A trend with history - a boost for confirmation
        avg_historical_confidence = sum(t.get('confidence', 0.5) for t in historical_trends) / len(historical_trends)

        if avg_historical_confidence > base_confidence:
            # Historical trends are more confident - a positive boost
            boost = (avg_historical_confidence - base_confidence) * 0.5
            return min(base_confidence + boost, 1.0)
        else:
            # Less confidence in history - a small boost
            return min(base_confidence * 1.05, 1.0)


    def _apply_confidence_boost(self, original_confidence: float, external_analysis: Dict) -> float:
        """Applies a confidence boost based on external analysis"""

        novelty = external_analysis.get("trend_novelty", 0.5)
        context_quality = external_analysis.get("historical_context_quality", 0.0)
        validation = external_analysis.get("external_validation", {}).get("cross_reference_score", 0.5)

        boost_factor = (novelty * 0.3 + context_quality * 0.4 + validation * 0.3)

        boosted = original_confidence * (1.0 + boost_factor * 0.2)  # Максимум +20%
        return min(boosted, 1.0)


    def _calculate_analysis_stats(self, trends: List[TrendCluster]) -> Dict:
        """Calculates analysis statistics"""

        if not trends:
            return {}

        rag_enriched = [t for t in trends if t.metadata.get("rag_enriched")]
        fully_enhanced = [t for t in trends if t.metadata.get("enhancement_level") == "complete"]

        confidences = [t.confidence for t in trends]
        cluster_sizes = [len(t.posts) for t in trends]

        return {
            "total_trends": len(trends),
            "rag_enriched": len(rag_enriched),
            "fully_enhanced": len(fully_enhanced),
            "avg_confidence": sum(confidences) / len(confidences),
            "avg_cluster_size": sum(cluster_sizes) / len(cluster_sizes),
            "clustering_quality": self.last_clustering_quality
        }


    def get_analysis_report(self) -> Dict:
        """Returns the latest analysis report"""

        return {
            "clustering_quality": self.last_clustering_quality,
            "analysis_stats": self.last_analysis_stats,
            "rag_enabled": bool(self.rag_manager),
            "external_knowledge_enabled": self.use_external_knowledge
        }


    async def _clusters_to_trends(self, df: pd.DataFrame, cluster_labels: np.ndarray, n_clusters: int) -> List[
            TrendCluster]:
            """Converts clusters into trends with smart themes"""

            trends = []

            texts = df['text'].tolist()
            embeddings = self.embedding_service.encode_texts(texts)

            for cluster_id in range(n_clusters):
                cluster_mask = (cluster_labels == cluster_id)
                cluster_df = df[cluster_mask]

                if cluster_id == -1 or len(cluster_df) < self.clustering.min_cluster_size:
                    continue

                cluster_texts = cluster_df['text'].tolist()
                cluster_embeddings = embeddings[cluster_mask]

                theme = self.theme_extractor.extract_theme(cluster_texts, cluster_embeddings)

                posts = []
                for _, row in cluster_df.iterrows():
                    post = SocialPost(
                        id=row['id'],
                        platform=row['platform'],
                        text=row['text'],
                        author=row['author'],
                        timestamp=row['timestamp'],
                        engagement={'views': row.get('views', 0), 'likes': row.get('likes', 0),
                                    'shares': row.get('shares', 0)}
                    )
                    posts.append(post)

                confidence = self._calculate_confidence(cluster_df, cluster_embeddings)

                trend = TrendCluster(
                    theme=theme,
                    posts=posts,
                    confidence=confidence
                )
                trends.append(trend)

            trends.sort(key=lambda x: x.confidence, reverse=True)
            return trends[:20]  # Максимум 20 трендов

    def _calculate_confidence(self, cluster_df: pd.DataFrame, embeddings: np.ndarray) -> float:
        """Calculates trend confidence based on several factors"""

        if len(cluster_df) < 2:
            return 0.1

        size_factor = min(1.0, len(cluster_df) / 10)

        if len(embeddings) > 1:
            centroid = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            coherence = 1.0 - (np.std(distances) / np.mean(distances) if np.mean(distances) > 0 else 0)
            coherence_factor = max(0.1, coherence)
        else:
            coherence_factor = 0.5

        confidence = (size_factor * 0.6 + coherence_factor * 0.4)
        return min(1.0, confidence)


