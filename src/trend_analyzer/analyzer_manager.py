from typing import List
import pandas as pd

from .base_analyzer import BaseTrendAnalyzer
from .ml_analyzer import MLAnalyzer
from src.common.models import TrendCluster


class AnalyzerManager:
    """Run analyze trends with RAG"""

    def __init__(self, rag_config=None):
        self.rag_manager = None
        if rag_config:
            from src.services.rag_manager import RAGManager
            self.rag_manager = RAGManager(rag_config)

        self.analyzer: BaseTrendAnalyzer = MLAnalyzer(rag_manager=self.rag_manager)

        print(f" Manager analyze trends initialyze. RAG: {bool(self.rag_manager)}")


    async def initialize(self):
        """Инициализация с RAG"""

        if self.rag_manager:
            await self.rag_manager.initialize()


    async def find_trends(self, df: pd.DataFrame) -> List[TrendCluster]:
        """Находит тренды в данных"""

        if df.empty:
            print("Not data")
            return []

        if self.rag_manager:
            await self.initialize()

        trends = await self.analyzer.analyze_trends(df)
        return trends


    def print_trends_report(self, trends: List[TrendCluster]):
        """Print report with RAG info"""

        if not trends:
            print("❌ Not trends")
            return

        print(f"\n Fihd trends: {len(trends)}")
        print("=" * 60)

        for i, trend in enumerate(trends, 1):
            print(f"\n📈 Trend #{i}: {trend.theme}")
            print(f"   Confidence: {trend.confidence:.1%}")
            print(f"   Count posts: {len(trend.posts)}")

            if trend.metadata.get("rag_enriched"):
                print(f"   📊 RAG analytics:")
                print(f"      context: {trend.metadata.get('historical_context', 'N/A')}")
                print(f"      speed growth: {trend.metadata.get('trend_velocity', 1.0):.1f}x")
                print(
                    f"      Similar historical trends: {len(trend.metadata.get('similar_historical_trends', []))}")

            print("   Examples posts:")
            for post in trend.posts[:2]:
                print(f"     • {post.text}")
