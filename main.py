import asyncio
import pandas as pd

from configs.config import app_config
from src.data_collector.collector_manager import CollectorManager
from src.trend_analyzer.analyzer_manager import AnalyzerManager
from src.trend_analyzer.ml_analyzer import MLAnalyzer
from src.services.rag_manager import RAGManager


COLLECTION_HOURS = 240
MIN_CLUSTER_SIZE = 7


async def get_data() -> pd.DataFrame:
    """Getting data"""

    print("📡 Собираем реальные данные...")
    collector_manager = CollectorManager(app_config)
    df = await collector_manager.collect_all_data(COLLECTION_HOURS)

    if not df.empty:
        print(f"✅ Собрано {len(df)} реальных постов")
        return df
    else:
        print("⚠️ Не удалось собрать реальные данные")
        return None


async def run_data_analysis():
    """Data analysis"""

    print("🚀 АНАЛИЗ РЕАЛЬНЫХ ДАННЫХ + RAG")
    print("=" * 60)
    print(f"⚙️  НАСТРОЙКИ:")
    print(f"   COLLECTION_HOURS: {COLLECTION_HOURS}")
    print(f"   MIN_CLUSTER_SIZE: {MIN_CLUSTER_SIZE}")
    print(f"\n1. 📡 ПОЛУЧЕНИЕ ДАННЫХ...")

    df = await get_data()

    if df.empty:
        print("❌ Нет данных для анализа")
        return

    print(f"\n📊 СТАТИСТИКА ДАННЫХ:")
    print(f"   Всего постов: {len(df)}")
    print(f"   Платформы: {df['platform'].value_counts().to_dict()}")
    print(f"   Источники: {len(df['author'].unique())}")
    print(
        f"   Диапазон дат: {df['timestamp'].min().strftime('%Y-%m-%d %H:%M')} - {df['timestamp'].max().strftime('%Y-%m-%d %H:%M')}")
    print(
        f"   Engagement: {df['engagement_score'].min():.1f} - {df['engagement_score'].max():.1f} (avg: {df['engagement_score'].mean():.1f})")

    print(f"\n2. 🔍 АНАЛИЗ ТРЕНДОВ С RAG...")

    rag_manager = RAGManager(app_config.rag_config)
    custom_analyzer = MLAnalyzer(
        min_cluster_size=MIN_CLUSTER_SIZE,
        rag_manager=rag_manager,
        use_external_knowledge=True
    )

    analyzer_manager = AnalyzerManager(rag_config=app_config.rag_config)
    analyzer_manager.analyzer = custom_analyzer

    trends = await analyzer_manager.find_trends(df)

    print(f"✅ Найдено {len(trends)} трендов")
    print(f"\n3. 📈 ДЕТАЛЬНАЯ АНАЛИТИКА ТРЕНДОВ:")

    for i, trend in enumerate(trends, 1):
        print(f"\n{i}. 🎯 {trend.theme}")
        print(f"   📊 Уверенность: {trend.confidence:.1%}")
        print(f"   📈 Постов: {len(trend.posts)}")
        print(f"   🏷️  Платформы: {list(set(p.platform.value for p in trend.posts))}")
        print(f"   📍 Источники: {list(set(p.author for p in trend.posts[:3]))}")
        print(f"   💡 Общий Engagement: {sum(p.get_engagement_score() for p in trend.posts):.1f}")

        if trend.metadata.get("rag_enriched"):
            similar_trends = trend.metadata.get("similar_historical_trends", [])
            velocity = trend.metadata.get("trend_velocity", 1.0)
            context = trend.metadata.get("historical_context", "")

            print(f"   🎯 RAG АНАЛИТИКА:")
            print(f"      • Исторических аналогов: {len(similar_trends)}")
            print(f"      • Скорость роста: {velocity:.1f}x")
            print(f"      • Контекст: {context}")

            if similar_trends:
                print(f"      • Топ аналоги:")
                for similar in similar_trends[:2]:
                    confidence = similar.get('confidence', 0)
                    theme = similar.get('theme', 'Unknown')
                    print(f"        - {theme[:60]}... ({confidence:.1%})")

        print(f"   🔥 Топ посты:")
        top_posts = sorted(trend.posts, key=lambda x: x.get_engagement_score(), reverse=True)[:2]
        for j, post in enumerate(top_posts, 1):
            short_text = post.text[:80] + "..." if len(post.text) > 80 else post.text
            print(f"      {j}. [{post.platform.value}] {post.author}: {short_text}")
            print(f"         Engagement: {post.get_engagement_score():.1f}")

    print(f"\n4. 📊 СТАТИСТИКА СИСТЕМЫ:")
    print(f"   Всего постов: {len(df)}")
    print(f"   Найдено трендов: {len(trends)}")
    rag_enriched = sum(1 for t in trends if t.metadata.get("rag_enriched"))
    print(f"   RAG обогащено: {rag_enriched}/{len(trends)}")

    if hasattr(custom_analyzer, 'last_clustering_quality'):
        print(f"   Качество кластеризации: {custom_analyzer.last_clustering_quality:.3f}")

    if hasattr(custom_analyzer, 'last_analysis_stats'):
        stats = custom_analyzer.last_analysis_stats
        print(f"   Средняя уверенность: {stats.get('avg_confidence', 0):.1%}")
        print(f"   Средний размер кластера: {stats.get('avg_cluster_size', 0):.1f}")


if __name__ == "__main__":
    asyncio.run(run_data_analysis())
