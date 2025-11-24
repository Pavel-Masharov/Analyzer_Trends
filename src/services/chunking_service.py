from typing import List, Dict, Any
from loguru import logger
from src.common.rag_models import HistoricalTrend


class ChunkingService:
    """A service for breaking historical trends into chunks"""

    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def chunk_trend(self, trend: HistoricalTrend) -> List[Dict[str, Any]]:
        """Breaks down the historical trend into semantic chunks"""

        chunks = []

        # Chunk 1: Main topic and metadata
        theme_chunk = self._create_theme_chunk(trend)
        chunks.append(theme_chunk)

        # Chunk 2: Key posts (if any)
        if trend.posts:
            posts_chunks = self._create_posts_chunks(trend)
            chunks.extend(posts_chunks)

        # Chunk 3: Analytics and Context
        analytics_chunk = self._create_analytics_chunk(trend)
        chunks.append(analytics_chunk)

        return chunks


    def _create_theme_chunk(self, trend: HistoricalTrend) -> Dict[str, Any]:
        """Creates a chunk with the main theme"""

        content = f"""
Тема тренда: {trend.theme}
Уверенность: {trend.confidence:.1%}
Дата обнаружения: {trend.timestamp.strftime('%Y-%m-%d')}
Платформы: {', '.join(trend.metadata.get('platforms', ['неизвестно']))}
Количество постов: {trend.metadata.get('post_count', 0)}
        """.strip()

        return {
            "content": content,
            "metadata": {
                "chunk_type": "theme",
                "trend_id": trend.id,
                "theme": trend.theme,
                "timestamp": trend.timestamp.isoformat()
            }
        }


    def _create_posts_chunks(self, trend: HistoricalTrend) -> List[Dict[str, Any]]:
        """Creates chunks with key posts"""

        chunks = []
        top_posts = sorted(trend.posts, key=lambda x: x.get_engagement_score(), reverse=True)[:5]

        for i, post in enumerate(top_posts):
            content = f"""
Пост #{i + 1} из тренда "{trend.theme}":
Автор: {post.author}
Платформа: {post.platform.value}
Текст: {post.text[:200]}{'...' if len(post.text) > 200 else ''}
Engagement: {post.get_engagement_score():.1f}
Лайки: {post.engagement.get('likes', 0)}, Комментарии: {post.engagement.get('comments', 0)}
            """.strip()

            chunks.append({
                "content": content,
                "metadata": {
                    "chunk_type": "post",
                    "trend_id": trend.id,
                    "post_id": post.id,
                    "engagement_score": post.get_engagement_score(),
                    "platform": post.platform.value
                }
            })

        return chunks


    def _create_analytics_chunk(self, trend: HistoricalTrend) -> Dict[str, Any]:
        """Creates a chunk with analytics"""

        avg_engagement = trend.metadata.get('avg_engagement', 0)
        platforms = trend.metadata.get('platforms', [])

        content = f"""
Аналитика тренда "{trend.theme}":
Средний engagement: {avg_engagement:.1f}
Охват платформ: {len(platforms)}
Историческая значимость: {'высокая' if trend.confidence > 0.8 else 'средняя' if trend.confidence > 0.6 else 'низкая'}
Контекст: {self._generate_trend_context(trend)}
        """.strip()

        return {
            "content": content,
            "metadata": {
                "chunk_type": "analytics",
                "trend_id": trend.id,
                "confidence": trend.confidence,
                "avg_engagement": avg_engagement
            }
        }


    def _generate_trend_context(self, trend: HistoricalTrend) -> str:
        """Generates a contextual description of the trend"""

        if trend.confidence > 0.85:
            return "Сильный тренд с высокой уверенностью"
        elif trend.confidence > 0.7:
            return "Умеренный тренд со стабильным ростом"
        else:
            return "Формирующийся тренд требует наблюдения"
