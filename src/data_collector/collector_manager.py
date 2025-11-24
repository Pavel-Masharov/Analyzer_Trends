from typing import Dict, List
import asyncio
import pandas as pd

from .base_collector import BaseDataCollector
from .telegram_collector import TelegramCollector
from src.common.models import SocialPost
from configs.base_config import AppConfig, SocialPlatform
from .vk_collector import VKCollector


class CollectorManager:
    """Run all clollectors"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.collectors: Dict[str, BaseDataCollector] = {}
        self._setup_collectors()


    def _setup_collectors(self):
        """Create collectors"""

        for source_config in self.config.data_sources:
            if not source_config.enabled:
                continue

            if source_config.platform == SocialPlatform.TELEGRAM:
                self.collectors["telegram"] = TelegramCollector(source_config)
            elif source_config.platform == SocialPlatform.VK:
                self.collectors["vk"] = VKCollector(source_config)


    async def collect_all_data(self, hours: int = 100) -> pd.DataFrame:
        """Collects data all platforms and return DataFrame"""
        all_posts: List[SocialPost] = []

        tasks = []
        for name, collector in self.collectors.items():
            task = collector.collect_recent_posts(hours)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for posts in results:
            all_posts.extend(posts)

        return self._posts_to_dataframe(all_posts)


    def _posts_to_dataframe(self, posts: List[SocialPost]) -> pd.DataFrame:
        data = []
        for post in posts:
            data.append({
                'id': post.id,
                'platform': post.platform.value,
                'text': post.text,
                'author': post.author,
                'timestamp': post.timestamp,
                'engagement_score': post.get_engagement_score(),
                'likes': post.engagement.get('likes', 0),
                'views': post.engagement.get('views', 0),
                'comments': post.engagement.get('comments', 0),
                'reposts': post.engagement.get('reposts', 0)
            })
        df = pd.DataFrame(data)
        return df


    async def health_check_all(self) -> Dict[str, bool]:
        """Checks the functionality of all collectors"""
        results = {}

        for name, collector in self.collectors.items():
            try:
                is_healthy = await collector.health_check()
                results[name] = is_healthy
                status = "✅" if is_healthy else "❌"
                print(f"   {name}: {status}")
            except Exception as e:
                results[name] = False
                print(f"   {name}: ❌ ({e})")

        return results
