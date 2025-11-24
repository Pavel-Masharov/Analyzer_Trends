import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger

from .base_collector import BaseDataCollector
from src.common.models import SocialPost, SocialPlatform, ContentType
from configs.base_config import DataSourceConfig


class VKCollector(BaseDataCollector):
    """
    Data collector from VK
    USE Official VK API: https://dev.vk.com/ru/method
    """

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.base_url = "https://api.vk.com/method"
        self.session = None
        self._validate_vk_config()


    def _validate_vk_config(self):
        """Check config VK"""
        if not self.config.api_key:
            raise ValueError("VK API-key not show")

        if not self.config.sources:
            raise ValueError("Not find group for monitoring")


    async def _get_session(self) -> aiohttp.ClientSession:
        """Create aiohttp sessions"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session


    async def collect_recent_posts(self, hours: int = 24) -> List[SocialPost]:
        """Collect posts VK origins"""

        logger.info(f"Collect data from {len(self.config.sources)} VK origins")

        session = await self._get_session()
        all_posts = []

        for source in self.config.sources:
            try:
                domain = self._extract_domain(source)
                posts = await self._fetch_group_posts(session, domain, hours)
                all_posts.extend(posts)
                logger.info(f"{source}: collect {len(posts)} posts")

                # Задержка для соблюдения лимитов API
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.error(f"❌ Error collect from {source}: {e}")

        logger.success(f" Total collected {len(all_posts)} posts from VK")
        return all_posts


    def _extract_domain(self, source: str) -> str:
        """Extracts domain from URL or ID"""

        if source.startswith('https://vk.com/'):
            return source.split('/')[-1]
        return source


    async def _fetch_group_posts(self, session: aiohttp.ClientSession, domain: str, hours: int) -> List[SocialPost]:
        """Get posts from VK origins"""

        url = f"{self.base_url}/wall.get"

        params = {
            'domain': domain,
            'count': 100,
            'access_token': self.config.api_key,
            'v': '5.131',
            'extended': 1
        }

        try:
            async with session.get(url, params=params) as response:
                data = await response.json()

                if 'error' in data:
                    logger.error(f"VK API Error: {data['error']}")
                    return []

                return await self._parse_vk_posts(data, domain, hours)

        except Exception as e:
            logger.error(f"Request error VK API: {e}")
            return []


    async def _parse_vk_posts(self, data: Dict, domain: str, hours: int) -> List[SocialPost]:
        """Parses VK API response into SocialPost objects"""

        posts = []
        cutoff_time = datetime.now() - timedelta(hours=hours)

        if 'response' not in data:
            return posts

        items = data['response'].get('items', [])
        groups = data['response'].get('groups', [])

        group_name = domain
        if groups and isinstance(groups, list) and len(groups) > 0:
            group_name = groups[0].get('name', domain)

        for item in items:
            if item.get('is_pinned') or item.get('marked_as_ads'):
                continue

            post_time = datetime.fromtimestamp(item['date'])
            if post_time < cutoff_time:
                continue

            engagement = {
                'likes': item.get('likes', {}).get('count', 0),
                'comments': item.get('comments', {}).get('count', 0),
                'reposts': item.get('reposts', {}).get('count', 0),
                'views': item.get('views', {}).get('count', 0)
            }

            content_type = ContentType.TEXT
            attachments = item.get('attachments', [])
            if attachments:
                if any(att.get('type') == 'photo' for att in attachments):
                    content_type = ContentType.IMAGE
                elif any(att.get('type') == 'video' for att in attachments):
                    content_type = ContentType.VIDEO
                elif any(att.get('type') == 'link' for att in attachments):
                    content_type = ContentType.LINK

            post_text = item.get('text', '').strip()
            if not post_text:
                if content_type == ContentType.IMAGE:
                    post_text = f"Изображение от {group_name}"
                elif content_type == ContentType.VIDEO:
                    post_text = f"Видео от {group_name}"
                elif content_type == ContentType.LINK:
                    post_text = f"Ссылка от {group_name}"
                else:
                    post_text = f"Пост от {group_name}"

            owner_id = item.get('owner_id', 0)
            post_id = item.get('id', 0)
            post_url = f"https://vk.com/wall{owner_id}_{post_id}"

            post = SocialPost(
                id=f"vk_{owner_id}_{post_id}",
                platform=SocialPlatform.VK,
                text=post_text[:2000],
                content_type=content_type,
                author=group_name,
                timestamp=post_time,
                engagement=engagement,
                url=post_url,
                metadata={
                    'source_domain': domain,
                    'has_attachments': bool(attachments),
                    'post_type': 'wall_post',
                    'owner_id': owner_id,
                    'engagement_raw': engagement
                }
            )
            posts.append(post)

        return posts


    async def health_check(self) -> bool:
        """Check availability VK API"""
        try:
            session = await self._get_session()
            url = f"{self.base_url}/groups.getById"

            params = {
                'group_ids': 'habr',
                'access_token': self.config.api_key,
                'v': '5.131'
            }

            async with session.get(url, params=params) as response:
                data = await response.json()
                return 'response' in data

        except Exception as e:
            logger.error(f"VK API health check failed: {e}")
            return False


    async def __aenter__(self):
        return self


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
