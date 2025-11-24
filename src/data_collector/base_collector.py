from abc import ABC, abstractmethod
from typing import List
from src.common.models import SocialPost
from configs.base_config import DataSourceConfig


class BaseDataCollector(ABC):
    """Base class for data collector"""

    def __init__(self, config: DataSourceConfig):
        self.config = config
        print(f"Инициализирован сборщик для {config.platform.value}")

    @abstractmethod
    async def collect_recent_posts(self, hours: int = 24) -> List[SocialPost]:
        """collect posts at time. Need relese at child classes"""
        pass
