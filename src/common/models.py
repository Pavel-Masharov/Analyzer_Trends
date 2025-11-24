from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum


class SocialPlatform(str, Enum):
    TELEGRAM = "telegram"
    VK = "vk"


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"


class SocialPost(BaseModel):
    id: str
    platform: SocialPlatform
    text: str
    content_type: ContentType = ContentType.TEXT
    author: str
    timestamp: datetime
    engagement: Dict[str, int] = Field(default_factory=dict)
    url: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


    def get_engagement_score(self) -> float:
        weights = {'likes': 1.0, 'shares': 2.0, 'comments': 1.5, 'views': 0.01}
        return sum(value * weights.get(metric, 1.0) for metric, value in self.engagement.items())


class TrendCluster(BaseModel):
    theme: str
    posts: List[SocialPost]
    confidence: float

    metadata: Dict = Field(
        default_factory=dict,
        description="Метаданные тренда: RAG обогащение, аналитика"
    )