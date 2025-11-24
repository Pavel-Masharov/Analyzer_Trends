from abc import ABC, abstractmethod
from typing import List
import pandas as pd

from src.common.models import TrendCluster


class BaseTrendAnalyzer(ABC):
    """Base class for analyzers"""

    @abstractmethod
    async def analyze_trends(self, df: pd.DataFrame) -> List[TrendCluster]:
        """Analyzes trends and returns find trends"""
        pass
