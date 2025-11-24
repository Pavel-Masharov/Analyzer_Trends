from .base_collector import BaseDataCollector
from configs.base_config import DataSourceConfig


class TelegramCollector(BaseDataCollector):

    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
