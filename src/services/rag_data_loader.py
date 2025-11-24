import pandas as pd
import json
import aiofiles
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.common.rag_models import HistoricalTrend
from src.trend_analyzer.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RAGDataLoader:
    """External data loader for the RAG system"""

    def __init__(self, rag_service, embedding_service: EmbeddingService = None):
        self.rag_service = rag_service
        self.embedding_service = embedding_service or EmbeddingService()


    async def load_csv_data(self, csv_path: str) -> int:
        """Loads trends from a CSV file into RAG"""

        if not Path(csv_path).exists():
            logger.error(f"CSV file not found: {csv_path}")
            return 0

        try:
            df = pd.read_csv(csv_path)
            loaded_count = 0

            for _, row in df.iterrows():
                embedding = self.embedding_service.encode_texts([str(row['theme'])])[0].tolist()

                historical_trend = HistoricalTrend(
                    id=f"external_csv_{hash(row['theme'])}_{loaded_count}",
                    theme=str(row['theme']),
                    posts=[],
                    confidence=float(row.get('confidence', 0.8)),
                    timestamp=datetime.strptime(row.get('date', '2024-01-01'), '%Y-%m-%d'),
                    embedding=embedding,
                    metadata={
                        "source": "external_csv",
                        "category": row.get('category', 'general'),
                        "keywords": row.get('keywords', '').split(','),
                        "original_source": row.get('source', 'unknown')
                    }
                )

                await self.rag_service.vector_store.add_trend(historical_trend)
                loaded_count += 1

            logger.info(f"Load {loaded_count} trends into CSV: {csv_path}")
            return loaded_count

        except Exception as e:
            logger.error(f"Error load CSV {csv_path}: {e}")
            return 0


    async def load_json_data(self, json_path: str) -> int:
        """Load trends from a JSON file into RAG"""

        if not Path(json_path).exists():
            logger.error(f"The JSON file was not found: {json_path}")
            return 0

        try:
            async with aiofiles.open(json_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)

            loaded_count = 0

            for item in data:
                embedding = self.embedding_service.encode_texts([str(item['theme'])])[0].tolist()

                date_str = item.get('date', '2024-01-01')
                try:
                    timestamp = datetime.strptime(date_str, '%Y-%m-%d')
                except:
                    timestamp = datetime.now()

                historical_trend = HistoricalTrend(
                    id=item.get('id', f"external_json_{hash(item['theme'])}_{loaded_count}"),
                    theme=str(item['theme']),
                    posts=[],
                    confidence=float(item.get('confidence', 0.7)),
                    timestamp=timestamp,
                    embedding=embedding,
                    metadata={
                        "source": "external_json",
                        "category": item.get('category', 'general'),
                        "tags": item.get('tags', []),
                        "description": item.get('description', ''),
                        "engagement_metrics": item.get('engagement_metrics', {}),
                        "original_source": item.get('source', 'unknown')
                    }
                )

                await self.rag_service.vector_store.add_trend(historical_trend)
                loaded_count += 1

            logger.info(f"Load {loaded_count} trends into JSON: {json_path}")
            return loaded_count

        except Exception as e:
            logger.error(f"Error load JSON {json_path}: {e}")
            return 0


    async def load_from_directory(self, data_dir: str) -> Dict[str, int]:
        """Downloads all data from a directory"""

        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.error(f"Directory was not found: {data_dir}")
            return {}

        results = {}

        for csv_file in data_dir.glob("*.csv"):
            count = await self.load_csv_data(str(csv_file))
            results[csv_file.name] = count

        for json_file in data_dir.glob("*.json"):
            count = await self.load_json_data(str(json_file))
            results[json_file.name] = count

        total_loaded = sum(results.values())
        logger.info(f"Load {total_loaded} trends into {len(results)} files")

        return results


    async def create_sample_external_trend(self, theme: str, confidence: float = 0.8,
                                           category: str = "general") -> str:
        """Creates and adds a custom external trend"""

        embedding = self.embedding_service.encode_texts([theme])[0].tolist()

        historical_trend = HistoricalTrend(
            id=f"custom_{hash(theme)}_{datetime.now().timestamp()}",
            theme=theme,
            posts=[],
            confidence=confidence,
            timestamp=datetime.now(),
            embedding=embedding,
            metadata={
                "source": "custom",
                "category": category,
                "added_manually": True
            }
        )

        trend_id = await self.rag_service.vector_store.add_trend(historical_trend)
        logger.info(f"Added a custom trend: {theme}")

        return trend_id
