import numpy as np
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """A service for creating vector representations of text"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info(f"Uploading a model for embeddings: {model_name}")

        self.model = SentenceTransformer(model_name)
        logger.success("The embedding model has been loaded")

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Converts texts into vectors"""

        if not texts:
            return np.array([])

        logger.info(f"Encoding {len(texts)} texts to vectors")

        # text in numbers
        embeddings = self.model.encode(texts, show_progress_bar=False)

        logger.success(f"Create {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        return embeddings

    def get_similarity(self, text1: str, text2: str) -> float:
        """Calculates the semantic similarity of two texts"""

        emb1 = self.model.encode([text1])
        emb2 = self.model.encode([text2])

        # Cosine similarity between vectors
        similarity = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
        return float(similarity)
