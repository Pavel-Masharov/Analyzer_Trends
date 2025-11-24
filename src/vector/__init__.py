from .faiss_vector_store import FAISSVectorStore

def create_vector_store(config):
    """A factory for creating vector storage"""
    return FAISSVectorStore(config)
