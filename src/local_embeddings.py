import logging
from typing import List

logger = logging.getLogger(__name__)

class SentenceTransformersPipeline:
    """
    Singleton wrapper for sentence-transformers embedding model.
    Loads the model once globally at server startup to prevent latency spikes.
    """
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load_model(self, model_name: str = "all-MiniLM-L6-v2"):
        """Eagerly load the embedding model."""
        if self._model is not None:
            return

        logger.info("[EMBEDDING CLIENT] Loading SentenceTransformer '%s' into process cache \u2014 this happens exactly once per process lifetime.", model_name)
        try:
            # We import here to avoid slow imports at module load time for environments not using this transport
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            logger.info("[EMBEDDING CLIENT] SentenceTransformer '%s' loaded and cached.", model_name)
            logger.info("[EMBEDDING CLIENT] Warm-up complete \u2014 model is hot and ready.")
        except Exception as e:
            logger.error("[EMBEDDING CLIENT] Failed to load SentenceTransformer '%s': %s", model_name, e)
            # Do not crash the server on failure to load, allow fallback or manual intervention
            self._model = None

    async def embed(self, query: str) -> list[float]:
        """Embed a single query string."""
        if self._model is None:
            logger.error("[EMBEDDING CLIENT] SentenceTransformer model is not loaded. Returning empty vector.")
            return []
        
        try:
            # sentence_transformers encode is synchronous and CPU bound, but generally fast for a single query
            embedding = self._model.encode([query])[0].tolist()
            return embedding
        except Exception as e:
            logger.error("[EMBEDDING CLIENT] Error embedding query: %s", e)
            return []

    async def simple_batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed a list of text strings."""
        if not texts:
            return []
        
        if self._model is None:
            logger.error("[EMBEDDING CLIENT] SentenceTransformer model is not loaded. Returning empty vectors.")
            return [[] for _ in texts]
        
        try:
            embeddings = self._model.encode(texts).tolist()
            return embeddings
        except Exception as e:
            logger.error("[EMBEDDING CLIENT] Error batch embedding %d texts: %s", len(texts), e)
            return [[] for _ in texts]

# Shared singleton instance
sentence_transformers_pipeline = SentenceTransformersPipeline()
