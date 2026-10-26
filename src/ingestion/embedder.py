from loguru import logger
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import LegalChunk


class LegalEmbedder:
    """Generates embeddings for legal text chunks using sentence-transformers.

    Uses a multilingual model that supports Bangla text out of the box.
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Embed multiple texts in batches."""
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return embeddings.tolist()

    def embed_chunks(self, chunks: list[LegalChunk], batch_size: int = 32) -> list[list[float]]:
        """Embed a list of LegalChunks. Uses section title + text for richer context."""
        texts = [f"{chunk.section_title}\n{chunk.text}" for chunk in chunks]
        logger.info(f"Embedding {len(texts)} chunks (batch_size={batch_size})")
        return self.embed_texts(texts, batch_size=batch_size)
