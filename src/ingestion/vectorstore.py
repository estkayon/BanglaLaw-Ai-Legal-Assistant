import chromadb
from loguru import logger

from src.ingestion.chunker import LegalChunk
from src.ingestion.embedder import LegalEmbedder
from src.utils.config import settings


class LegalVectorStore:
    """ChromaDB vector store for legal document chunks.

    Operations are idempotent — re-inserting the same chunk_id overwrites
    the previous entry instead of duplicating it.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedder: LegalEmbedder | None = None,
    ):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name or settings.chroma_collection_name

        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = embedder

        logger.info(
            f"Vector store ready: {self.collection_name} "
            f"({self.collection.count()} existing documents)"
        )

    def _ensure_embedder(self) -> LegalEmbedder:
        """Lazy-load embedder on first use."""
        if self.embedder is None:
            self.embedder = LegalEmbedder()
        return self.embedder

    def add_chunks(self, chunks: list[LegalChunk], batch_size: int = 32) -> int:
        """Add or update chunks in the vector store. Returns count of upserted chunks."""
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        embedder = self._ensure_embedder()

        # Process in batches to avoid memory issues
        total_added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [chunk.chunk_id for chunk in batch]
            documents = [chunk.text for chunk in batch]
            metadatas = [chunk.metadata for chunk in batch]
            embeddings = embedder.embed_chunks(batch, batch_size=batch_size)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total_added += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} chunks")

        logger.info(f"Upserted {total_added} chunks into '{self.collection_name}'")
        return total_added

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> dict:
        """Query the vector store and return matching chunks with scores."""
        embedder = self._ensure_embedder()
        query_embedding = embedder.embed_text(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return {
            "ids": results["ids"][0] if results["ids"] else [],
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
        }

    def delete_by_act(self, act_id: str) -> None:
        """Remove all chunks belonging to a specific act."""
        self.collection.delete(where={"act_id": act_id})
        logger.info(f"Deleted chunks for act_id={act_id}")

    def reset(self) -> None:
        """Delete and recreate the collection. Use with caution."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Collection '{self.collection_name}' reset")

    @property
    def count(self) -> int:
        return self.collection.count()
