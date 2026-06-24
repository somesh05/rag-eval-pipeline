from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"


class RAGVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        print("Loading embedding model (first run will download it once)....")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = None
        self.bm25_corpus_ids = []
        self.bm25_corpus_texts = []

    def add_documents(self, chunks: List[Dict]):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        ids = [c["id"] for c in chunks]
        metadatas = [
            {"source": c["source"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ]
        embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        self._rebuild_bm25_index()
        print(f"Added {len(chunks)} chunks to the vector store")

    def clear(self):
        """Delete every document from the vector store and reset the BM25 index."""
        self.client.delete_collection(name="rag_documents")
        self.collection = self.client.get_or_create_collection(
            name="rag_documents",
            metadata={"hnsw:space": "cosine"},
        )
        self.bm25 = None
        self.bm25_corpus_ids = []
        self.bm25_corpus_texts = []

    def _rebuild_bm25_index(self):
        """Pull the full corpus back out of Chroma and rebuild the BM25 index."""
        all_data = self.collection.get(include=["documents"])
        self.bm25_corpus_ids = all_data["ids"]
        self.bm25_corpus_texts = all_data["documents"]
        if self.bm25_corpus_texts:
            tokenized = [doc.lower().split() for doc in self.bm25_corpus_texts]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def _vector_search(self, query: str, k: int) -> List[Dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "score": 1 - results["distances"][0][i],
            })
        return hits

    def _bm25_search(self, query: str, k: int) -> List[Dict]:
        if self.bm25 is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        scored = list(zip(self.bm25_corpus_ids, self.bm25_corpus_texts, scores))
        scored.sort(key=lambda x: x[2], reverse=True)
        top = scored[:k]
        max_score = max((s for _, _, s in top), default=1) or 1
        hits = []
        for chunk_id, text, score in top:
            hits.append({"id": chunk_id, "text": text, "score": score / max_score})
        return hits

    def retrieve(self, query: str, k: int = 4, mode: str = "hybrid") -> List[Dict]:
        """mode: "vector" for pure semantic search, "hybrid" for BM25 + vector fusion."""
        if mode == "vector":
            return self._vector_search(query, k)
        elif mode == "hybrid":
            vector_hits = self._vector_search(query, k=k * 2)
            bm25_hits = self._bm25_search(query, k=k * 2)

            combined: Dict[str, Dict] = {}
            for hit in vector_hits:
                combined[hit["id"]] = {**hit, "vector_score": hit["score"], "bm25_score": 0.0}
            for hit in bm25_hits:
                if hit["id"] in combined:
                    combined[hit["id"]]["bm25_score"] = hit["score"]
                else:
                    combined[hit["id"]] = {**hit, "vector_score": 0.0, "bm25_score": hit["score"]}

            for data in combined.values():
                data["score"] = 0.5 * data["vector_score"] + 0.5 * data["bm25_score"]

            merged = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
            return merged[:k]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")
