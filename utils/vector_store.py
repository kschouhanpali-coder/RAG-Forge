import os
import re
import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Initialize Persistent Client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Create or Get the collection (without a default embedding function as we manage our own embeddings)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME
        )
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

        # Re-fit vectorizer if documents already exist in the persisted DB
        try:
            data = self.collection.get(include=["documents"])
            docs = data.get("documents", [])
            if docs:
                self.vectorizer.fit(docs)
        except Exception:
            pass

    def chunk_text(self, text: str) -> list[str]:
        """
        Splits text into chunks using a pure Python word-boundary-respecting splitter.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        chunk_size = 500
        chunk_overlap = 50
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # Respect word boundaries
            if end < text_len:
                lookback_limit = max(start, end - int(chunk_size * 0.2))
                boundary = -1
                for idx in range(end - 1, lookback_limit - 1, -1):
                    if text[idx] in ['\n', ' ', '\r', '\t']:
                        boundary = idx
                        break
                if boundary != -1:
                    end = boundary + 1
            
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(chunk_content)
                
            start = end - chunk_overlap
            if start >= end:
                start = end
            if end == text_len:
                break
                
        return chunks

    def vectorize(self, chunks: list[str]) -> list[list[float]]:
        """
        Fits and transforms chunks using sklearn TF-IDF Vectorizer.
        Returns dense vectors (list of list of floats).
        """
        if not chunks:
            return []
        tfidf_matrix = self.vectorizer.fit_transform(chunks)
        return tfidf_matrix.toarray().tolist()

    def store(self, chunks: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        """
        Stores chunks, TF-IDF vectors, and metadata in ChromaDB.
        Recreates the collection to ensure the embedding dimension matches.
        """
        if not chunks or vectors is None or len(vectors) == 0:
            return
            
        # Recreate collection to handle dynamic TF-IDF dimensionality changes
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
            
        self.collection = self.client.create_collection(
            name=CHROMA_COLLECTION_NAME
        )
        
        # Generate sequential unique IDs
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        self.collection.add(
            documents=chunks,
            embeddings=vectors,
            metadatas=metadatas,
            ids=ids
        )

    def search(self, query: str, k: int = 3) -> list[dict]:
        """
        Searches ChromaDB:
        1. Preprocess query (lowercase, remove special chars)
        2. TF-IDF transform query
        3. Retrieve stored embeddings and compute manual cosine similarity
        4. Return top-k chunks with metadata and similarity scores
        """
        # 1. Preprocess query
        clean_query = query.lower()
        clean_query = re.sub(r'[^a-zA-Z0-9\s]', '', clean_query).strip()
        
        if not clean_query:
            return []
            
        # Retrieve all chunks from collection
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])
        docs = data.get("documents", [])
        embeddings = data.get("embeddings", [])
        metadatas = data.get("metadatas", [])
        
        if len(docs) == 0 or embeddings is None or len(embeddings) == 0:
            return []
            
        try:
            # 2. Transform query using the fitted vectorizer
            query_vector = self.vectorizer.transform([clean_query]).toarray()
            
            # 3. Calculate cosine similarity
            doc_embeddings = np.array(embeddings)
            similarities = cosine_similarity(query_vector, doc_embeddings).flatten()
            
            # Build and sort list of ranked chunks
            scored_docs = []
            for doc, meta, score in zip(docs, metadatas, similarities):
                scored_docs.append({
                    "page_content": doc,
                    "metadata": meta or {},
                    "similarity_score": float(score)
                })
                
            scored_docs.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Filter out chunks with 0.0 similarity (meaning absolutely no term overlap)
            filtered_docs = [d for d in scored_docs if d["similarity_score"] > 0.0]
            
            # 4. Return Top-K
            return filtered_docs[:k]
            
        except Exception as e:
            print(f"Error during TF-IDF similarity calculation: {e}")
            # Fallback
            fallback_results = []
            for doc, meta in zip(docs[:k], metadatas[:k]):
                fallback_results.append({
                    "page_content": doc,
                    "metadata": meta or {},
                    "similarity_score": 0.0
                })
            return fallback_results

    def get_all_chunks(self) -> dict:
        """Returns all chunks stored in the ChromaDB collection."""
        return self.collection.get(include=["documents", "metadatas"])

    def get_collection_stats(self) -> dict:
        """
        Returns stats about the ChromaDB index.
        """
        data = self.collection.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        total_chunks = len(documents)
        
        unique_docs = set()
        total_size_bytes = 0
        for doc, meta in zip(documents, metadatas):
            total_size_bytes += len(doc.encode('utf-8'))
            if meta and "source" in meta:
                unique_docs.add(meta["source"])
                
        return {
            "total_documents": len(unique_docs),
            "total_chunks": total_chunks,
            "collection_size_bytes": total_size_bytes
        }

    # Helper methods for document deletion/source check
    def delete_document(self, source_name: str) -> None:
        """Deletes all chunks belonging to a specific source and re-indexes remaining corpus."""
        data = self.collection.get(include=["documents", "metadatas"])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        
        # Keep everything except what belongs to source_name
        keep_docs = []
        keep_metas = []
        for doc, meta in zip(documents, metadatas):
            if meta and meta.get("source") != source_name:
                keep_docs.append(doc)
                keep_metas.append(meta)
                
        if keep_docs:
            vectors = self.vectorize(keep_docs)
            self.store(keep_docs, vectors, keep_metas)
        else:
            self.clear_all()

    def clear_all(self) -> None:
        """Wipes the collection."""
        try:
            self.client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
        self.vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)

    def get_document_sources(self) -> list[str]:
        """Returns unique source names currently indexed."""
        data = self.collection.get(include=["metadatas"])
        metadatas = data.get("metadatas", [])
        if not metadatas:
            return []
        sources = set()
        for meta in metadatas:
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(list(sources))
