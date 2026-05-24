import time
import re
from utils.vector_store import VectorStore
from utils.llm import GeminiLLM
from config import DEFAULT_TOP_K

class RAGPipeline:
    def __init__(self, api_key: str = None, persist_dir: str = None):
        self.vector_store = VectorStore(persist_dir) if persist_dir else VectorStore()
        self.llm = GeminiLLM(api_key=api_key)

    def update_api_key(self, api_key: str):
        """Updates the LLM client with a new API key."""
        self.llm = GeminiLLM(api_key=api_key)

    def ingest_document(self, name: str, text: str) -> int:
        """
        Ingests a document:
        1. Chunks it using LangChain RecursiveCharacterTextSplitter.
        2. Retrieves all existing chunks and merges them.
        3. Re-fits and re-vectorizes the entire corpus.
        4. Re-stores the corpus in ChromaDB to update vector dimension.
        """
        # Chunk text
        new_chunks = self.vector_store.chunk_text(text)
        if not new_chunks:
            return 0
            
        # Get existing corpus chunks and metadatas
        existing_data = self.vector_store.collection.get(include=["documents", "metadatas"])
        existing_docs = existing_data.get("documents", [])
        existing_metas = existing_data.get("metadatas", [])
        
        # Merge new chunks with existing corpus
        all_chunks = list(existing_docs) + new_chunks
        
        # Build metadatas
        all_metas = list(existing_metas)
        current_time = time.strftime("%Y-%m-%d %H:%M:%S")
        for i, chunk in enumerate(new_chunks):
            all_metas.append({
                "source": name,
                "chunk_index": i,
                "timestamp": current_time
            })
            
        # Vectorize entire corpus
        vectors = self.vector_store.vectorize(all_chunks)
        
        # Store in ChromaDB (re-creates collection)
        self.vector_store.store(all_chunks, vectors, all_metas)
        
        return len(new_chunks)

    def delete_document(self, source_name: str) -> bool:
        """Deletes all chunks belonging to a source and re-indexes the store."""
        self.vector_store.delete_document(source_name)
        return True

    def clear_database(self):
        """Wipes the database."""
        self.vector_store.clear_all()

    def query(self, question: str, top_k: int = 3) -> tuple[str, list[dict], list[float], dict]:
        """
        Executes the query pipeline:
        1. Preprocess query (lowercase, remove special chars)
        2. TF-IDF vectorize the query
        3. ChromaDB search → retrieve top-k chunks via cosine similarity
        4. Build LangChain context string from chunks
        5. Pass to Gemini via LangChain PromptTemplate
        6. Return answer + retrieved chunks + similarity scores + metrics
        """
        metrics = {
            "retrieval_latency": 0.0,
            "generation_latency": 0.0,
            "total_latency": 0.0,
            "chunks_searched": 0,
            "avg_similarity_score": 0.0,
            "preprocessed_query": "",
            "retrieval_precision": 0.0,
            "answer_relevancy": 0.0,
            "faithfulness_score": 0.0,
            "context_recall": 0.0,
            "overall_score": 0.0
        }
        
        total_start = time.time()
        
        # 1. Preprocess query
        clean_q = question.lower()
        clean_q = re.sub(r'[^a-zA-Z0-9\s]', '', clean_q).strip()
        metrics["preprocessed_query"] = clean_q
        
        # Measure retrieval & similarity ranking
        retrieval_start = time.time()
        retrieved_chunks = self.vector_store.search(question, k=top_k)
        metrics["retrieval_latency"] = time.time() - retrieval_start
        
        # Get count of total chunks searched
        stats = self.vector_store.get_collection_stats()
        metrics["chunks_searched"] = stats["total_chunks"]
        
        if not retrieved_chunks:
            metrics["total_latency"] = time.time() - total_start
            if stats.get("total_documents", 0) == 0:
                return (
                    "⚠️ It looks like there are no documents in my knowledge base. Please upload some files or ingest URLs first so I can assist you!",
                    [],
                    [],
                    metrics
                )
            else:
                return (
                    "Based on the retrieved documents, the context does not contain any information related to your query.",
                    [],
                    [],
                    metrics
                )
            
        # Extract scores
        similarity_scores = [chunk["similarity_score"] for chunk in retrieved_chunks]
        metrics["avg_similarity_score"] = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        
        # 2. Build LangChain context string from chunks
        context_parts = []
        for c in retrieved_chunks:
            source = c["metadata"].get("source", "Unknown Source")
            content = c["page_content"]
            context_parts.append(f"--- [Source: {source}] ---\n{content}")
        context_str = "\n\n".join(context_parts)
        
        # 3. Gemini LLM Generation
        generation_start = time.time()
        try:
            if not self.llm.is_configured():
                answer = "⚠️ Google Gemini API key is missing. Please set it in the sidebar to chat."
            else:
                answer = self.llm.generate_answer(context=context_str, question=question)
        except Exception as e:
            answer = f"⚠️ Generation failed: {str(e)}"
            
        metrics["generation_latency"] = time.time() - generation_start
        metrics["total_latency"] = time.time() - total_start
        
        # Calculate RAG Metrics
        # 1. Retrieval Precision
        retrieval_precision = metrics["avg_similarity_score"]
        
        # 2. Answer Relevancy
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            q_vec = self.vector_store.vectorizer.transform([clean_q])
            ans_clean = answer.lower()
            ans_clean = re.sub(r'[^a-zA-Z0-9\s]', '', ans_clean).strip()
            a_vec = self.vector_store.vectorizer.transform([ans_clean])
            answer_relevancy = float(cosine_similarity(q_vec, a_vec).flatten()[0])
        except Exception:
            answer_relevancy = 0.0
            
        # Tokenizer helper
        def get_tokens(text):
            text = text.lower()
            text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
            return set(text.split())
            
        ans_tokens = get_tokens(answer)
        ctx_tokens = get_tokens(context_str)
        
        # 3. Faithfulness Score (Jaccard)
        if not ans_tokens or not ctx_tokens:
            faithfulness_score = 0.0
        else:
            intersection = ans_tokens.intersection(ctx_tokens)
            union = ans_tokens.union(ctx_tokens)
            faithfulness_score = len(intersection) / len(union)
            
        # 4. Context Recall
        if not ctx_tokens:
            context_recall = 0.0
        else:
            intersection = ans_tokens.intersection(ctx_tokens)
            context_recall = len(intersection) / len(ctx_tokens)
            
        # 5. Overall RAG Score (Weighted Average - Equal weights)
        overall_score = (retrieval_precision + answer_relevancy + faithfulness_score + context_recall) / 4.0
        
        metrics.update({
            "retrieval_precision": retrieval_precision,
            "answer_relevancy": answer_relevancy,
            "faithfulness_score": faithfulness_score,
            "context_recall": context_recall,
            "overall_score": overall_score
        })
        
        return answer, retrieved_chunks, similarity_scores, metrics
