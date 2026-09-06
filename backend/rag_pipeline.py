import os
from dotenv import load_dotenv
load_dotenv()
import time
import requests
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder
from tenacity import retry, stop_after_attempt, wait_exponential

class RAGPipeline:
    def __init__(self, qdrant_path="./qdrant_data", collection_name="msmarco_hybrid"):
        print("Initializing RAG Pipeline components...")
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = collection_name
        
        # Ensure collection exists
        from qdrant_client.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))}
            )
        
        # Load embedding models for query
        self.dense_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        
        # Load Reranker
        print("Loading reranker...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    def retrieve(self, query, top_k=20):
        # Generate query embeddings (Dense + Sparse)
        dense_vec = list(self.dense_model.embed([query]))[0].tolist()
        sparse_embed = list(self.sparse_model.embed([query]))[0]
        sparse_vec = models.SparseVector(
            indices=sparse_embed.indices.tolist(),
            values=sparse_embed.values.tolist()
        )
        
        # Hybrid Search using Prefetch and RRF Fusion
        prefetch = [
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=top_k
            )
        ]
        
        query_res = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k
        )
        hybrid_hits = query_res.points
        
        # Combine unique documents by metadata doc_id
        unique_docs = {}
        for hit in hybrid_hits:
            doc_id = hit.payload.get('doc_id', hit.id)
            if doc_id not in unique_docs:
                unique_docs[doc_id] = hit.payload['text']
                
        return list(unique_docs.values())

    def rerank(self, query, documents, top_n=3):
        if not documents:
            return []
            
        pairs = [[query, doc] for doc in documents]
        scores = self.reranker.predict(pairs)
        
        # Sort documents by score
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in doc_score_pairs[:top_n]]

    def add_document(self, text: str, doc_id: str, title: str = ""):
        import uuid
        import nltk
        from qdrant_client.models import PointStruct, SparseVector
        
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('punkt_tab')
            
        chunks = []
        
        # 1. Semantic Chunking
        sentences = nltk.sent_tokenize(text)
        current_chunk = []
        current_length = 0
        max_words = 100
        
        for sentence in sentences:
            words_count = len(sentence.split())
            if current_length + words_count > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(sentence)
            current_length += words_count
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        # 2. Sliding Window Chunking
        words = text.split()
        window_size = 150
        overlap = 30
        
        if len(words) <= window_size:
            chunks.append(text)
        else:
            i = 0
            while i < len(words):
                chunk = " ".join(words[i:i+window_size])
                chunks.append(chunk)
                i += (window_size - overlap)
                
        # Deduplicate and clean
        chunks = list(set([c.strip() for c in chunks if c.strip()]))
                
        dense_embeds = list(self.dense_model.embed(chunks))
        sparse_embeds = list(self.sparse_model.embed(chunks))
        
        points = []
        for j, chunk in enumerate(chunks):
            dense_vec = dense_embeds[j].tolist()
            sparse_vec = SparseVector(
                indices=sparse_embeds[j].indices.tolist(),
                values=sparse_embeds[j].values.tolist()
            )
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec,
                    "sparse": sparse_vec
                },
                payload={
                    "text": chunk,
                    "doc_id": doc_id,
                    "title": title
                }
            )
            points.append(point)
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Indexed document {doc_id} with {len(chunks)} chunks.")

    def get_all_documents(self):
        try:
            records, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            docs = {}
            for r in records:
                doc_id = r.payload.get("doc_id")
                title = r.payload.get("title", "Unknown Document")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = title
            return [{"doc_id": k, "title": v} for k, v in docs.items()]
        except Exception as e:
            print(f"Error fetching documents: {e}")
            return []

    def delete_document(self, doc_id: str):
        from qdrant_client.http import models
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id",
                                match=models.MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            print(f"Deleted document {doc_id} from Qdrant.")
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            raise e