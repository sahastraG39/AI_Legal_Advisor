"""
Vector Store Module for AI Legal Document Explainer

This module handles vector storage and similarity search for:
- Document embeddings
- Legal text search
- Semantic similarity
- Multiple vector database backends
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import json
import numpy as np

# Environment variables
from dotenv import load_dotenv

# Vector databases
import pinecone
import faiss
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Embeddings
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Main vector store class for handling document embeddings and similarity search."""
    
    def __init__(self, 
                 store_type: str = 'chroma',
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 dimension: int = 384):
        """
        Initialize vector store.
        
        Args:
            store_type: Type of vector store ('pinecone', 'faiss', 'chroma', 'qdrant')
            embedding_model: Sentence transformer model name
            dimension: Embedding dimension
        """
        self.store_type = store_type
        self.dimension = dimension
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize the specified vector store
        self._initialize_store()
        
        # Document metadata storage
        self.documents = {}
        self.document_counter = 0
    
    def _initialize_store(self):
        """Initialize the specified vector store backend."""
        try:
            if self.store_type == 'pinecone':
                self._initialize_pinecone()
            elif self.store_type == 'faiss':
                self._initialize_faiss()
            elif self.store_type == 'chroma':
                self._initialize_chroma()
            elif self.store_type == 'qdrant':
                self._initialize_qdrant()
            else:
                raise ValueError(f"Unsupported store type: {self.store_type}")
                
            logger.info(f"Vector store '{self.store_type}' initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone vector database."""
        api_key = os.getenv('PINECONE_API_KEY')
        environment = os.getenv('PINECONE_ENVIRONMENT')
        
        if not api_key or not environment:
            raise ValueError("PINECONE_API_KEY and PINECONE_ENVIRONMENT must be set")
        
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        index_name = "legal-documents"
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=self.dimension,
                metric="cosine"
            )
        
        self.pinecone_index = pinecone.Index(index_name)
    
    def _initialize_faiss(self):
        """Initialize FAISS vector database."""
        self.faiss_index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.faiss_vectors = []
        self.faiss_metadata = []
    
    def _initialize_chroma(self):
        """Initialize ChromaDB vector database."""
        persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './chroma_db')
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        
        # Create or get collection
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name="legal_documents",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _initialize_qdrant(self):
        """Initialize Qdrant vector database."""
        # For local development, use in-memory storage
        self.qdrant_client = QdrantClient(":memory:")
        
        # Create collection
        self.qdrant_client.create_collection(
            collection_name="legal_documents",
            vectors_config=VectorParams(size=self.dimension, distance=Distance.COSINE)
        )
    
    def add_document(self, 
                    text: str,
                    metadata: Dict[str, Any],
                    document_id: Optional[str] = None) -> str:
        """
        Add a document to the vector store.
        
        Args:
            text: Document text content
            metadata: Document metadata
            document_id: Optional custom document ID
            
        Returns:
            Document ID
        """
        try:
            # Generate document ID if not provided
            if not document_id:
                document_id = f"doc_{self.document_counter}"
                self.document_counter += 1
            
            # Generate embeddings
            embeddings = self.embedding_model.encode([text])[0]
            
            # Store document metadata
            self.documents[document_id] = {
                'text': text,
                'metadata': metadata,
                'embedding': embeddings
            }
            
            # Add to vector store
            if self.store_type == 'pinecone':
                self._add_to_pinecone(document_id, embeddings, metadata)
            elif self.store_type == 'faiss':
                self._add_to_faiss(document_id, embeddings, metadata)
            elif self.store_type == 'chroma':
                self._add_to_chroma(document_id, text, embeddings, metadata)
            elif self.store_type == 'qdrant':
                self._add_to_qdrant(document_id, embeddings, metadata)
            
            logger.info(f"Document {document_id} added to vector store")
            return document_id
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def _add_to_pinecone(self, doc_id: str, embeddings: np.ndarray, metadata: Dict):
        """Add document to Pinecone."""
        self.pinecone_index.upsert(
            vectors=[(doc_id, embeddings.tolist(), metadata)]
        )
    
    def _add_to_faiss(self, doc_id: str, embeddings: np.ndarray, metadata: Dict):
        """Add document to FAISS."""
        self.faiss_index.add(embeddings.reshape(1, -1))
        self.faiss_vectors.append(embeddings)
        self.faiss_metadata.append({'id': doc_id, **metadata})
    
    def _add_to_chroma(self, doc_id: str, text: str, embeddings: np.ndarray, metadata: Dict):
        """Add document to ChromaDB."""
        self.chroma_collection.add(
            documents=[text],
            embeddings=[embeddings.tolist()],
            metadatas=[metadata],
            ids=[doc_id]
        )
    
    def _add_to_qdrant(self, doc_id: str, embeddings: np.ndarray, metadata: Dict):
        """Add document to Qdrant."""
        self.qdrant_client.upsert(
            collection_name="legal_documents",
            points=[
                PointStruct(
                    id=doc_id,
                    vector=embeddings.tolist(),
                    payload=metadata
                )
            ]
        )
    
    def search_similar(self, 
                      query: str,
                      top_k: int = 5,
                      threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of top results to return
            threshold: Similarity threshold
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Generate query embeddings
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search in vector store
            if self.store_type == 'pinecone':
                results = self._search_pinecone(query_embedding, top_k)
            elif self.store_type == 'faiss':
                results = self._search_faiss(query_embedding, top_k)
            elif self.store_type == 'chroma':
                results = self._search_chroma(query, top_k)
            elif self.store_type == 'qdrant':
                results = self._search_qdrant(query_embedding, top_k)
            
            # Filter by threshold and add document text
            filtered_results = []
            for result in results:
                if result['score'] >= threshold:
                    doc_id = result['id']
                    if doc_id in self.documents:
                        result['text'] = self.documents[doc_id]['text']
                        result['metadata'] = self.documents[doc_id]['metadata']
                        filtered_results.append(result)
            
            return filtered_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def _search_pinecone(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search in Pinecone."""
        results = self.pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )
        
        return [
            {
                'id': match.id,
                'score': match.score,
                'metadata': match.metadata
            }
            for match in results.matches
        ]
    
    def _search_faiss(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search in FAISS."""
        # Normalize query embedding for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        scores, indices = self.faiss_index.search(
            query_norm.reshape(1, -1), top_k
        )
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.faiss_metadata):
                results.append({
                    'id': self.faiss_metadata[idx]['id'],
                    'score': float(score),
                    'metadata': self.faiss_metadata[idx]
                })
        
        return results
    
    def _search_chroma(self, query: str, top_k: int) -> List[Dict]:
        """Search in ChromaDB."""
        results = self.chroma_collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        return [
            {
                'id': doc_id,
                'score': score,
                'metadata': metadata
            }
            for doc_id, score, metadata in zip(
                results['ids'][0],
                results['distances'][0],
                results['metadatas'][0]
            )
        ]
    
    def _search_qdrant(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Search in Qdrant."""
        results = self.qdrant_client.search(
            collection_name="legal_documents",
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        return [
            {
                'id': result.id,
                'score': result.score,
                'metadata': result.payload
            }
            for result in results
        ]
    
    def batch_add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents in batch.
        
        Args:
            documents: List of documents with 'text' and 'metadata' keys
            
        Returns:
            List of document IDs
        """
        document_ids = []
        
        for doc in documents:
            try:
                doc_id = self.add_document(
                    text=doc['text'],
                    metadata=doc['metadata'],
                    document_id=doc.get('id')
                )
                document_ids.append(doc_id)
            except Exception as e:
                logger.error(f"Error adding document in batch: {e}")
        
        return document_ids
    
    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        return self.documents.get(document_id)
    
    def update_document(self, 
                       document_id: str,
                       text: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing document.
        
        Args:
            document_id: Document ID to update
            text: New text content (optional)
            metadata: New metadata (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if document_id not in self.documents:
                return False
            
            # Update local storage
            if text is not None:
                self.documents[document_id]['text'] = text
                # Regenerate embeddings
                new_embeddings = self.embedding_model.encode([text])[0]
                self.documents[document_id]['embedding'] = new_embeddings
            
            if metadata is not None:
                self.documents[document_id]['metadata'].update(metadata)
            
            # Update vector store
            if text is not None:
                embeddings = self.documents[document_id]['embedding']
                metadata = self.documents[document_id]['metadata']
                
                if self.store_type == 'pinecone':
                    self._add_to_pinecone(document_id, embeddings, metadata)
                elif self.store_type == 'faiss':
                    # FAISS doesn't support updates, would need to rebuild
                    logger.warning("FAISS doesn't support document updates")
                elif self.store_type == 'chroma':
                    self._add_to_chroma(document_id, text, embeddings, metadata)
                elif self.store_type == 'qdrant':
                    self._add_to_qdrant(document_id, embeddings, metadata)
            
            logger.info(f"Document {document_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document {document_id}: {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        try:
            if document_id not in self.documents:
                return False
            
            # Remove from local storage
            del self.documents[document_id]
            
            # Remove from vector store
            if self.store_type == 'pinecone':
                self.pinecone_index.delete(ids=[document_id])
            elif self.store_type == 'faiss':
                logger.warning("FAISS doesn't support document deletion")
            elif self.store_type == 'chroma':
                self.chroma_collection.delete(ids=[document_id])
            elif self.store_type == 'qdrant':
                self.qdrant_client.delete(
                    collection_name="legal_documents",
                    points_selector=[document_id]
                )
            
            logger.info(f"Document {document_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            'store_type': self.store_type,
            'total_documents': len(self.documents),
            'embedding_dimension': self.dimension,
            'embedding_model': self.embedding_model.get_sentence_embedding_dimension()
        }
        
        if self.store_type == 'pinecone':
            stats['pinecone_index_stats'] = self.pinecone_index.describe_index_stats()
        elif self.store_type == 'faiss':
            stats['faiss_index_size'] = self.faiss_index.ntotal
        elif self.store_type == 'chroma':
            stats['chroma_collection_count'] = self.chroma_collection.count()
        elif self.store_type == 'qdrant':
            stats['qdrant_collection_info'] = self.qdrant_client.get_collection(
                collection_name="legal_documents"
            )
        
        return stats
    
    def clear_all(self):
        """Clear all documents from the vector store."""
        try:
            # Clear local storage
            self.documents.clear()
            self.document_counter = 0
            
            # Clear vector store
            if self.store_type == 'pinecone':
                # Pinecone doesn't support clearing all, would need to delete index
                logger.warning("Pinecone doesn't support clearing all documents")
            elif self.store_type == 'faiss':
                self.faiss_index = faiss.IndexFlatIP(self.dimension)
                self.faiss_vectors.clear()
                self.faiss_metadata.clear()
            elif self.store_type == 'chroma':
                self.chroma_collection.delete(where={})
            elif self.store_type == 'qdrant':
                self.qdrant_client.delete_collection("legal_documents")
                self._initialize_qdrant()
            
            logger.info("All documents cleared from vector store")
            
        except Exception as e:
            logger.error(f"Error clearing vector store: {e}")


def main():
    """Test function for the vector store."""
    try:
        # Initialize vector store (using ChromaDB as default for testing)
        vector_store = VectorStore(store_type='chroma')
        
        # Sample legal documents
        sample_documents = [
            {
                'text': 'This employment agreement is between Company and Employee.',
                'metadata': {'type': 'contract', 'category': 'employment'}
            },
            {
                'text': 'The defendant is charged with breach of contract.',
                'metadata': {'type': 'case_law', 'category': 'contract_dispute'}
            },
            {
                'text': 'Section 101 defines intellectual property rights.',
                'metadata': {'type': 'statute', 'category': 'intellectual_property'}
            }
        ]
        
        print("Testing Vector Store...")
        print("=" * 50)
        
        # Add documents
        doc_ids = vector_store.batch_add_documents(sample_documents)
        print(f"Added {len(doc_ids)} documents: {doc_ids}")
        
        # Search for similar documents
        query = "employment contract terms"
        results = vector_store.search_similar(query, top_k=3)
        
        print(f"\nSearch results for '{query}':")
        for i, result in enumerate(results, 1):
            print(f"{i}. Document ID: {result['id']}")
            print(f"   Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            print(f"   Metadata: {result['metadata']}")
            print()
        
        # Get statistics
        stats = vector_store.get_statistics()
        print("Vector Store Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    except Exception as e:
        print(f"Error testing vector store: {e}")


if __name__ == "__main__":
    main()
