import streamlit as st
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def chunk_text(text, chunk_size):
    """
    Split text into chunks of approximately chunk_size characters.
    Try to split at paragraph or sentence boundaries when possible.
    """
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_pos = 0
    
    while current_pos < len(text):
        # If we're near the end, just take the rest
        if current_pos + chunk_size >= len(text):
            chunks.append(text[current_pos:])
            break
        
        # Try to find paragraph break within the chunk_size from current position
        end_pos = text.rfind('\n\n', current_pos, current_pos + chunk_size)
        
        # If no paragraph break, try to find sentence break
        if end_pos == -1:
            end_pos = text.rfind('. ', current_pos, current_pos + chunk_size)
            if end_pos != -1:
                end_pos += 2  # Include the period and space
        
        # If no sentence break, try to find any newline
        if end_pos == -1:
            end_pos = text.rfind('\n', current_pos, current_pos + chunk_size)
            if end_pos != -1:
                end_pos += 1  # Include the newline
        
        # If still no natural break, just cut at chunk_size
        if end_pos == -1 or end_pos <= current_pos:
            end_pos = current_pos + chunk_size
        
        # Add the chunk
        chunks.append(text[current_pos:end_pos])
        current_pos = end_pos
    
    return chunks

def get_active_chunk_context():
    """Get the active chunk or return empty string if no chunks available"""
    if hasattr(st.session_state, 'context_chunks') and st.session_state.context_chunks and len(st.session_state.context_chunks) > st.session_state.active_chunk:
        return st.session_state.context_chunks[st.session_state.active_chunk]
    return ""

class VectorStore:
    """Simple vector database implementation for text chunks"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None
        self.chunks = []
        self.is_initialized = False
        
    def add_documents(self, chunks):
        """Add documents to the vector store and create vectors"""
        self.chunks = chunks
        if chunks:
            self.vectors = self.vectorizer.fit_transform(chunks)
            self.is_initialized = True
        
    def similarity_search(self, query, top_k=3):
        """Search for most similar chunks to the query"""
        if not self.is_initialized or not self.chunks:
            return []
        
        # Transform query to vector space
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        
        # Get indices of top_k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Filter out chunks with zero similarity
        results = [(self.chunks[i], similarities[i]) for i in top_indices if similarities[i] > 0]
        
        # Return just the chunks
        return [chunk for chunk, score in results]

# Initialize vector store in session state if it doesn't exist
def initialize_vector_store():
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStore()

def update_vector_store():
    """Update vector store with current chunks"""
    initialize_vector_store()
    if hasattr(st.session_state, 'context_chunks') and st.session_state.context_chunks:
        st.session_state.vector_store.add_documents(st.session_state.context_chunks)

def search_context(query, top_k=3):
    """
    Search through context chunks using the appropriate method based on size.
    Returns the top k chunks that are most relevant to the query.
    """
    if not hasattr(st.session_state, 'context_chunks') or not st.session_state.context_chunks:
        return []
    
    # Check if total context size exceeds threshold (5000 characters)
    total_context_size = sum(len(chunk) for chunk in st.session_state.context_chunks)
    
    # Use vector search for large contexts
    if total_context_size > 5000:
        initialize_vector_store()
        # Make sure vector store is updated with latest chunks
        if not st.session_state.vector_store.is_initialized:
            update_vector_store()
        
        return st.session_state.vector_store.similarity_search(query, top_k)
    
    # For smaller contexts, use the simple keyword-based search
    else:
        # Extract keywords from the query (simple approach)
        keywords = re.findall(r'\b\w{3,}\b', query.lower())
        
        # Score each chunk based on keyword matches
        chunk_scores = []
        for i, chunk in enumerate(st.session_state.context_chunks):
            chunk_lower = chunk.lower()
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            chunk_scores.append((i, score))
        
        # Sort by score and get top k
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [st.session_state.context_chunks[i] for i, score in chunk_scores[:top_k] if score > 0]
        
        return top_chunks