import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings, chunks):
        self.chunks = chunks
        self.embeddings = np.array(embeddings).astype('float32')
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
    def search(self, query_embedding, top_k=5):
        """Retrieve top-k relevant chunks"""
        query_vector = np.array([query_embedding]).astype('float32')
        distances, indices = self.index.search(query_vector, top_k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            results.append({
                'chunk': self.chunks[idx],
                'score': float(dist),
                'index': int(idx)
            })
            
        return results
        
        