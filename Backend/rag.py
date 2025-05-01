import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
DIMENSION = 384

class VectorDB:
    def __init__(self):
        self.index = faiss.IndexFlatL2(DIMENSION)
        self.id_map = {}
    
    def add_vector(self, mongo_id: str, embedding: list):
        vector = np.array([embedding], dtype=np.float32)
        faiss_id = len(self.id_map)
        self.index.add(vector)
        self.id_map[faiss_id] = mongo_id
    
    def query_vectors(self, query_embedding: list, top_k: int = 3):
        vector = np.array([query_embedding], dtype=np.float32)
        _, indices = self.index.search(vector, top_k)
        return [self.id_map[i] for i in indices[0] if i in self.id_map]

def embed_text(text: str):
    embedding = model.encode([text])[0]
    return embedding.tolist()