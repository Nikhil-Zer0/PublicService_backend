import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DIMENSION = 384
_model = None  # Don't load at import time

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model

class VectorDB:
    def __init__(self, index_path="faiss_index.pkl"):
        self.index_path = index_path
        self.index = None
        self.id_map = {}
        self.load_index()  # Try loading existing
        
    def load_index(self):
        if os.path.exists(self.index_path):
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.index = data["index"]
                self.id_map = data["id_map"]
        else:
            self.index = faiss.IndexFlatL2(DIMENSION)
            
    def save_index(self):
        with open(self.index_path, "wb") as f:
            pickle.dump({"index": self.index, "id_map": self.id_map}, f)
    
    def add_vector(self, mongo_id: str, embedding: list):
        if self.index is None:
            self.load_index()
        vector = np.array([embedding], dtype=np.float32)
        faiss_id = len(self.id_map)
        self.index.add(vector)
        self.id_map[faiss_id] = mongo_id
        self.save_index()
    
    # def add_vector(self, mongo_id: str, embedding: list):
    #     vector = np.array([embedding], dtype=np.float32)
    #     faiss_id = len(self.id_map)
    #     self.index.add(vector)
    #     self.id_map[faiss_id] = mongo_id
    
    def query_vectors(self, query_embedding: list, top_k: int = 3):
        vector = np.array([query_embedding], dtype=np.float32)
        _, indices = self.index.search(vector, top_k)
        return [self.id_map[i] for i in indices[0] if i in self.id_map]

def embed_text(text: str):
    model = get_model()  # Load only when first used
    embedding = model.encode([text])[0]
    return embedding.tolist()