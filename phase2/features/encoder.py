from sentence_transformers import SentenceTransformer

class TaskEncoder:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    # Task text to 384-dimensional vector. Similar tasks have similar vectors
    def encode(self, text):
        return self.model.encode(text, normalize_embeddings=True)