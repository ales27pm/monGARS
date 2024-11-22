
import faiss
import numpy as np

class FAISSOptimizer:
    def __init__(self, dimensions, gpu=True):
        self.dimensions = dimensions
        self.index = self.initialize_index(gpu)

    def initialize_index(self, gpu):
        """Initialize a FAISS index, optionally on GPU."""
        try:
            index = faiss.IndexFlatL2(self.dimensions)  # L2 distance
            if gpu:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("FAISS-GPU index initialized successfully.")
            else:
                print("FAISS-CPU index initialized successfully.")
            return index
        except Exception as e:
            print(f"Error initializing FAISS index: {e}")
            return None

    def add_embeddings(self, embeddings):
        try:
            self.index.add(embeddings)
            print(f"Added {len(embeddings)} embeddings to the FAISS index.")
        except Exception as e:
            print(f"Error adding embeddings to the FAISS index: {e}")

    def search(self, query, top_k=5):
        try:
            distances, indices = self.index.search(query, top_k)
            print(f"Search results: Distances: {distances}, Indices: {indices}")
            return distances, indices
        except Exception as e:
            print(f"Error searching FAISS index: {e}")
            return None, None

if __name__ == "__main__":
    optimizer = FAISSOptimizer(dimensions=128, gpu=True)
    embeddings = np.random.random((100, 128)).astype('float32')
    optimizer.add_embeddings(embeddings)
    query = np.random.random((1, 128)).astype('float32')
    optimizer.search(query, top_k=5)
