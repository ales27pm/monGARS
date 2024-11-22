
import faiss
import numpy as np

if __name__ == "__main__":
    try:
        dimensions = 128
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(dimensions))
        print("FAISS-GPU index initialized successfully.")

        embeddings = np.random.random((10, dimensions)).astype('float32')
        index.add(embeddings)
        print("Added embeddings to FAISS-GPU index.")

        query = np.random.random((1, dimensions)).astype('float32')
        distances, indices = index.search(query, 5)
        print("Search results:", distances, indices)
    except Exception as e:
        print("FAISS-GPU test failed:", e)
