import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
from vector_cluster_store_py import VectorStore

# Initialize store
store = VectorStore("/dev/sdb", 768)  # device path, vector dimension

# Store a vector
vector_id = 1
vector = np.random.randn(128).astype(np.float32)  # 128-dimensional vector
metadata = "Example vector"
store.store_vector(vector_id, vector, metadata)

# Retrieve a vector
retrieved = store.get_vector(vector_id)

# Find similar vectors
query = np.random.randn(128).astype(np.float32)
results = store.find_nearest(query, k=5)  # Returns top 5 matches

# Each result is a tuple of (vector_id, similarity_score)
for vector_id, similarity in results:
    print(f"Vector {vector_id}: Similarity {similarity:.4f}")
