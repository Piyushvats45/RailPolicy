import json
import os
import pickle
import numpy as np
from fastembed import TextEmbedding
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

def build_index(chunks_path: str, index_out_path: str, meta_out_path: str,
                 model_name: str = EMBED_MODEL_NAME):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loading embedding model: {model_name} ...")
    model = TextEmbedding(model_name=model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    embeddings = np.array(list(model.embed(texts))).astype("float32")

    # Normalize explicitly (unit length) so IndexFlatIP gives cosine similarity.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_out_path)
    with open(meta_out_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    print(f"Saved index to {index_out_path}, metadata to {meta_out_path}")

if __name__ == "__main__":
    build_index(
        chunks_path=os.path.join(BASE_DIR, "data", "chunks.json"),
        index_out_path=os.path.join(BASE_DIR, "data", "faq_index.faiss"),
        meta_out_path=os.path.join(BASE_DIR, "data", "faq_meta.pkl"),
    )
