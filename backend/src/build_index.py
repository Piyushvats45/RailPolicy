import json
import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/


def build_index(chunks_path: str, index_out_path: str, meta_out_path: str,
                 model_name: str = "all-MiniLM-L6-v2"):
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks ...")
    # normalize_embeddings=True -> unit-length vectors, so a simple
    # inner-product index gives us cosine similarity "for free".
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

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
