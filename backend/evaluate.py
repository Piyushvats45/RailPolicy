import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_SET = [
    {"question": "How much does it cost to cancel a confirmed ticket 3 days before travel?", "expected_chunk": "chunk_1"},
    {"question": "Will I automatically get a refund if my ticket stays on the waiting list?", "expected_chunk": "chunk_2"},
    {"question": "What happens to a RAC ticket that doesn't confirm?", "expected_chunk": "chunk_3"},
    {"question": "How do I claim a refund after the chart is already prepared?", "expected_chunk": "chunk_4"},
    {"question": "How many days does a normal cancellation refund take to arrive?", "expected_chunk": "chunk_5"},
    {"question": "Can I get money back for a confirmed Tatkal booking I need to cancel?", "expected_chunk": "chunk_6"},
    {"question": "What is the clerkage deduction and when is it charged?", "expected_chunk": "chunk_7"},
    {"question": "Can I cancel only some passengers in a group booking, not all of them?", "expected_chunk": "chunk_8"},
]


def evaluate_hit_rate(index_path: str, meta_path: str, model_name: str, k: int) -> float:
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        chunks = pickle.load(f)

    hits = 0
    for case in TEST_SET:
        query_vec = model.encode([case["question"]], normalize_embeddings=True)
        query_vec = np.array(query_vec).astype("float32")
        _, indices = index.search(query_vec, k)
        retrieved_ids = {chunks[i]["id"] for i in indices[0] if i != -1}
        if case["expected_chunk"] in retrieved_ids:
            hits += 1

    return hits / len(TEST_SET)


if __name__ == "__main__":
    print("Evaluating retrieval quality (Hit Rate @ k)\n")
    for k in [1, 2, 3, 5]:
        hit_rate = evaluate_hit_rate(
            index_path=os.path.join(BASE_DIR, "data", "faq_index.faiss"),
            meta_path=os.path.join(BASE_DIR, "data", "faq_meta.pkl"),
            model_name="all-MiniLM-L6-v2",
            k=k,
        )
        print(f"k={k}: Hit Rate = {hit_rate:.2%}")
