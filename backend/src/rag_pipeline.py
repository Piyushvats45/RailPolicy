import os
import pickle
import numpy as np
from fastembed import TextEmbedding
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# LLM_PROVIDER controls where generation happens:
#   "ollama" (default) -> free, local, runs on your own machine. Use for local dev.
#   "groq"             -> free-tier hosted API. Use this in cloud deployments
#                         (Render/Railway free tiers can't run Ollama -- not
#                         enough RAM/CPU to host a model server).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = "llama3.2:3b"
GROQ_MODEL = "openai/gpt-oss-20b"   # free tier on Groq as of writing

if LLM_PROVIDER == "groq":
    from groq import Groq  # pip install groq
else:
    import ollama  # pip install ollama -- talks to local Ollama server, no API key


class RAGPipeline:
    def __init__(self, index_path: str, meta_path: str):
        self.embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
        self.index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)

        if LLM_PROVIDER == "groq":
            self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        query_vec = np.array(list(self.embed_model.embed([query]))).astype("float32")
        query_vec = query_vec / np.linalg.norm(query_vec, axis=1, keepdims=True)

        scores, indices = self.index.search(query_vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({**chunk, "similarity_score": float(score)})
        return results

    def generate_answer(self, query: str, retrieved_chunks: list[dict]) -> str:
        context = "\n\n".join(
            f"[Source {i+1}]\n{c['text']}" for i, c in enumerate(retrieved_chunks)
        )

        system_prompt = (
            "You are a customer support assistant for IRCTC ticket cancellation "
            "and refund questions. Answer ONLY using the provided sources below. "
            "If the sources don't contain the answer, say clearly that you don't "
            "have that information rather than guessing. When you answer, mention "
            "which Source number(s) you used, e.g. '(Source 2)'."
        )
        user_message = f"Sources:\n{context}\n\nQuestion: {query}"

        if LLM_PROVIDER == "groq":
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=500,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response.choices[0].message.content
        else:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
            )
            return response["message"]["content"]

    def answer(self, query: str, k: int = 3) -> dict:
        retrieved = self.retrieve(query, k=k)
        answer_text = self.generate_answer(query, retrieved)
        return {
            "query": query,
            "answer": answer_text,
            "retrieved_chunks": retrieved,
        }


if __name__ == "__main__":
    rag = RAGPipeline(
        index_path=os.path.join(BASE_DIR, "data", "faq_index.faiss"),
        meta_path=os.path.join(BASE_DIR, "data", "faq_meta.pkl"),
    )

    test_questions = [
        "If I cancel my ticket 2 days before departure, how much do I lose?",
        "I have a Tatkal ticket and need to cancel, will I get money back?",
        "My train ticket booking payment failed, what should I do?",
    ]

    for q in test_questions:
        result = rag.answer(q)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print("Retrieved from:", [c["id"] for c in result["retrieved_chunks"]])
