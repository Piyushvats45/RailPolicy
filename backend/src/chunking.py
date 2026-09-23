import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/

def faq_pairs_as_chunks(faqs: list[dict]) -> list[dict]:
    """Each Q&A pair is already a self-contained, single-topic unit."""
    chunks = []
    for i, item in enumerate(faqs):
        text = f"Q: {item['question']}\nA: {item['answer']}"
        chunks.append({
            "id": f"chunk_{i}",
            "text": text,
            "source_url": item.get("source_url", "unknown"),
        })
    return chunks


def sliding_window_chunks(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """General-purpose chunker for long, unstructured text (PDFs, articles)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


if __name__ == "__main__":
    with open(os.path.join(BASE_DIR, "data", "faqs_raw.json"), encoding="utf-8") as f:
        faqs = json.load(f)

    chunks = faq_pairs_as_chunks(faqs)
    with open(os.path.join(BASE_DIR, "data", "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Created {len(chunks)} chunks from {len(faqs)} FAQ entries.")
