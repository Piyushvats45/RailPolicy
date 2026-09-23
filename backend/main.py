import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="IRCTC Policy Assistant API")

# Local dev origins are always allowed. Add your deployed frontend's URL via
# the FRONTEND_URL env var (set this in Render's dashboard) so the deployed
# React app is allowed to call this API too.
default_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
extra_origin = os.environ.get("FRONTEND_URL")
allowed_origins = default_origins + ([extra_origin] if extra_origin else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag: RAGPipeline | None = None


@app.on_event("startup")
def load_pipeline():
    """
    Load the embedding model + FAISS index once, when the server starts.
    On a fresh cloud deploy there's no pre-built index yet, so build it
    automatically here rather than requiring a manual build step.
    """
    global rag
    index_path = os.path.join(BASE_DIR, "data", "faq_index.faiss")
    meta_path = os.path.join(BASE_DIR, "data", "faq_meta.pkl")

    if not os.path.exists(index_path):
        print("No index found -- building it now (first boot on this machine)...")
        from src.chunking import faq_pairs_as_chunks
        from src.build_index import build_index
        import json

        with open(os.path.join(BASE_DIR, "data", "faqs_raw.json"), encoding="utf-8") as f:
            faqs = json.load(f)
        chunks = faq_pairs_as_chunks(faqs)
        chunks_path = os.path.join(BASE_DIR, "data", "chunks.json")
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        build_index(chunks_path=chunks_path, index_out_path=index_path, meta_out_path=meta_path)

    rag = RAGPipeline(index_path=index_path, meta_path=meta_path)
    print("RAG pipeline loaded.")


class AskRequest(BaseModel):
    question: str
    k: int = 3


class SourceOut(BaseModel):
    id: str
    text: str
    similarity_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceOut]


@app.get("/api/health")
def health():
    return {"status": "ok", "pipeline_loaded": rag is not None}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet.")
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = rag.answer(payload.question, k=payload.k)
    sources = [
        SourceOut(id=c["id"], text=c["text"], similarity_score=c["similarity_score"])
        for c in result["retrieved_chunks"]
    ]
    return AskResponse(answer=result["answer"], sources=sources)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
