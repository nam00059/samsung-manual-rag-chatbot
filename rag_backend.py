# rag_backend.py  -----------------------------------------------------
"""
Local RAG backend: MiniLM embeddings + FAISS retrieval + TinyLlama-1.1B
All heavy objects load once and stay cached by the import system.
"""

import pathlib, pickle, faiss, numpy as np
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# ------------------------------------------------------------------ #
# 1. Vector store (artefacts saved from your notebook)               #
# ------------------------------------------------------------------ #
DATA_DIR = pathlib.Path(__file__).parent / "model_assets"   # ← folder with *.pkl / .faiss
paragraphs: list[str] = pickle.load(open(DATA_DIR / "cleaned_paragraphs.pkl", "rb"))
index                  = faiss.read_index(str(DATA_DIR / "faiss_index.faiss"))

# ------------------------------------------------------------------ #
# 2. Embedding model (same as build-time)                            #
# ------------------------------------------------------------------ #
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # loads once

# ------------------------------------------------------------------ #
# 3. TinyLlama-1.1B (4-bit, CPU-only)                                #
# ------------------------------------------------------------------ #
GGUF = hf_hub_download(
    "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    "tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
)
llm = Llama(model_path=GGUF, n_ctx=2048, n_threads=8, n_batch=512)

# ------------------------------------------------------------------ #
# 4. Retrieval helper                                                #
# ------------------------------------------------------------------ #
def _retrieve(query: str, k: int = 5):
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, I  = index.search(q_vec, k)
    return [{"chunk_id": int(i), "text": paragraphs[i]} for i in I[0]]

# ------------------------------------------------------------------ #
# 5. Build prompt & generate answer                                  #
# ------------------------------------------------------------------ #
def _prompt(q: str, ctx_chunks):
    ctx = "\n\n---\n\n".join(f"[#{c['chunk_id']}] {c['text']}" for c in ctx_chunks)
    return (
        "You are a Samsung product-manual assistant. "
        "Use ONLY the context to answer; if not answerable, say you don't know. "
        "Show the chunk id(s) at the end.\n\n"
        f"### Context\n{ctx}\n\n### Question\n{q}\n\n### Answer:"
    )

def _ask_llama(prompt: str, max_new: int = 128) -> str:
    out = llm(prompt, max_tokens=max_new, stop=["### Question"])
    return out["choices"][0]["text"].strip()

# ------------------------------------------------------------------ #
# 6. Public API                                                      #
# ------------------------------------------------------------------ #
def rag_answer(question: str, k: int = 5) -> str:
    ctx = _retrieve(question, k)
    pr  = _prompt(question, ctx)
    return _ask_llama(pr)
