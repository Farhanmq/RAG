import os
from pathlib import Path

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "data"))
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-base")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen3:4b-instruct")
LLM_URL = os.getenv("RAG_LLM_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("RAG_LLM_API_KEY", "unused")  # Ollama ignores it; hosted APIs need it

CHUNK_TOKENS = 320  # embedder tokens; leaves room for the title/section prefix within bge's 512 limit
CHUNK_OVERLAP = 64
CANDIDATES = 30  # per retriever, before fusion
RERANK_TOP = 20  # fused candidates sent to the cross-encoder; 30 gave no extra recall for 50% more latency
MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.05"))  # below this we refuse to answer
