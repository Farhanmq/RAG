import json
from collections import defaultdict
from functools import cached_property

import bm25s
import faiss
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.config import CANDIDATES, EMBED_MODEL, INDEX_DIR, RERANK_MODEL, RERANK_TOP

QUERY_PROMPT = "Represent this sentence for searching relevant passages: "  # bge query instruction


def rrf(*rankings, k=60):
    """Reciprocal rank fusion: merge rankings without having to calibrate their scores."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking):
            scores[i] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


class Retriever:
    def __init__(self, index_dir=INDEX_DIR, embed_model=EMBED_MODEL):
        with open(index_dir / "chunks.jsonl") as f:
            self.chunks = [json.loads(line) for line in f]
        self.dense = faiss.read_index(str(index_dir / "dense.faiss"))
        self.bm25 = bm25s.BM25.load(index_dir / "bm25")
        self.embedder = SentenceTransformer(embed_model)

    @cached_property
    def reranker(self):
        return CrossEncoder(RERANK_MODEL)

    def dense_search(self, query, n):
        emb = self.embedder.encode([query], prompt=QUERY_PROMPT, normalize_embeddings=True)
        return self.dense.search(emb, n)[1][0].tolist()

    def bm25_search(self, query, n):
        tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
        return self.bm25.retrieve(tokens, k=n, show_progress=False)[0][0].tolist()

    def search(self, query, k=5, mode="hybrid", rerank=True):
        if mode == "dense":
            ids = self.dense_search(query, CANDIDATES)
        elif mode == "bm25":
            ids = self.bm25_search(query, CANDIDATES)
        else:
            ids = rrf(self.dense_search(query, CANDIDATES), self.bm25_search(query, CANDIDATES))

        scores = [None] * len(ids)
        if rerank:
            ids = ids[:RERANK_TOP]
            scores = self.reranker.predict([(query, self.chunks[i]["text"]) for i in ids]).tolist()
            scores, ids = zip(*sorted(zip(scores, ids), reverse=True))
        return [{**self.chunks[i], "score": s} for s, i in zip(scores[:k], ids[:k])]
