import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag import generate as gen
from rag.retrieve import Retriever

state = {}


@asynccontextmanager
async def lifespan(app):
    state["retriever"] = Retriever()
    state["retriever"].reranker  # load now rather than on the first request
    yield


app = FastAPI(title="RAG Knowledge Assistant", lifespan=lifespan)


@app.exception_handler(openai.APIError)
def llm_unavailable(request, exc):
    return JSONResponse({"detail": f"LLM backend error: {exc}"}, status_code=503)


class Query(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(5, ge=1, le=20)


def sources(chunks, cited=()):
    return [
        {
            "n": n,
            "paper": c["paper"],
            "title": c["title"],
            "section": c["section"],
            "score": c["score"],
            "cited": n in cited,
        }
        for n, c in enumerate(chunks, 1)
    ]


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(Path(__file__).with_name("index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(state["retriever"].chunks)}


@app.post("/ask")
def ask(q: Query):
    t0 = time.perf_counter()
    chunks = state["retriever"].search(q.question, q.k)
    t1 = time.perf_counter()
    text = gen.answer(q.question, chunks)
    t2 = time.perf_counter()
    return {
        "answer": text,
        "sources": sources(chunks, gen.citations(text)),
        "latency_ms": {"retrieval": round((t1 - t0) * 1000), "generation": round((t2 - t1) * 1000)},
    }


@app.post("/ask/stream")
def ask_stream(q: Query):
    chunks = state["retriever"].search(q.question, q.k)

    def events():
        yield f"event: sources\ndata: {json.dumps(sources(chunks))}\n\n"
        try:
            for token in gen.generate(q.question, chunks):
                yield f"data: {json.dumps(token)}\n\n"
        except openai.APIError as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
