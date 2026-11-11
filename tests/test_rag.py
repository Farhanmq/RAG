from fastapi.testclient import TestClient

from rag import api
from rag import generate as gen
from rag.config import CHUNK_TOKENS
from rag.ingest import chunk, split_tokens, tokenizer
from rag.retrieve import rrf

BODY = " ".join(f"w{i}" for i in range(400))


def test_split_tokens_is_bounded_and_overlapping():
    pieces = split_tokens(BODY)
    lengths = [len(tokenizer()(p, add_special_tokens=False).input_ids) for p in pieces]
    assert len(pieces) > 1 and max(lengths) <= CHUNK_TOKENS
    assert pieces[0][-30:] in pieces[1]


def test_chunk_drops_references_and_adds_context():
    doc = f"# Title\n## 1 Introduction\n{BODY}\n## References\n[1] Some cited paper " + "x " * 100
    chunks = chunk(doc, "My Paper")
    assert chunks and all(c["section"] == "1 Introduction" for c in chunks)
    assert chunks[0]["text"].startswith("My Paper | 1 Introduction\n")
    assert not any("cited" in c["text"] for c in chunks)


def test_rrf_rewards_agreement():
    assert set(rrf([1, 2, 3], [3, 2, 4])[:2]) == {2, 3}  # found by both retrievers -> ranked first


def test_citations():
    assert gen.citations("A [1]. B [2, 3]. C [10].") == {1, 2, 3, 10}


def test_refuses_on_low_rerank_score_without_calling_llm():
    chunks = [{"text": "irrelevant", "score": 0.0}]
    assert gen.answer("what is x?", chunks) == gen.REFUSAL


class FakeRetriever:
    chunks = [{"paper": "1234.5678", "title": "T", "section": "S", "text": "T | S\nfoo", "score": 0.9}]

    def search(self, query, k=5):
        return self.chunks[:k]


def test_api(monkeypatch):
    api.state["retriever"] = FakeRetriever()
    monkeypatch.setattr(gen, "generate", lambda *args: iter(["Foo ", "is bar [1]."]))
    client = TestClient(api.app)

    r = client.post("/ask", json={"question": "what is foo?"})
    assert r.status_code == 200
    assert r.json()["answer"] == "Foo is bar [1]."
    assert r.json()["sources"][0]["cited"]

    r = client.post("/ask/stream", json={"question": "what is foo?"})
    assert "event: sources" in r.text and '"is bar [1]."' in r.text and "event: done" in r.text

    assert client.post("/ask", json={"question": ""}).status_code == 422
