---
title: RAG Knowledge Assistant
emoji: 📚
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
---

# RAG Knowledge Assistant

Question answering over 138 arXiv papers on LLMs and retrieval-augmented generation, built to run fully locally
on a laptop (Apple M4, 16 GB). Every design choice is backed by an ablation on a 250-question benchmark, including
questions the system *should* refuse.

```
question ─┬─ dense (bge-base + FAISS) ─┐
          └─ BM25 (bm25s) ─────────────┴─ RRF fusion ─ cross-encoder rerank ─┬─ low score → refuse
                                                                             └─ Qwen3-4B (Ollama) → cited answer
```

## Results

Benchmark: 200 answerable questions generated from indexed passages and 50 unanswerable ones from 12 held-out
papers ([experiments/testset.py](experiments/testset.py)). Full numbers are in [results/](results/).

**Retrieval** (200 answerable questions):

| config | Recall@1 | Recall@5 | MRR@10 |
|---|---|---|---|
| fixed-size chunks + dense (baseline) | 0.325 | 0.605 | 0.439 |
| structured chunks + dense | 0.395 | 0.680 | 0.517 |
| structured + BM25 | 0.575 | 0.825 | 0.683 |
| structured + hybrid (RRF) | 0.450 | 0.810 | 0.598 |
| **structured + hybrid + rerank** | **0.645** | **0.880** | **0.747** |
| fine-tuned embedder + dense | 0.455 | 0.765 | 0.589 |
| fine-tuned embedder + hybrid + rerank | 0.640 | 0.880 | 0.742 |

**End to end** (75 answerable + 25 unanswerable questions, Qwen3-4B as generator and judge):

| | baseline (fixed + dense) | full system |
|---|---|---|
| answer accuracy | 0.640 | **0.840** |
| faithfulness (claims supported by sources) | 0.891 | **0.973** |
| valid citations | 0.906 | **0.960** |
| correct refusal on unanswerable | 0.920 | **0.960** |
| false refusal on answerable | 0.173 | **0.013** |
| median latency | 11.2 s | 11.5 s |

The baseline is the classic tutorial setup: fixed-size chunks, dense retrieval only, no reranking. Better retrieval
mostly shows up as *fewer false refusals*: when the right passage is in the context, the model stops saying "I don't
know". Latency is dominated by generation on a memory-constrained laptop, so the reranker's extra second barely shows.

Failure analysis ([results/generation_fine-tuned.jsonl](results/generation_fine-tuned.jsonl)): the one unanswerable
question the system answered ("What is the method used to generate text embeddings in the knowledge base retrieval
system?") is generic enough that other indexed papers really do answer it. That's a test-set error, not a
hallucination. Most remaining wrong answers come from underspecified questions that match several papers.

**Reranker trade-off** (Recall@5 vs latency per query on an M4 GPU, [results/reranker_sweep.txt](results/reranker_sweep.txt)):

| reranker | top-10 | top-20 | top-30 |
|---|---|---|---|
| bge-reranker-base (278M) | 0.885 / 659 ms | **0.925 / 1352 ms** | 0.925 / 2014 ms |
| ms-marco-MiniLM-L6 (22M) | 0.875 / 76 ms | 0.895 / 156 ms | 0.900 / 263 ms |
| mxbai-rerank-xsmall (70M) | 0.880 / 911 ms | 0.880 / 2311 ms | 0.880 / 3606 ms |

The sweep ran before the switch to token-based chunking, hence the higher absolute recall. The default reranks the
top 20: same recall as 30 at two-thirds of the cost. For latency-sensitive use, `RAG_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2`
gives 8× faster reranking for 3 points of recall.

### What the numbers say

- **Reranking is the biggest single win** (+7 points Recall@5 and +20 points Recall@1 over hybrid alone).
- **Structure-aware chunking helps on its own** (+7.5 points Recall@5 with the same dense retriever).
- **BM25 beats dense retrieval here.** Part of that is real (papers are full of exact names like "RefoMed" or
  "BIOMISTRAL") and part is a known bias of synthetic benchmarks: generated questions reuse the passage's wording.
  Hybrid trails BM25 alone because the dense side is weak, but it is the better input to the reranker.
- **Fine-tuning the embedder works, then disappears behind the reranker.** 704 synthetic query–passage pairs with
  mined hard negatives (5 minutes on a laptop GPU) raise dense Recall@5 from 0.680 to 0.765. Once the cross-encoder
  rescores the candidates, both embedders end up at 0.880. The fine-tuned model is still the better choice when a
  reranker's latency is unaffordable.

## What's in here

| file | what it does |
|---|---|
| [rag/ingest.py](rag/ingest.py) | downloads papers, parses PDFs to markdown, structure-aware chunking, builds FAISS + BM25 indexes |
| [rag/retrieve.py](rag/retrieve.py) | dense, BM25 and hybrid (reciprocal rank fusion) retrieval with cross-encoder reranking |
| [rag/generate.py](rag/generate.py) | grounded generation with inline citations; refuses when retrieval finds nothing relevant |
| [rag/api.py](rag/api.py) | FastAPI service: `/ask`, streaming `/ask/stream` (SSE), `/health`, and a web UI at `/` ([rag/index.html](rag/index.html)) |
| [experiments/testset.py](experiments/testset.py) | synthetic benchmark: answerable questions from indexed chunks, unanswerable ones from held-out papers |
| [experiments/evaluate.py](experiments/evaluate.py) | retrieval ablations + LLM-as-judge generation metrics |
| [experiments/finetune.py](experiments/finetune.py) | fine-tunes the embedder on synthetic queries with mined hard negatives |

## Design decisions

- **Structure-aware chunking.** PDFs are converted to markdown (pymupdf4llm), split on section headings, and the
  reference list is dropped. Each chunk is prefixed with `paper title | section`, so a chunk that says "our method"
  still carries what the method is. Both schemes cut on embedder tokens; the baseline is a plain sliding window over
  the raw PDF text.
- **Hybrid retrieval with RRF.** Dense and BM25 fail on different queries (paraphrase vs exact names like
  "BIOMISTRAL" or "RefoMed"). Reciprocal rank fusion merges them without calibrating their scores.
- **Reranking.** A cross-encoder rescores the top 20 fused candidates; it is the single biggest quality gain.
- **Abstention.** If the best reranker score is below a threshold, the system answers "I don't know" without calling
  the LLM at all. That is cheaper and more reliable than hoping a 4B model notices on its own.
- **Honest unanswerable questions.** Every 12th paper is downloaded but never indexed. Questions generated from those
  papers look exactly like real ones, so they test abstention properly.
- **Relevance by text overlap, not chunk id.** A retrieved chunk is a hit if it is from the gold paper and covers at
  least half of the gold passage. This keeps metrics comparable across chunking schemes.
- **No framework.** Retrieval and generation are ~100 lines of plain Python; LangChain would add indirection without
  adding capability here.

## Run it

Requires Python 3.11–3.13 and [Ollama](https://ollama.com).

```bash
python3.13 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"
```

```bash
ollama pull qwen3:4b-instruct
```

```bash
python -m rag.ingest download --n 150 && python -m rag.ingest index && python -m rag.ingest index --fixed --out data/index_fixed
```

```bash
uvicorn rag.api:app --port 8000
```

```bash
curl -s localhost:8000/ask -H 'content-type: application/json' -d '{"question": "How does Self-RAG decide when to retrieve?"}'
```

Or with Docker (Ollama stays on the host, since Docker on macOS can't use the GPU):

```bash
docker compose up --build
```

Reproduce the evaluation:

```bash
python -m experiments.testset && python -m experiments.finetune
```

```bash
RAG_EMBED_MODEL=models/bge-base-rag python -m rag.ingest index --out data/index_ft && python -m experiments.evaluate
```

Every setting can be overridden with environment variables (`RAG_LLM_MODEL`, `RAG_LLM_URL`, `RAG_LLM_API_KEY`,
`RAG_EMBED_MODEL`, `RAG_RERANK_MODEL`, ...), so the same code points at vLLM or a hosted OpenAI-compatible API.

## Deploy to Hugging Face Spaces

The Docker image runs as-is on a free CPU Space, with the index shipped inside it and the LLM called through
Hugging Face's OpenAI-compatible router (a free CPU can't run a 4B model at a usable speed).

```bash
hf repos create <user>/rag --repo-type space --space-sdk docker
```

```bash
hf upload <user>/rag . --repo-type space --include "rag/*" --include "data/index/**" --include Dockerfile --include pyproject.toml --include README.md
```

Then set these in the Space's settings:

| name | value | kind |
|---|---|---|
| `RAG_LLM_URL` | `https://router.huggingface.co/v1` | variable |
| `RAG_LLM_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | variable |
| `RAG_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` (8× faster on CPU, see the sweep above) | variable |
| `RAG_LLM_API_KEY` | a Hugging Face token with the "Inference Providers" permission | secret |

The evaluation numbers above are for the local setup (Qwen3-4B, bge reranker), not the hosted one.

## Engineering notes

Problems found while building this, and the fixes:

- **Silent truncation.** Chunking by word count let results tables (`|0.53|0.61|...` counts as one "word") grow past
  the embedder's 512-token limit, so their embeddings ignored the tail, and five of them could overflow the LLM's
  4k context. Chunks are now cut on the embedder's own tokens (320 + 64 overlap); the worst-case prompt is 2.5k tokens.
- **Broken font encodings.** One paper's Arabic text extracted as `�` characters that cost 4× the tokens of normal
  text. These are stripped at parse time.
- **Thinking models.** Qwen3's hybrid-thinking variant ignores Ollama's "no thinking" switches and reasons anyway,
  adding seconds per answer; the non-thinking instruct variant is used instead.
- **Memory on a 16 GB machine.** Training with eager attention used 14 GB; disabling only attention dropout keeps
  PyTorch's fused attention usable on Apple GPUs. The evaluator frees retrievers it no longer needs before generation.

## Limitations

- **Synthetic, LLM-judged benchmark.** Questions are written and graded by the same 4B model that answers them; a
  stronger judge (`RAG_JUDGE_MODEL`) would give more trustworthy generation scores. I spot-checked questions by hand,
  and some are underspecified (e.g. "the 'Meaning' intent category").
- **Gold passages come from an earlier chunking.** The test set was generated before the switch to token-based
  chunks. Relevance is judged by text overlap, so the comparison stays fair across configs, but absolute recall is
  a few points lower than with aligned chunks.
- **Abstention threshold is set a priori** (reranker score 0.05), not tuned on held-out data.
- **No multi-hop questions.** Every question is answerable from a single passage.
