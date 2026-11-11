"""Retrieval ablations plus end-to-end generation metrics (LLM-as-judge), written to results/."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from rag.config import DATA_DIR, EMBED_MODEL, INDEX_DIR
from rag.generate import REFUSAL, answer, citations, client, prompt
from rag.retrieve import Retriever

JUDGE_MODEL = os.getenv("RAG_JUDGE_MODEL", "qwen3:4b-instruct")
RESULTS = Path("results")
FT_MODEL, FT_INDEX = Path("models/bge-base-rag"), DATA_DIR / "index_ft"

# name -> (index dir, mode, rerank); the last one is the full system
CONFIGS = {
    "fixed + dense (baseline)": (DATA_DIR / "index_fixed", "dense", False),
    "structured + dense": (INDEX_DIR, "dense", False),
    "structured + bm25": (INDEX_DIR, "bm25", False),
    "structured + hybrid": (INDEX_DIR, "hybrid", False),
    "structured + hybrid + rerank": (INDEX_DIR, "hybrid", True),
}
if FT_INDEX.exists():
    CONFIGS["fine-tuned + dense"] = (FT_INDEX, "dense", False)
    CONFIGS["fine-tuned + hybrid + rerank"] = (FT_INDEX, "hybrid", True)
BASELINE, SYSTEM = next(iter(CONFIGS)), list(CONFIGS)[-1]

CORRECT = """Question: {question}
Reference answer: {reference}
Candidate answer: {answer}
Does the candidate answer convey the key information of the reference answer? Reply with only yes or no."""

FAITHFUL = """{context}

Answer to check: {answer}
Is every factual claim in this answer supported by the sources above? Reply with only yes or no."""


def is_hit(chunk, q):
    """A retrieved chunk counts as relevant if it comes from the gold paper and covers most of the gold passage.
    Text overlap (not chunk ids) keeps this comparable across different chunking schemes."""
    gold = set(q["evidence"].split())
    return chunk["paper"] == q["paper"] and len(gold & set(chunk["text"].split())) / len(gold) >= 0.5


def judge(text):
    r = client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": text}], temperature=0)
    return r.choices[0].message.content.strip().lower().startswith("yes")


def eval_retrieval(retriever, qs, mode, rerank):
    recall1 = recall5 = mrr = 0
    for q in qs:
        hits = [is_hit(c, q) for c in retriever.search(q["question"], 10, mode, rerank)]
        recall1 += hits[0]
        recall5 += any(hits[:5])
        mrr += next((1 / (rank + 1) for rank, h in enumerate(hits) if h), 0)
    n = len(qs)
    return {"recall@1": recall1 / n, "recall@5": recall5 / n, "mrr@10": mrr / n}


def eval_generation(retriever, qs, mode, rerank, name):
    rows = []
    for q in qs:
        t = time.perf_counter()
        chunks = retriever.search(q["question"], 5, mode, rerank)
        ans = answer(q["question"], chunks)
        row = {**q, "response": ans, "latency": time.perf_counter() - t, "refused": REFUSAL[:12] in ans}
        if not row["refused"]:
            cited = citations(ans)
            row["citations_ok"] = bool(cited) and cited <= set(range(1, len(chunks) + 1))
            row["faithful"] = judge(FAITHFUL.format(context=prompt(q["question"], chunks), answer=ans))
            if q["answerable"]:
                row["correct"] = judge(CORRECT.format(question=q["question"], reference=q["answer"], answer=ans))
        rows.append(row)

    with open(RESULTS / f"generation_{name.split()[0]}.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)

    ans_rows = [r for r in rows if r["answerable"]]
    una_rows = [r for r in rows if not r["answerable"]]
    given = [r for r in rows if not r["refused"]]
    mean = lambda xs: sum(xs) / max(len(xs), 1)
    return {
        "accuracy": mean([r.get("correct", False) for r in ans_rows]),
        "faithfulness": mean([r["faithful"] for r in given]),
        "citation_validity": mean([r["citations_ok"] for r in given]),
        "abstention_on_unanswerable": mean([r["refused"] for r in una_rows]),
        "false_refusal": mean([r["refused"] for r in ans_rows]),
        "latency_p50_s": statistics.median(r["latency"] for r in rows),
    }


def table(results):
    cols = list(next(iter(results.values())))
    lines = ["| config | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {name} | " + " | ".join(f"{v:.3f}" for v in m.values()) + " |" for name, m in results.items()]
    return "\n".join(lines)


def main(n_gen, min_recall):
    RESULTS.mkdir(exist_ok=True)
    with open(DATA_DIR / "testset.jsonl") as f:
        qs = [json.loads(line) for line in f]
    answerable = [q for q in qs if q["answerable"]]
    gen_qs = answerable[: n_gen * 3 // 4] + [q for q in qs if not q["answerable"]][: n_gen // 4]

    dirs = {d for d, _, _ in CONFIGS.values()}
    retrievers = {d: Retriever(d, str(FT_MODEL) if d == FT_INDEX else EMBED_MODEL) for d in dirs}
    retrieval = {name: eval_retrieval(retrievers[d], answerable, m, r) for name, (d, m, r) in CONFIGS.items()}
    print(table(retrieval), "\n")

    generation = {}
    retrievers = {d: retrievers[d] for d in (CONFIGS[BASELINE][0], CONFIGS[SYSTEM][0])}  # free the rest (16 GB box)
    if n_gen:
        for name in [BASELINE, SYSTEM]:
            d, m, r = CONFIGS[name]
            generation[name] = eval_generation(retrievers[d], gen_qs, m, r, name)
        print(table(generation))

    (RESULTS / "results.json").write_text(json.dumps({"retrieval": retrieval, "generation": generation}, indent=2))
    best = retrieval[SYSTEM]["recall@5"]
    if best < min_recall:
        sys.exit(f"recall@5 {best:.3f} is below the required {min_recall}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=100, help="questions for the (slow) generation eval; 0 to skip")
    ap.add_argument("--min-recall", type=float, default=0.0, help="fail if recall@5 drops below this")
    args = ap.parse_args()
    main(args.n_gen, args.min_recall)
