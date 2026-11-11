"""Retrieval ablations plus end-to-end generation metrics (LLM-as-judge), written to results/.

A third of the benchmark is a dev split, used only to tune the abstention threshold; everything reported is on the rest.
"""

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

from experiments.testset import teacher
from rag.config import DATA_DIR, EMBED_MODEL, INDEX_DIR, JUDGE_MODEL
from rag.generate import REFUSAL, answer, citations, prompt
from rag.retrieve import Retriever

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

mean = lambda xs: sum(xs) / max(len(xs), 1)


def found_at(chunks, q):
    """Rank at which every gold passage has been retrieved, or None. A chunk covers a gold passage if it is from the
    gold paper and contains at least half its words; text overlap (not chunk ids) keeps chunking schemes comparable."""
    missing = [set(e.split()) for e in q["evidence"]]
    for rank, c in enumerate(chunks, 1):
        words = set(c["text"].split())
        missing = [g for g in missing if c["paper"] != q["paper"] or len(g & words) < len(g) / 2]
        if not missing:
            return rank
    return None


def judge(text):
    r = teacher.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": text}], temperature=0)
    return r.choices[0].message.content.strip().lower().startswith("yes")


def eval_retrieval(retriever, qs, mode, rerank):
    ranks = [found_at(retriever.search(q["question"], 10, mode, rerank), q) or 99 for q in qs]
    return {
        "recall@1": mean([r <= 1 for r in ranks]),
        "recall@5": mean([r <= 5 for r in ranks]),
        "recall@10": mean([r <= 10 for r in ranks]),
        "mrr@10": mean([1 / r for r in ranks]),
    }


def tune_threshold(retriever, dev):
    """Abstention threshold on the top reranker score that best separates answerable from unanswerable dev questions
    (balanced accuracy, so wrongly refusing and wrongly answering cost the same)."""
    top = [(retriever.search(q["question"], 1)[0]["score"], q["answerable"]) for q in dev]
    balanced = lambda t: mean([s >= t for s, a in top if a]) + mean([s < t for s, a in top if not a])
    return max((s for s, _ in top), key=balanced)


def eval_generation(retriever, qs, mode, rerank, name, min_score):
    rows = []
    for q in qs:
        t = time.perf_counter()
        chunks = retriever.search(q["question"], 5, mode, rerank)
        ans = answer(q["question"], chunks, min_score=min_score)
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

    answerable = [r for r in rows if r["answerable"]]
    given = [r for r in rows if not r["refused"]]
    return {
        "accuracy": mean([r.get("correct", False) for r in answerable]),
        "accuracy_multi_hop": mean([r.get("correct", False) for r in answerable if len(r["evidence"]) > 1]),
        "faithfulness": mean([r["faithful"] for r in given]),
        "citation_validity": mean([r["citations_ok"] for r in given]),
        "abstention_on_unanswerable": mean([r["refused"] for r in rows if not r["answerable"]]),
        "false_refusal": mean([r["refused"] for r in answerable]),
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
    random.Random(0).shuffle(qs)
    dev, test = qs[: len(qs) // 3], qs[len(qs) // 3 :]
    single = [q for q in test if q["answerable"] and len(q["evidence"]) == 1]
    multi = [q for q in test if q["answerable"] and len(q["evidence"]) > 1]

    dirs = {d for d, _, _ in CONFIGS.values()}
    retrievers = {d: Retriever(d, str(FT_MODEL) if d == FT_INDEX else EMBED_MODEL) for d in dirs}
    retrieval = {name: eval_retrieval(retrievers[d], single, m, r) for name, (d, m, r) in CONFIGS.items()}
    multi_hop = {name: eval_retrieval(retrievers[d], multi, m, r) for name, (d, m, r) in CONFIGS.items()}
    print(table(retrieval), "\n\nmulti-hop:\n" + table(multi_hop), "\n")

    retrievers = {d: retrievers[d] for d in (CONFIGS[BASELINE][0], CONFIGS[SYSTEM][0])}  # free the rest
    threshold = tune_threshold(retrievers[CONFIGS[SYSTEM][0]], dev)
    print(f"abstention threshold tuned on {len(dev)} dev questions: {threshold:.4f}\n")

    generation = {}
    for name in [BASELINE, SYSTEM] if n_gen else []:
        d, m, r = CONFIGS[name]
        generation[name] = eval_generation(retrievers[d], test[:n_gen], m, r, name, threshold)
    if generation:
        print(table(generation))

    results = {"judge": JUDGE_MODEL, "threshold": threshold, "retrieval": retrieval, "retrieval_multi_hop": multi_hop}
    (RESULTS / "results.json").write_text(json.dumps({**results, "generation": generation}, indent=2))
    if retrieval[SYSTEM]["recall@5"] < min_recall:
        sys.exit(f"recall@5 {retrieval[SYSTEM]['recall@5']:.3f} is below the required {min_recall}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-gen", type=int, default=200, help="test questions for the (slow) generation eval; 0 to skip")
    ap.add_argument("--min-recall", type=float, default=0.0, help="fail if recall@5 drops below this")
    args = ap.parse_args()
    main(args.n_gen, args.min_recall)
