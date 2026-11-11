"""Generate a synthetic QA test set.

Answerable questions come from random indexed chunks; unanswerable ones come from held-out
papers that were never indexed, so the right behaviour for them is to refuse.
"""

import argparse
import json
import random
import re

from rag.config import DATA_DIR, INDEX_DIR, LLM_MODEL
from rag.generate import client
from rag.ingest import chunk, parse

QGEN = """Below is a passage from the research paper "{title}".
Write one question a researcher might ask that this passage answers, plus the answer.
Rules: the question must be specific and self-contained - name the method, model or dataset.
Never refer to "this paper", "the passage", "the authors" or "this study". Answer in one or two sentences.
Return JSON: {{"question": "...", "answer": "..."}}

Passage:
{text}"""
VAGUE = re.compile(r"\b(this|the) (paper|passage|study|work|section|table|figure)\b|\bthe authors\b", re.I)


def make_question(c):
    try:
        r = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": QGEN.format(title=c["title"], text=c["text"])}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        qa = json.loads(r.choices[0].message.content)
        q, a = qa["question"].strip(), qa["answer"].strip()
    except Exception:
        return None
    if VAGUE.search(q) or len(q.split()) < 6:
        return None
    return {
        "question": q,
        "answer": a,
        "paper": c["paper"],
        "chunk_id": c.get("id"),
        "evidence": c["text"].split("\n", 1)[-1],
    }


def sample(chunks, n, rng):
    chunks = [c for c in chunks if len(c["text"].split()) >= 150]
    return rng.sample(chunks, min(n, len(chunks)))


def main(n_answerable, n_unanswerable):
    rng = random.Random(0)
    with open(INDEX_DIR / "chunks.jsonl") as f:
        indexed = [json.loads(line) for line in f]
    heldout = [
        {"paper": p["id"], "title": p["title"], **c}
        for p in json.loads((DATA_DIR / "papers.json").read_text())
        if p["heldout"]
        for c in chunk(parse(p["id"]), p["title"])
    ]

    out = []
    for pool, n, answerable in [(indexed, n_answerable, True), (heldout, n_unanswerable, False)]:
        made = 0
        for c in sample(pool, n * 2, rng):  # oversample: some generations get filtered out
            if made == n:
                break
            if qa := make_question(c):
                out.append({**qa, "answerable": answerable})
                made += 1
                print(f"[{'A' if answerable else 'U'} {made}/{n}] {qa['question']}")

    with open(DATA_DIR / "testset.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable", type=int, default=200)
    ap.add_argument("--unanswerable", type=int, default=50)
    args = ap.parse_args()
    main(args.answerable, args.unanswerable)
