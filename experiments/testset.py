"""Generate the QA benchmark with the judge model.

- single-hop: answered by one indexed chunk
- multi-hop: needs two chunks from different sections of the same paper
- unanswerable: written from held-out papers that were never indexed, so the right behaviour is to refuse
"""

import argparse
import json
import random
import re
from collections import defaultdict

from openai import OpenAI

from rag.config import DATA_DIR, INDEX_DIR, JUDGE_API_KEY, JUDGE_MODEL, JUDGE_URL
from rag.ingest import chunk, parse

teacher = OpenAI(base_url=JUDGE_URL, api_key=JUDGE_API_KEY, timeout=300, max_retries=2)

QGEN = """Below {what} from the research paper "{title}".
Write one question a researcher might ask that {need}, plus the answer.
Rules: the question must be specific and self-contained - name the method, model or dataset.
Never refer to "this paper", "the passage", "the authors" or "this study". Answer in one or two sentences.
Return JSON: {{"question": "...", "answer": "..."}}

{passages}"""
VAGUE = re.compile(r"\b(this|the) (paper|passages?|study|work|section|table|figure)\b|\bthe authors\b", re.I)


def make_question(*chunks):
    bodies = [c["text"].split("\n", 1)[-1] for c in chunks]  # drop the "title | section" prefix
    one = len(chunks) == 1
    text = QGEN.format(
        what="is a passage" if one else "are two passages",
        title=chunks[0]["title"],
        need="this passage answers" if one else "can only be answered by combining information from BOTH passages",
        passages="\n\n".join(f"Passage {i}:\n{b}" for i, b in enumerate(bodies, 1)),
    )
    try:
        r = teacher.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": text}],
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
        "paper": chunks[0]["paper"],
        "chunk_ids": [c.get("id") for c in chunks],
        "evidence": bodies,
    }


def multi_hop_pairs(chunks, n, rng):
    by_paper = defaultdict(list)
    for c in chunks:
        by_paper[c["paper"]].append(c)
    papers = [cs for cs in by_paper.values() if len({c["section"] for c in cs}) > 1]
    pairs = (rng.sample(rng.choice(papers), 2) for _ in range(n * 20))
    return [p for p in pairs if p[0]["section"] != p[1]["section"]][:n]


def main(n_single, n_multi, n_unanswerable):
    rng = random.Random(0)
    with open(INDEX_DIR / "chunks.jsonl") as f:
        indexed = [c for line in f if len((c := json.loads(line))["text"].split()) >= 150]
    heldout = [
        {"paper": p["id"], "title": p["title"], **c}
        for p in json.loads((DATA_DIR / "papers.json").read_text())
        if p["heldout"]
        for c in chunk(parse(p["id"]), p["title"])
        if len(c["text"].split()) >= 150
    ]
    single = lambda chunks, n: [(c,) for c in rng.sample(chunks, min(n, len(chunks)))]
    pools = [  # oversample 2x: some generations get filtered out
        ("single", single(indexed, n_single * 2), n_single, True),
        ("multi", multi_hop_pairs(indexed, n_multi * 2, rng), n_multi, True),
        ("unanswerable", single(heldout, n_unanswerable * 2), n_unanswerable, False),
    ]

    out = []
    for kind, pool, n, answerable in pools:
        made = 0
        for chunks in pool:
            if made == n:
                break
            if qa := make_question(*chunks):
                out.append({**qa, "answerable": answerable})
                made += 1
                print(f"[{kind} {made}/{n}] {qa['question']}")

    with open(DATA_DIR / "testset.jsonl", "w") as f:
        f.writelines(json.dumps(q) + "\n" for q in out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--single", type=int, default=200)
    ap.add_argument("--multi-hop", type=int, default=100)
    ap.add_argument("--unanswerable", type=int, default=100)
    args = ap.parse_args()
    main(args.single, args.multi_hop, args.unanswerable)
