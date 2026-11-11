"""Download arXiv papers, parse them, chunk them and build the dense + BM25 indexes."""

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import cache, partial
from pathlib import Path
from urllib.request import urlretrieve

import arxiv
import bm25s
import faiss
import pymupdf
import pymupdf4llm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from rag.config import CHUNK_OVERLAP, CHUNK_TOKENS, DATA_DIR, EMBED_MODEL, INDEX_DIR, PDF_DIR

QUERY = 'cat:cs.CL AND (abs:"retrieval augmented generation" OR abs:"large language models")'
HELDOUT_EVERY = 12  # every 12th paper is kept out of the index to test unanswerable questions


def download(n):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    papers = []
    for r in arxiv.Client(delay_seconds=3, num_retries=5).results(arxiv.Search(query=QUERY, max_results=n)):
        pid = r.get_short_id().split("v")[0]
        path = PDF_DIR / f"{pid}.pdf"
        try:
            if not path.exists():
                urlretrieve(r.pdf_url, path.with_suffix(".part"))  # rename after, so a crash never leaves a broken pdf
                path.with_suffix(".part").rename(path)
                time.sleep(1)  # be polite to arXiv
        except Exception as e:
            print(f"skipping {pid}: {e}")
            continue
        papers.append({"id": pid, "title": r.title, "heldout": len(papers) % HELDOUT_EVERY == HELDOUT_EVERY - 1})
        print(f"{len(papers)}/{n} {r.title[:80]}")
    (DATA_DIR / "papers.json").write_text(json.dumps(papers, indent=1))


def parse(pid, structured=True):
    path = str(PDF_DIR / f"{pid}.pdf")
    try:
        if structured:
            # papers are born-digital, so skip OCR, figure text and page headers/footers
            md = pymupdf4llm.to_markdown(path, use_ocr=False, force_text=False, header=False, footer=False)
            md = md.replace("�", "")  # glyphs from broken font encodings: pure noise that eats context
            return re.sub(r"<!--.*?-->|</?(?:br|mark|sup|sub|span|b|i|u)\b[^>]*>", " ", md)
        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"failed to parse {pid}: {e}")
        return ""


@cache
def tokenizer():
    return AutoTokenizer.from_pretrained(EMBED_MODEL)


def split_tokens(text):
    """Sliding window over the embedder's own tokens, so no chunk is silently truncated at embedding time
    (word counts are misleading: a results table tokenizes to ~3x its word count)."""
    spans = tokenizer()(text, return_offsets_mapping=True, add_special_tokens=False, verbose=False)["offset_mapping"]
    step = CHUNK_TOKENS - CHUNK_OVERLAP
    starts = range(0, max(len(spans) - CHUNK_OVERLAP, 1), step) if spans else []
    return [text[spans[i][0] : spans[min(i + CHUNK_TOKENS, len(spans)) - 1][1]] for i in starts]


def chunk(doc, title, structured=True):
    """Structured: split on markdown headings, drop references, prefix each chunk with title and section.
    Fixed: plain sliding window over the raw text (the baseline)."""
    if not structured:
        return [{"section": "", "text": t} for t in split_tokens(doc) if len(t) >= 150]
    doc = re.split(r"\n#+ [*_\d. ]*(?:references|bibliography)\b", doc, flags=re.I)[0]
    chunks, section = [], ""
    for block in re.split(r"\n(?=#{1,6} )", doc):
        if block.startswith("#"):
            heading, _, block = block.partition("\n")
            section = heading.strip("#*_ ")
        for t in split_tokens(block):
            if len(t) >= 150:
                chunks.append({"section": section, "text": f"{title} | {section}\n{t}"})
    return chunks


def build_index(structured=True, out=INDEX_DIR):
    papers = [p for p in json.loads((DATA_DIR / "papers.json").read_text()) if not p["heldout"]]
    with ProcessPoolExecutor() as pool:
        docs = list(pool.map(partial(parse, structured=structured), [p["id"] for p in papers]))

    chunks = []
    for p, doc in zip(papers, docs):
        for c in chunk(doc, p["title"], structured):
            chunks.append({"id": len(chunks), "paper": p["id"], "title": p["title"], **c})
    texts = [c["text"] for c in chunks]
    print(f"{len(papers)} papers -> {len(chunks)} chunks")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w") as f:
        f.writelines(json.dumps(c) + "\n" for c in chunks)

    emb = SentenceTransformer(EMBED_MODEL).encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=True
    )
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out / "dense.faiss"))

    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords="en"))
    bm25.save(out / "bm25")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["download", "index"])
    ap.add_argument("--n", type=int, default=150, help="papers to download")
    ap.add_argument("--fixed", action="store_true", help="fixed-size chunking baseline")
    ap.add_argument("--out", default=str(INDEX_DIR))
    args = ap.parse_args()
    if args.step == "download":
        download(args.n)
    else:
        build_index(structured=not args.fixed, out=Path(args.out))
