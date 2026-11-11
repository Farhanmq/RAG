"""Fine-tune the embedder on synthetic (question, passage) pairs from the indexed corpus, with mined hard negatives.

Chunks behind test questions (and their overlapping neighbours) are excluded, so the test set stays unseen.
"""

import argparse
import json
import random

from datasets import Dataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.util import mine_hard_negatives

from experiments.testset import make_question
from rag.config import DATA_DIR, EMBED_MODEL, INDEX_DIR
from rag.retrieve import QUERY_PROMPT

PAIRS = DATA_DIR / "train_pairs.jsonl"


def banned_ids():
    """Test chunks and their overlapping neighbours: never trained on, even if the test set was regenerated."""
    with open(DATA_DIR / "testset.jsonl") as f:
        test_ids = {i for line in f for i in json.loads(line)["chunk_ids"]} - {None}
    return {i + d for i in test_ids for d in (-1, 0, 1)}


def make_pairs(n):
    banned = banned_ids()
    with open(INDEX_DIR / "chunks.jsonl") as f:
        chunks = [c for line in f if (c := json.loads(line))["id"] not in banned and len(c["text"].split()) >= 100]

    with open(PAIRS, "w") as f:
        for i, c in enumerate(random.Random(1).sample(chunks, min(n, len(chunks)))):
            if qa := make_question(c):
                f.write(json.dumps({"anchor": qa["question"], "positive": c["text"], "chunk_id": c["id"]}) + "\n")
            print(f"\r{i + 1}/{n}", end="")


def train(out, epochs):
    banned = banned_ids()
    with open(PAIRS) as f:
        rows = [r for line in f if (r := json.loads(line)).pop("chunk_id") not in banned]
    data = Dataset.from_list(rows)
    # MPS fused attention can't do attention dropout; turning just that off keeps it (eager attention used 14 GB)
    model = SentenceTransformer(EMBED_MODEL, config_kwargs={"attention_probs_dropout_prob": 0.0})
    model.max_seq_length = 384  # chunks are ~250 words
    # hard negatives: passages ranked 10-50 for the question - similar, but (very likely) not the answer
    data = mine_hard_negatives(data, model, range_min=10, range_max=50, num_negatives=1, sampling_strategy="random")

    args = SentenceTransformerTrainingArguments(
        output_dir=out,
        num_train_epochs=epochs,
        per_device_train_batch_size=64,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # duplicates would act as false in-batch negatives
        prompts={"anchor": QUERY_PROMPT},
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)  # big effective batch on little memory
    SentenceTransformerTrainer(model=model, args=args, train_dataset=data, loss=loss).train()
    model.save(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1500, help="chunks to generate training questions for")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default="models/bge-base-rag")
    args = ap.parse_args()
    if not PAIRS.exists():
        make_pairs(args.pairs)
    train(args.out, args.epochs)
