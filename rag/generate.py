import re

from openai import OpenAI

from rag.config import LLM_API_KEY, LLM_MODEL, LLM_URL, MIN_RERANK_SCORE

REFUSAL = "I don't know based on the available papers."
SYSTEM = f"""You answer questions about research papers using only the numbered sources given.
Cite the supporting source after every claim, like [2]. Do not use outside knowledge.
If the sources do not contain the answer, reply exactly: "{REFUSAL}"
Be concise: at most a short paragraph."""

client = OpenAI(base_url=LLM_URL, api_key=LLM_API_KEY, timeout=120, max_retries=2)


def prompt(question, chunks):
    sources = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    return f"Sources:\n{sources}\n\nQuestion: {question}"


def generate(question, chunks, model=LLM_MODEL, min_score=MIN_RERANK_SCORE):
    """Stream the answer token by token. Refuses without calling the LLM when retrieval found nothing relevant."""
    if not chunks or (chunks[0]["score"] is not None and chunks[0]["score"] < min_score):
        yield REFUSAL
        return
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(question, chunks)}],
        temperature=0,
        stream=True,
    )
    for part in stream:
        if part.choices and (delta := part.choices[0].delta.content):
            yield delta


def answer(question, chunks, **kwargs):
    return "".join(generate(question, chunks, **kwargs)).strip()


def citations(text):
    return {int(n) for group in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", group)}
