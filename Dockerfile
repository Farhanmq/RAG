FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/cache

# install dependencies first so code changes don't invalidate this layer; CPU-only torch keeps the image small
COPY pyproject.toml .
RUN mkdir rag && touch rag/__init__.py \
    && pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu . \
    && useradd -m -u 1000 app && mkdir /cache && chown app /cache
COPY rag rag
COPY data/index data/index

USER app
EXPOSE 8000
CMD ["uvicorn", "rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
