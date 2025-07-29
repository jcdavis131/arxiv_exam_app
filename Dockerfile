FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=0

RUN addgroup --system app && adduser --system --group app
WORKDIR /app

COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip install .

COPY . .
USER app

EXPOSE 8000
CMD ["uvicorn", "arxiv_exam_app:app", "--host", "0.0.0.0", "--port", "8000"]