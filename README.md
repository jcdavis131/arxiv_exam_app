# arxiv-exam-app

FastAPI micro-service that converts any arXiv paper into an on-demand multiple-choice exam.

## Features

- **Interactive Web UI**: A clean, minimalist frontend to generate and interact with exams.
- **Full PDF Content Analysis**: Extracts and processes text from the full arXiv PDF (foundational for RAG).
- **Downloadable Exams**: Download generated exams, including sample answers, as JSON.
- **Async & type-safe**: FastAPI, httpx, Pydantic v2.
- **Pluggable LLM backend**: OpenAI or local Hugging Face Text-Generation-Inference.
- **LRU-cached** arXiv metadata for speed.
- **12-factor ready**: configuration via environment variables, stateless, Docker-friendly.
- **CI**: Ruff lint + pytest via GitHub Actions.

## Quick start

```bash
pip install .[dev]        # install project + dev extras
export OPENAI_API_KEY=sk-... # if using the OpenAI backend
uvicorn arxiv_exam_app:app --reload
# open http://127.0.0.1:8000 to access the web UI
# or http://127.0.0.1:8000/docs for API docs
```

**Local model**

```bash
export LLM_BACKEND=hf
export HF_API_URL=http://localhost:8080/v1/chat/completions # TGI endpoint
uvicorn arxiv_exam_app:app --reload
```

## Docker

```bash
docker build -t arxiv-exam-app .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... arxiv-exam-app
# Access the web UI at http://127.0.0.1:8000
```

## CI status

GitHub Actions workflow in `.github/workflows/ci.yml` lints with Ruff and runs tests on Python 3.11.

## Design Aesthetic

The web UI is crafted with a "Scandinavian+Japanese+drafting board" aesthetic, focusing on clean lines, natural tones, and a minimalist, functional layout.

## License

MIT

**Contact**

For any inquiries or issues, please visit our GitHub repository: [https://github.com/jcdavis131/arxiv_exam_app](https://github.com/jcdavis131/arxiv_exam_app)
