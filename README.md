# Applied Machine Learning project

## Team members:

    - Patrick Gheba
    - Luca Serban
    - Ana-Maria Izbas
    - George Tutui

## Project idea:

    Fake News Detection using the LIAR dataset.
    Dataset: https://www.kaggle.com/datasets/doanquanvietnamca/liar-dataset

## Website for deployment:

https://aml.guba.dev/

## Local testing:

### Docker (recommended)

Requires [Docker](https://docs.docker.com/get-docker/) (Desktop on Windows/macOS, Engine on Linux).

```bash
# Windows / macOS
docker compose up --build

# Linux (if your user is not in the docker group)
sudo docker compose up --build
```

| Service            | URL                        |
| ------------------ | -------------------------- |
| FastAPI backend    | http://localhost:8000      |
| Swagger UI         | http://localhost:8000/docs |
| Streamlit frontend | http://localhost:8502      |

Stop with `Ctrl+C`, then `docker compose down`.

---

### Without Docker

**Prerequisites:** Python 3.11+

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data (first run only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# 4. Start the API
uvicorn main:app --reload --port 8000

# 5. In a second terminal (with the venv activated), start Streamlit
streamlit run streamlit_app.py --server.port 8502
```

The `API_BASE_URL` environment variable controls which backend Streamlit calls (default: `http://localhost:8000`).

---
