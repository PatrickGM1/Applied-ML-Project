FROM python:3.11-slim

WORKDIR /app

# Install only API dependencies (no streamlit) for a smaller image
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas scikit-learn scipy joblib nltk \
    fastapi "uvicorn[standard]" pydantic httpx

# Copy source (dockerignore keeps this fast)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
