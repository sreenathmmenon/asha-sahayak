FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai httpx gradio

# Copy application code
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

EXPOSE 7860

CMD ["uvicorn", "asha_sahayak.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
