FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai httpx gradio openenv-core

# The HF Space repo root contains all files from asha_sahayak/ directly.
# We copy them into a subdirectory so 'import asha_sahayak' works.
COPY . /app/asha_sahayak/

# Install the package in editable mode
RUN pip install --no-cache-dir -e /app/asha_sahayak/

EXPOSE 7860

CMD ["uvicorn", "asha_sahayak.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
