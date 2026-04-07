FROM python:3.10-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic openai httpx "gradio==6.10.0" openenv-core

# Copy repo contents into /app/asha_sahayak/ so 'import asha_sahayak' resolves
COPY . /app/asha_sahayak/

# Install the package
RUN pip install --no-cache-dir -e /app/asha_sahayak/

# Ensure asha_sahayak is importable regardless of working directory
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "asha_sahayak.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
