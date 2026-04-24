FROM python:3.10-slim

WORKDIR /app

# Copy repo contents
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    openai \
    httpx \
    "gradio==6.13.0" \
    openenv-core

# Install the package (no-deps since deps already installed above)
RUN pip install --no-cache-dir --no-deps -e /app/

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "asha_sahayak.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
