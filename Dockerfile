FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_URL=http://localhost:11434

WORKDIR /app

# Install system dependencies, curl, and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama inside container
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app

# Ensure start.sh has execute permissions
RUN chmod +x /app/start.sh

# Hugging Face Spaces routes traffic to port 7860
EXPOSE 7860

# Run orchestrator entrypoint
ENTRYPOINT ["/app/start.sh"]
