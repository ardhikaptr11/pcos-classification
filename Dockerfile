FROM python:3.12-slim

# Update package lists and upgrade installed packages to patch CVEs
RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Instal uv globally in order to be able to run uv run / uv sync
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Copy dependencies first to take advantage of Docker Layer Caching
COPY pyproject.toml uv.lock ./

# Install only production dependencies
RUN uv sync --no-dev --frozen

# Copy remaining application code
COPY . .


ENV MODEL_PATH="model_artifacts/hub/model/MLmodel"
EXPOSE 8000
CMD ["uv", "run", "python", "main.py", "deploy", "serve"]