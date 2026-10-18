FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

COPY pyproject.toml uv.lock /app/

RUN uv sync --locked --no-dev --no-install-project

COPY /src /app/src
COPY .env /app/.env


# Streamlit port
EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "src/interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
