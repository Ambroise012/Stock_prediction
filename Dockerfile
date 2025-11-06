FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* /app/

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

COPY /src /app/src
COPY .env /app/.env


# Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "src/interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
