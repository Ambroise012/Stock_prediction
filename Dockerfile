FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY /src /app/src
COPY /models/ /app/models

# Streamlit port
EXPOSE 8501

CMD ["streamlit", "run", "src/interface.py", "--server.address=0.0.0.0", "--server.port=8501"]
