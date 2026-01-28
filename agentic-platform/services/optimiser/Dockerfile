FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY problems/ ./problems/
COPY agents/ ./agents/
COPY orchestration/ ./orchestration/
COPY evaluation/ ./evaluation/
COPY search_space/ ./search_space/
COPY main.py serializers.py console.py ./
COPY data/ ./data/

RUN pip install --no-cache-dir .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
