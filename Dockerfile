FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY prompts/ prompts/

RUN pip install --no-cache-dir . \
    && useradd --create-home worker \
    && mkdir /data \
    && chown worker:worker /data

ENV PYTHONPATH=/app/src
ENV LOCAL_CODE_WORKER_CONTAINER=1

USER worker

EXPOSE 8765
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "local_code_worker"]
