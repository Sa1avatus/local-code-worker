FROM python:3.12-slim

WORKDIR /app

ARG INSTALL_ROUTELLM=0

COPY pyproject.toml .
COPY src/ src/
COPY prompts/ prompts/

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && if [ "$INSTALL_ROUTELLM" = "1" ]; then \
         pip install --no-cache-dir ".[dev,routellm]"; \
       else \
         pip install --no-cache-dir ".[dev]"; \
       fi \
    && useradd --create-home worker \
    && mkdir /data \
    && chown worker:worker /data

ENV PYTHONPATH=/app/src
ENV LOCAL_CODE_WORKER_CONTAINER=1

USER worker

EXPOSE 8765
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "local_code_worker"]
