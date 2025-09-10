FROM python:3.11-slim

WORKDIR /app

COPY devops/pyproject.toml devops/poetry.lock ./

RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

COPY src .
