FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry==2.5.1 \
    && poetry config virtualenvs.create false

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

COPY . .

# Run as an unprivileged user, not root.
RUN useradd --create-home app
USER app

EXPOSE 8000
# One process only: conversation memory lives in process memory.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
