FROM node:20-slim AS web
WORKDIR /web
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.12-slim AS app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt ./
RUN grep -v -E '^(pytest|httpx)' requirements.txt > runtime-requirements.txt \
    && pip install -r runtime-requirements.txt
# Run from source: data/ and models/ live inside the package dir and are not package-data.
COPY ringbreaker/ ./ringbreaker/
COPY --from=web /web/dist ./dashboard/dist
RUN useradd --create-home app && chown -R app /app/ringbreaker/models
USER app
ENV RINGBREAKER_DASHBOARD_DIST=/app/dashboard/dist
CMD ["sh", "-c", "exec python -m uvicorn ringbreaker.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
