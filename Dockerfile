FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN python -m pip install --no-cache-dir ".[api]"

EXPOSE 8070
CMD ["python", "-m", "uvicorn", "rcm_api:app", "--host", "0.0.0.0", "--port", "8070"]
