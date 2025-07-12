FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY intent_router.py .

EXPOSE 8002

CMD ["uvicorn", "intent_router:app", "--host", "0.0.0.0", "--port", "8002"]