FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY app ./app

# ✅ Copy new engine package (this is what you're missing)
COPY signal_engine ./signal_engine

# ✅ Ensure Python can import /app/signal_engine
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]