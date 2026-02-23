FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy API code
COPY app ./app

# Copy engine package
COPY signal_engine ./signal_engine

# ✅ ADD THIS LINE
COPY tools ./tools

#✅ ADD THIS
COPY .env.example ./.env.example

ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]