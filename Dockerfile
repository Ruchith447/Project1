# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies first (layer caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code and data files
COPY main.py .
COPY cleaned-content.json .
COPY cleaned-discourse.json .

# Expose port 80 (standard HTTP port)
EXPOSE 80

# Run the app with Uvicorn, binding to 0.0.0.0 to accept external connections
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

