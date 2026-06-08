# MLflow tracking server for the World Cup prediction project
FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# MLflow UI port
EXPOSE 5000

# Launch the tracking server / UI backed by the SQLite DB
CMD ["mlflow", "ui", \
     "--backend-store-uri", "sqlite:///mlflow.db", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
