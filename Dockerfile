# Use Python 3.11
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your 22MB folder
COPY . .

# Run the app using gunicorn on port 8080
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app:app
