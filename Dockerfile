FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Hugging Face Spaces routes traffic to port 7860 by default
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]