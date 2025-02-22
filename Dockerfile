# Use official Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy dependencies & install
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
