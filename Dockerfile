# Use official lightweight Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable buffering
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (needed for packages like git, gcc, and build tools if required)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run the training pipeline during build time to download the datasets, 
# balance classes, train models, and generate interactive hotspot maps.
RUN python main.py

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run healthcheck to verify container health
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit web dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
