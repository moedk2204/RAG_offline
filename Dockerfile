# Use the full Bookworm image (comes with build-essential pre-installed)
# This avoids many of the 'slim' image download/mirror issues.
FROM python:3.12-bookworm

# Set the working directory in the container
WORKDIR /app

# libgomp1 is needed for FAISS
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create data directories
RUN mkdir -p data/inputs data/vector_db

# Expose the port Gradio runs on
EXPOSE 7860

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV HF_HOME="/app/data/hf_cache"

# Run the app
CMD ["python", "app.py"]
