# Use a lightweight Python base image
FROM python:3.10-slim

# Install ffmpeg (needed by the app)
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy app files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Hugging Face Spaces uses by default
EXPOSE 7860

# Run Streamlit app
CMD ["streamlit", "run", "video_slicer_v2.py", "--server.port=7860", "--server.address=0.0.0.0"]
