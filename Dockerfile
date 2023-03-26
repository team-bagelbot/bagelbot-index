# Start with python 3.10 image
FROM python:3.10-slim

# Environment vars
ARG PROJECT_ID=local
ENV PROJECT_ID=$PROJECT_ID

# Copy the current directory into /app on the image
WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Entry point command
CMD ["python3", "main.py"]
