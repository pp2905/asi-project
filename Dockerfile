# Dockerfile

# Base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r src/requirements.txt

# Expose any necessary ports
# EXPOSE ...

# Run the pipeline command when the container starts
CMD ["kedro", "run"]