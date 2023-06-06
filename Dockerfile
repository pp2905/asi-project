FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r src/requirements.txt

CMD ["kedro", "run"]