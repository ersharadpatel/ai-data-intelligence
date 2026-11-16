# Use an official lightweight Python 3.11 image.
FROM python:3.11-slim

# Set the working directory inside the container.
WORKDIR /workspace

# Prevent Python from generating bytecode files.
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python output appears immediately in container logs.
ENV PYTHONUNBUFFERED=1

# Copy the dependency list into the container.
COPY requirements.txt .

# Install the required Python libraries.
RUN pip install --no-cache-dir -r requirements.txt

# Keep the container running for interactive development.
CMD ["sleep", "infinity"]