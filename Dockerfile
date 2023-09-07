# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies your application needs
RUN pip install -r requirements.txt

# Define the command to run your application
CMD ["python", "test.py"]
