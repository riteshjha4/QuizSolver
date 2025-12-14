FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

# Command to run the app
# Note: We bind to 0.0.0.0 and port 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
