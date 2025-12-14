# Use the official Playwright image which comes with Python and Browsers pre-installed
FROM mcr.microsoft.com/playwright/python:v1.48.0-noble

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Create a non-root user (Hugging Face requirement for security)
# The playwright image usually has a 'pwuser', but let's be safe and use a new user or root if strict permissions aren't forced. 
# HF Spaces run as user 1000 by default.
RUN useradd -m -u 1000 user
USER user

# Expose port 7860 (Hugging Face specific port)
EXPOSE 7860

# Command to run the app
# Note: We bind to 0.0.0.0 and port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]