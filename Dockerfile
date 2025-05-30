FROM python:3.12-slim

# Install system dependencies needed to build some Python packages
RUN apt-get update && apt-get install -y build-essential

WORKDIR /app

# Copy requirements first and install them
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --default-timeout=100 --retries=10 --no-cache-dir --force-reinstall -r requirements.txt
RUN python -m spacy download en_core_web_sm


# Copy your application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Start your Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
