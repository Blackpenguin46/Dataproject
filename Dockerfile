# Dockerfile for Streamlit app
FROM python:3.12-slim

WORKDIR /app

COPY test.py /app/
#COPY process_data.json /app/  # If you want to persist data, consider using a volume instead

# Install dependencies
RUN pip install --no-cache-dir streamlit pandas plotly scikit-learn requests

EXPOSE 8501

CMD ["streamlit", "run", "test.py", "--server.address=0.0.0.0"] 