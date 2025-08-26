FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only required files and directories
COPY sn_mcp_client/ ./sn_mcp_client/
COPY config.client.ssc.json .
COPY pyproject.toml .


EXPOSE 8000

CMD ["uvicorn", "sn_mcp_client.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
