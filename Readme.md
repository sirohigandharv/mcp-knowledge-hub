## To build the project
```
docker-compose up --build

```

## To force create new build 
```
docker-compose down --volumes --remove-orphans
docker-compose build --no-cache
docker-compose up
```

----

## GCP Installation (Cloud Run) Instructions

# Login
gcloud auth login

# Create Repo in Google Cloud Artifact Repo
gcloud artifacts repositories create mcp-knowledge-hub --repository-format=docker --location=us-central1 --description="MCP project images for Service Now MCP"

# Authenticate with GCP
gcloud auth configure-docker us-central1-docker.pkg.dev


# Build
docker-compose build


# Tag images for Artifact Registry
docker tag mcp-knowledge-hub-mcp-server:latest us-central1-docker.pkg.dev/gs-gcp-pca/mcp-knowledge-hub/mcp-knowledge-hub-mcp-server:latest
docker tag mcp-knowledge-hub-mcp-client:latest us-central1-docker.pkg.dev/gs-gcp-pca/mcp-knowledge-hub/mcp-knowledge-hub-mcp-client:latest


# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/gs-gcp-pca/mcp-knowledge-hub/mcp-knowledge-hub-mcp-server:latest
docker push us-central1-docker.pkg.dev/gs-gcp-pca/mcp-knowledge-hub/mcp-knowledge-hub-mcp-client:latest


# Deploy

gcloud run deploy mcp-server --image us-central1-docker.pkg.dev/gs-gcp-pca/mcp-knowledge-hub/mcp-knowledge-hub-mcp-server:latest --port 8080 --allow-unauthenticated --region us-central1 --timeout=20m --cpu=4 --memory=16Gi --execution-environment=gen2 --set-env-vars "SERVICENOW_INSTANCE_URL=https://epamsvsdemo4.service-now.com, SERVICENOW_USERNAME=eliteademouser, SERVICENOW_PASSWORD=Epam$4321, SERVICENOW_AUTH_TYPE=basic, MCP_TOOL_PACKAGE=knowledge_author"


This will give you a public HTTPS URL like: https://mcp-server-xxxxxx-uc.a.run.app (use it in the below command)


gcloud run deploy mcp-client --image us-central1-docker.pkg.dev/gs-gcp-pca/mcp-knowledge-hub/mcp-knowledge-hub-mcp-client:latest --port 8000 --allow-unauthenticated --region us-central1 --timeout=20m --cpu=4 --memory=16Gi --execution-environment=gen2 --set-env-vars "MCP_SERVER_URL=https://mcp-server-544300801285.us-central1.run.app/sse, storage_mode=cloud, GCP_BUCKET_NAME=project_sn_mcp_service_now_kp_files"


Visit the client’s URL (Cloud Run will give you another HTTPS link), and it should connect to the MCP server over HTTPS.