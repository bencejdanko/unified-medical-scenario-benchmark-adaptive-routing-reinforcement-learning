#!/bin/bash
set -e

# Configuration
FHIR_PORT=8080
IMAGE_NAME="medagentbench"
REMOTE_IMAGE="jyxsu6/medagentbench:latest"

echo "=== MedAgentBench Real Lifecycle Manager ==="

# 1. Cleanup
echo "[1/6] Cleaning up ports and containers..."
docker rm -f medagentbench_server 2>/dev/null || true
# Kill any local servers on 8080 (though usually it's docker)
fuser -k $FHIR_PORT/tcp 2>/dev/null || true

# 2. Ensure Docker Image
echo "[2/6] Ensuring MedAgentBench Docker image exists..."
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "  - Image missing. Pulling $REMOTE_IMAGE (this may take a few mins)..."
    docker pull $REMOTE_IMAGE
    docker tag $REMOTE_IMAGE $IMAGE_NAME
else
    echo "  - Image found."
fi

# 3. Start FHIR Server
echo "[3/6] Starting FHIR Server container..."
docker run -d --name medagentbench_server -p $FHIR_PORT:8080 $IMAGE_NAME

echo "  - Waiting for FHIR Server to be ready (at /fhir/metadata)..."
MAX_RETRIES=60
COUNT=0
until $(curl --output /dev/null --silent --fail http://localhost:$FHIR_PORT/fhir/metadata); do
    printf '.'
    sleep 2
    COUNT=$((COUNT+1))
    if [ $COUNT -ge $MAX_RETRIES ]; then
        echo -e "\n  - ERROR: FHIR Server failed to start in time. Check 'docker logs medagentbench_server'."
        exit 1
    fi
done
echo -e "\n  - SUCCESS: FHIR Server is up."

# 4. Verify Real Data via Direct Call
echo "[4/6] Verifying real data accessibility (Peter Stafford)..."
CURL_RESULT=$(curl -s "http://localhost:8080/fhir/Patient?given=Peter&family=Stafford&birthdate=1932-12-29")
if [[ "$CURL_RESULT" == *"S6534835"* ]]; then
    echo "  - SUCCESS: Real patient record found in database."
else
    echo "  - ERROR: Could not find real patient record. Data mismatch or server error."
    echo "    Result: $CURL_RESULT"
    exit 1
fi

# 5. Test MCP Server Lifecycle
echo "[5/6] Testing MCP Server Tooling..."

# Using the client test we wrote earlier
python3 mcp_client_test.py

# 6. Run Benchmark Evaluation Entry
echo "[6/6] Running full benchmark interaction test..."
python3 test_one.py

echo "=== Lifecycle Complete: ALL VERIFIED ==="
