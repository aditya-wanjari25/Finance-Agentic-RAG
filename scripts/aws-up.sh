#!/bin/bash
# scripts/aws-up.sh
# Restores all AWS resources for demo or interview
# Run this ~20 minutes before you need the system live

set -e

REGION="us-east-1"
CLUSTER="finsight-rag"
SERVICE="finsight-rag-service"
OPENSEARCH_DOMAIN="finsight-rag"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo ""
echo "================================================="
echo "  FinSight RAG — Spinning Up AWS Resources"
echo "================================================="
echo "  Estimated time: 15-20 minutes"
echo "  Estimated cost: ~\$40/month while running"
echo "================================================="
echo ""

# Step 1 — Recreate OpenSearch domain
echo "1️⃣  Creating OpenSearch domain..."

# Check if already exists
DOMAIN_STATUS=$(aws opensearch describe-domains \
    --domain-names $OPENSEARCH_DOMAIN \
    --region $REGION \
    --query 'DomainStatusList[0].Processing' \
    --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$DOMAIN_STATUS" == "NOT_FOUND" ] || [ "$DOMAIN_STATUS" == "None" ]; then
    aws opensearch create-domain \
        --domain-name $OPENSEARCH_DOMAIN \
        --engine-version OpenSearch_2.11 \
        --cluster-config InstanceType=t3.small.search,InstanceCount=1 \
        --ebs-options EBSEnabled=true,VolumeType=gp3,VolumeSize=20 \
        --node-to-node-encryption-options Enabled=true \
        --encryption-at-rest-options Enabled=true \
        --domain-endpoint-options EnforceHTTPS=true \
        --advanced-security-options "{
            \"Enabled\": true,
            \"InternalUserDatabaseEnabled\": true,
            \"MasterUserOptions\": {
                \"MasterUserName\": \"finsight-admin\",
                \"MasterUserPassword\": \"FinSight2025!\"
            }
        }" \
        --access-policies "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Principal\": {\"AWS\": \"*\"},
                \"Action\": \"es:*\",
                \"Resource\": \"arn:aws:es:$REGION:$AWS_ACCOUNT_ID:domain/$OPENSEARCH_DOMAIN/*\"
            }]
        }" \
        --region $REGION > /dev/null
    echo "  ✅ OpenSearch creation initiated"
else
    echo "  ℹ️  OpenSearch already exists (status: $DOMAIN_STATUS)"
fi

# Step 2 — Wait for OpenSearch to be ready
echo ""
echo "2️⃣  Waiting for OpenSearch to be ready..."
echo "  (this takes 10-15 minutes — go make a coffee ☕)"
echo ""

while true; do
    STATUS=$(aws opensearch describe-domains \
        --domain-names $OPENSEARCH_DOMAIN \
        --region $REGION \
        --query 'DomainStatusList[0].Processing' \
        --output text 2>/dev/null)

    if [ "$STATUS" == "False" ]; then
        echo "  ✅ OpenSearch is ready"
        break
    fi
    echo "  ⏳ Still provisioning... (checking again in 30s)"
    sleep 30
done

# Step 3 — Get OpenSearch endpoint and update secret
echo ""
echo "3️⃣  Updating OpenSearch endpoint in Secrets Manager..."

OPENSEARCH_ENDPOINT=$(aws opensearch describe-domains \
    --domain-names $OPENSEARCH_DOMAIN \
    --region $REGION \
    --query 'DomainStatusList[0].Endpoint' \
    --output text)

echo "  Endpoint: $OPENSEARCH_ENDPOINT"

aws secretsmanager update-secret \
    --secret-id finsight-rag/OPENSEARCH_ENDPOINT \
    --secret-string "$OPENSEARCH_ENDPOINT" \
    --region $REGION > /dev/null
echo "  ✅ Secret updated"

# Step 4 — Re-ingest documents into fresh OpenSearch
echo ""
echo "4️⃣  Re-ingesting documents into OpenSearch..."
echo "  (OpenSearch is fresh — ChromaDB data doesn't transfer)"

# Check if we have local data to re-ingest
if [ -f "data/raw/AAPL_10K_2025.pdf" ]; then
    echo "  Found AAPL_10K_2025.pdf — running ingestion..."
    OPENSEARCH_ENDPOINT=$OPENSEARCH_ENDPOINT \
    USE_OPENSEARCH=true \
    python -c "
from ingestion.parsers.pdf_parser import FinancialPDFParser
from ingestion.chunkers.hierarchical_chunker import HierarchicalChunker
from ingestion.embedders.openai_embedder import OpenAIEmbedder
from retrieval.opensearch_store import OpenSearchStore
import os

os.environ['OPENSEARCH_ENDPOINT'] = '$OPENSEARCH_ENDPOINT'
os.environ['OPENSEARCH_USERNAME'] = 'finsight-admin'
os.environ['OPENSEARCH_PASSWORD'] = 'FinSight2025!'

parser = FinancialPDFParser(ticker='AAPL', year=2025)
blocks = parser.parse('data/raw/AAPL_10K_2025.pdf')
chunker = HierarchicalChunker()
chunks = chunker.chunk(blocks)
embedder = OpenAIEmbedder()
embeddings = embedder.embed_chunks(chunks)
store = OpenSearchStore()
store.store(chunks, embeddings)
print(f'Ingested {len(chunks)} chunks')
"
    echo "  ✅ Ingestion complete"
else
    echo "  ⚠️  No local PDFs found — re-ingest manually after startup"
    echo "  Run: python -m ingestion.pipeline"
fi

# Step 5 — Restart ECS service
echo ""
echo "5️⃣  Starting ECS service..."
aws ecs update-service \
    --cluster $CLUSTER \
    --service $SERVICE \
    --desired-count 1 \
    --force-new-deployment \
    --region $REGION > /dev/null
echo "  ✅ ECS service starting..."

# Step 6 — Wait for task to be running
echo ""
echo "6️⃣  Waiting for container to be healthy..."
sleep 30

while true; do
    RUNNING=$(aws ecs describe-services \
        --cluster $CLUSTER \
        --services $SERVICE \
        --region $REGION \
        --query 'services[0].runningCount' \
        --output text)

    if [ "$RUNNING" == "1" ]; then
        echo "  ✅ Container is running"
        break
    fi
    echo "  ⏳ Waiting for container... (checking in 15s)"
    sleep 15
done

# Step 7 — Get public IP and verify
echo ""
echo "7️⃣  Getting public endpoint..."
sleep 10

TASK_ARN=$(aws ecs list-tasks \
    --cluster $CLUSTER \
    --region $REGION \
    --query 'taskArns[0]' \
    --output text)

ENI_ID=$(aws ecs describe-tasks \
    --cluster $CLUSTER \
    --tasks $TASK_ARN \
    --region $REGION \
    --query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' \
    --output text)

PUBLIC_IP=$(aws ec2 describe-network-interfaces \
    --network-interface-ids $ENI_ID \
    --region $REGION \
    --query 'NetworkInterfaces[0].Association.PublicIp' \
    --output text)

# Save IP for status script
echo $PUBLIC_IP > scripts/.current-ip

echo ""
echo "================================================="
echo "  ✅ FinSight RAG is LIVE"
echo ""
echo "  API:    http://$PUBLIC_IP:8000"
echo "  Docs:   http://$PUBLIC_IP:8000/docs"
echo "  Health: http://$PUBLIC_IP:8000/health"
echo ""
echo "  Run ./scripts/aws-down.sh when done"
echo "================================================="
echo ""

# Final health check
curl -s http://$PUBLIC_IP:8000/health | python -m json.tool