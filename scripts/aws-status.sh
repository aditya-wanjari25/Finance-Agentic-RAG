#!/bin/bash
# scripts/aws-status.sh
# Shows current state of all AWS resources

REGION="us-east-1"

echo ""
echo "================================================="
echo "  FinSight RAG — AWS Status"
echo "================================================="
echo ""

# ECS Status
echo "🐳 ECS Service:"
aws ecs describe-services \
    --cluster finsight-rag \
    --services finsight-rag-service \
    --region $REGION \
    --query 'services[0].{running:runningCount,desired:desiredCount,status:status}' \
    --output table 2>/dev/null || echo "  Not found"

# OpenSearch Status
echo ""
echo "🔍 OpenSearch:"
STATUS=$(aws opensearch describe-domains \
    --domain-names finsight-rag \
    --region $REGION \
    --query 'DomainStatusList[0].{processing:Processing,endpoint:Endpoint}' \
    --output table 2>/dev/null || echo "  Not found / deleted")
echo "$STATUS"

# Current IP if available
if [ -f "scripts/.current-ip" ]; then
    IP=$(cat scripts/.current-ip)
    echo ""
    echo "🌐 Last known IP: $IP"
    echo "   Testing connectivity..."
    curl -s --max-time 5 http://$IP:8000/health | python -m json.tool 2>/dev/null \
        || echo "   ⚠️  Not reachable (service may be down)"
fi

# S3 documents
echo ""
echo "📦 S3 Documents:"
aws s3 ls s3://finsight-rag-documents/raw/ --recursive \
    --human-readable 2>/dev/null || echo "  No documents found"

echo ""
echo "================================================="
echo ""