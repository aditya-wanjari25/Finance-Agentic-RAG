#!/bin/bash
# scripts/aws-down.sh
# Tears down all expensive AWS resources
# Safe to run anytime — S3, ECR, and secrets are preserved
# Estimated savings: ~$40/month when down

set -e  # exit on any error

REGION="us-east-1"
CLUSTER="finsight-rag"
SERVICE="finsight-rag-service"
OPENSEARCH_DOMAIN="finsight-rag"

echo ""
echo "================================================="
echo "  FinSight RAG — Tearing Down AWS Resources"
echo "================================================="
echo ""
echo "This will stop:"
echo "  • ECS Fargate service (~\$15/month saved)"
echo "  • OpenSearch domain (~\$25/month saved)"
echo ""
echo "This will PRESERVE:"
echo "  • S3 documents (free tier)"
echo "  • ECR Docker image (negligible cost)"
echo "  • Secrets Manager (negligible cost)"
echo "  • CloudWatch logs (negligible cost)"
echo ""
read -p "Proceed? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted."
    exit 0
fi

# Step 1 — Stop ECS service
echo ""
echo "⏹️  Stopping ECS service..."
aws ecs update-service \
    --cluster $CLUSTER \
    --service $SERVICE \
    --desired-count 0 \
    --region $REGION > /dev/null
echo "  ✅ ECS service stopped (0 running tasks)"

# Step 2 — Delete OpenSearch domain
echo ""
echo "🗑️  Deleting OpenSearch domain (takes 5-10 min in background)..."
aws opensearch delete-domain \
    --domain-name $OPENSEARCH_DOMAIN \
    --region $REGION > /dev/null
echo "  ✅ OpenSearch deletion initiated"

# Step 3 — Save current Docker image tag to file
# So aws-up.sh knows which image to deploy
IMAGE_TAG=$(aws ecr describe-images \
    --repository-name finsight-rag \
    --region $REGION \
    --query 'sort_by(imageDetails,&imagePushedAt)[-1].imageTags[0]' \
    --output text 2>/dev/null || echo "latest")
echo $IMAGE_TAG > scripts/.last-image-tag
echo "  📝 Saved image tag: $IMAGE_TAG"

echo ""
echo "================================================="
echo "  ✅ Teardown complete"
echo ""
echo "  Monthly savings: ~\$40"
echo "  To restore: ./scripts/aws-up.sh"
echo "  Estimated restore time: ~20 minutes"
echo "================================================="
echo ""