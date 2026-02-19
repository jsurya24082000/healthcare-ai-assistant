#!/bin/bash
# Healthcare RAG Assistant - AWS EC2 Deployment Script
# Deploys containerized microservice to EC2 with monitoring

set -e

echo "🚀 Healthcare RAG EC2 Deployment"
echo "================================"

# Configuration
INSTANCE_TYPE="t3.large"  # 2 vCPU, 8GB RAM
AMI_ID="ami-0c55b159cbfafe1f0"  # Amazon Linux 2
KEY_NAME="healthcare-rag-key"
SECURITY_GROUP="healthcare-rag-sg"
REGION="us-east-1"

# Step 1: Create security group (if not exists)
echo "📦 Setting up security group..."
aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "Healthcare RAG API security group" \
    --region $REGION 2>/dev/null || true

# Allow HTTP, HTTPS, and SSH
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp --port 22 --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp --port 8000 --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp --port 443 --cidr 0.0.0.0/0 \
    --region $REGION 2>/dev/null || true

# Step 2: Launch EC2 instance
echo "🖥️  Launching EC2 instance ($INSTANCE_TYPE)..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=healthcare-rag-api}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "✅ Instance launched: $INSTANCE_ID"

# Wait for instance to be running
echo "⏳ Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "🌐 Public IP: $PUBLIC_IP"

# Step 3: Setup script for the instance
cat << 'SETUP_SCRIPT' > /tmp/setup.sh
#!/bin/bash
set -e

# Update system
sudo yum update -y

# Install Docker
sudo amazon-linux-extras install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/jsurya24082000/healthcare-ai-assistant.git
cd healthcare-ai-assistant

# Set environment variables
echo "OPENAI_API_KEY=${OPENAI_API_KEY}" > .env

# Build and run containers
docker-compose up -d --build

echo "✅ Healthcare RAG API deployed successfully!"
SETUP_SCRIPT

# Step 4: Copy and execute setup script
echo "📤 Deploying application..."
scp -i ~/.ssh/$KEY_NAME.pem -o StrictHostKeyChecking=no /tmp/setup.sh ec2-user@$PUBLIC_IP:/home/ec2-user/
ssh -i ~/.ssh/$KEY_NAME.pem -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP "chmod +x /home/ec2-user/setup.sh && /home/ec2-user/setup.sh"

echo ""
echo "================================"
echo "🎉 Deployment Complete!"
echo "================================"
echo "API Endpoint: http://$PUBLIC_IP:8000"
echo "Health Check: http://$PUBLIC_IP:8000/health"
echo "Metrics:      http://$PUBLIC_IP:8000/metrics"
echo "Prometheus:   http://$PUBLIC_IP:9090"
echo "Grafana:      http://$PUBLIC_IP:3000"
echo ""
echo "Test with:"
echo "curl -X POST http://$PUBLIC_IP:8000/query -H 'Content-Type: application/json' -d '{\"question\": \"What is the patient privacy policy?\"}'"
