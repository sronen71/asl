#!/bin/bash

# Launch Spot instance
instance_id=$(aws ec2 request-spot-instances \
  --spot-price "0.4" \  # Specify your desired maximum price per hour
  --instance-count 1 \
  --launch-specification '{
    "ImageId": "ami-01126ee2e34c5f04e",
    "InstanceType": "g5.2xlarge",
    "KeyName": "pair23.pem",
  }' \
  --output text \
  --query 'SpotInstanceRequests[0].InstanceId')

# Wait for the instance to reach the 'running' state
aws ec2 wait instance-running --instance-ids $instance_id

# Get the public IP address of the instance
public_ip=$(aws ec2 describe-instances --instance-ids $instance_id --output text --query 'Reservations[0].Instances[0].PublicIpAddress')

# Copy the Python script to the instance
scp -i pari23.pem asl-try-again.py ubuntu@$public_ip:/home/ubuntu/asl-try-again.py
scp -i pair23.pem setup.sh ubuntu@$public_ip:/home/ubuntu/setup.sh
#
# Execute the Python script on the instance via SSH
ssh -i pair23.pem ubuntu@$public_ip "cd /home/ubuntu && source setup.sh"

ssh -i pair23.pem ubuntu@$public_ip "cd /home/ubuntu && python asl-try-again.py"
# Install AWS CLI on the instance
ssh -i pair23.pem ubuntu@$public_ip "sudo apt-get update && sudo apt-get install -y awscli"

# Download files from S3 to the instance
ssh -i pair23.pem ubuntu@$public_ip "aws s3 cp s3://your-bucket-name/path/to/file /home/ubuntu/"

