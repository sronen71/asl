#!/bin/bash

# Replace these variables with your desired values
BUCKET_NAME="asl-tfrecords"
LOCAL_FOLDER_PATH="/kaggle/input/asl-preprocessing/records"
AWS_REGION="us-west-2"

# Create S3 bucket
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION

# Transfer data to S3 bucket
aws s3 sync $LOCAL_FOLDER_PATH s3://$BUCKET_NAME

# Verify the transfer
aws s3 ls s3://$BUCKET_NAME

