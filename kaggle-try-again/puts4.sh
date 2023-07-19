#!/bin/bash

# Replace these variables with your desired values
BUCKET_NAME="asl-input"
LOCAL_FOLDER_PATH="/kaggle/input/asl-fingerspelling"
AWS_REGION="us-west-2"

# Create S3 bucket
aws s3 mb s3://$BUCKET_NAME --region $AWS_REGION

# Transfer data to S3 bucket
aws s3 cp $LOCAL_FOLDER_PATH/character_to_prediction_index.json  s3://$BUCKET_NAME
aws s3 cp $LOCAL_FOLDER_PATH/train.csv  s3://$BUCKET_NAME
aws s3 cp $LOCAL_FOLDER_PATH/supplemental_metadata.csv  s3://$BUCKET_NAME



# Verify the transfer
aws s3 ls s3://$BUCKET_NAME

