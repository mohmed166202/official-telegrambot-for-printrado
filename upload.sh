#!/bin/bash
set -x  # Enable debugging

# Set variables
EC2_USER="ec2-user"
EC2_HOST="ec2-3-75-215-87.eu-central-1.compute.amazonaws.com"
PEM_KEY="/c/Users/moham/Downloads/book-recommender-key-pair.pem"
REMOTE_PATH="/home/ec2-user/app"

echo "Uploading files to EC2 instance..."

# Upload files using SCP
scp -i "$PEM_KEY" main.py config.py requirements.txt "$EC2_USER@$EC2_HOST:$REMOTE_PATH"

# Check if SCP was successful
if [ $? -ne 0 ]; then
    echo "Failed to upload files."
    exit 1
fi

echo "Upload successful!"
exit 0
