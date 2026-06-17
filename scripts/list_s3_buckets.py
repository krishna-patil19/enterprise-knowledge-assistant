"""List available S3 buckets."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import boto3

key = os.getenv("AWS_ACCESS_KEY_ID", "")
secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
region = os.getenv("AWS_REGION", "us-east-1")

if key and secret and len(key) > 10:
    client = boto3.client("s3", region_name=region, aws_access_key_id=key, aws_secret_access_key=secret)
else:
    client = boto3.client("s3", region_name=region)

res = client.list_buckets()
print("Available S3 Buckets:")
for b in res["Buckets"]:
    print(f"  - {b['Name']}")
