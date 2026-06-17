"""Upload local sample files to both S3 buckets for testing.

Bucket 1 (Ott-demo-bucket-1): SQL + Python files (analytics_queries.sql, mock_table.sql, etl_pipeline.py)
Bucket 2 (Ott-demo-bucket-2): Docs + Config files (clickhouse_guide.md, analytics_config.yaml)

This gives each bucket different content so you can clearly see the bucket switching works.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

import boto3

aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY", "")
region = os.getenv("AWS_REGION", "us-east-1")

if aws_key and aws_secret and len(aws_key) > 10:
    s3 = boto3.client("s3", region_name=region, aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
else:
    s3 = boto3.client("s3", region_name=region)

base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Bucket 1: SQL + Python
bucket1_files = {
    "sql/analytics_queries.sql": os.path.join(base_dir, "sql", "analytics_queries.sql"),
    "sql/mock_table.sql": os.path.join(base_dir, "sql", "mock_table.sql"),
    "python/etl_pipeline.py": os.path.join(base_dir, "python", "etl_pipeline.py"),
}

# Bucket 2: Docs + Config
bucket2_files = {
    "docs/clickhouse_guide.md": os.path.join(base_dir, "docs", "clickhouse_guide.md"),
    "configs/analytics_config.yaml": os.path.join(base_dir, "configs", "analytics_config.yaml"),
}

print("=" * 60)
print("UPLOADING FILES TO S3 BUCKETS")
print("=" * 60)

print("\n--- Bucket: ott-demo-bucket-1 ---")
for s3_key, local_path in bucket1_files.items():
    try:
        s3.upload_file(local_path, "ott-demo-bucket-1", s3_key)
        print(f"  [OK] Uploaded: {s3_key}")
    except Exception as e:
        print(f"  [FAIL] {s3_key}: {str(e)}")

print("\n--- Bucket: ott-demo-bucket-2 ---")
for s3_key, local_path in bucket2_files.items():
    try:
        s3.upload_file(local_path, "ott-demo-bucket-2", s3_key)
        print(f"  [OK] Uploaded: {s3_key}")
    except Exception as e:
        print(f"  [FAIL] {s3_key}: {str(e)}")

print("\n[DONE] Upload complete!")
print("Bucket-1 has: SQL + Python files")
print("Bucket-2 has: Docs + Config files")
