# S3 Auto-Ingestion Daemon Entry Point
# File: run_watcher.py

import os
import logging
from backend.ingestion.watcher import S3Watcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    # Point to the simulated S3 bucket
    project_root = os.path.dirname(os.path.abspath(__file__))
    s3_bucket_dir = os.path.join(project_root, "s3_bucket")
    
    watcher = S3Watcher(s3_bucket_dir)
    watcher.start()
