# Temporary test script to verify AWS S3 ingestion
import os
import sys
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval import database

if __name__ == "__main__":
    print("Testing real AWS S3 bucket integration...")
    print(f"S3_INGESTION_SOURCE = {os.environ.get('S3_INGESTION_SOURCE')}")
    print(f"S3_BUCKET_NAME = {os.environ.get('S3_BUCKET_NAME')}")
    
    try:
        pipeline = IngestionPipeline()
        stats = pipeline.run_scan_and_index()
        print("\nScan completed successfully!")
        print("Statistics:")
        print(f"  Scanned       : {stats['scanned']}")
        print(f"  Indexed       : {stats['indexed']}")
        print(f"  Skipped       : {stats['skipped']}")
        print(f"  Blocked PII   : {stats['blocked_pii']}")
        print(f"  Errors        : {stats['errors']}")
        print("Details:")
        for detail in stats['details']:
            print(f"  - {detail}")
            
        print("\nClickHouse DB Stats:")
        db_stats = database.get_db_stats()
        print(f"  Files         : {db_stats['files']}")
        print(f"  Chunks        : {db_stats['chunks']}")
        print(f"  Relationships : {db_stats['relationships']}")
    except Exception as e:
        print(f"\nScan failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
