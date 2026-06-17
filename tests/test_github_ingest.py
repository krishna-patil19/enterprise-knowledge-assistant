from dotenv import load_dotenv
import os

# Explicitly load .env from the current directory, overriding any existing environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'), override=True)

import logging
from backend.ingestion.pipeline import IngestionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_test():
    source = os.environ.get("S3_INGESTION_SOURCE")
    repo = os.environ.get("GITHUB_REPO_URL")
    
    logger.info(f"Ingestion Source: {source}")
    logger.info(f"GitHub Repo: {repo}")
    
    if source != "github":
        logger.error("Environment variable S3_INGESTION_SOURCE is not set to 'github'. Check .env loading.")
        return
        
    pipeline = IngestionPipeline()
    stats = pipeline.run_scan_and_index()
    
    logger.info("GitHub Ingestion Stats:")
    for k, v in stats.items():
        if k != "details":
            logger.info(f"  {k}: {v}")
            
    print("\n✅ Test Complete. Stats:")
    print(stats)

if __name__ == "__main__":
    run_test()
