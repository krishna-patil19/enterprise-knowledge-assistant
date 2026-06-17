from backend.retrieval import database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_all_files():
    db_files = database.get_all_files()
    removed = 0
    
    for f in db_files:
        path = f["path"]
        logger.info(f"Deleting file: {path}")
        database.delete_file(path)
        removed += 1
            
    logger.info(f"Cleanup complete. Removed {removed} files. Database is clean.")

if __name__ == "__main__":
    cleanup_all_files()
