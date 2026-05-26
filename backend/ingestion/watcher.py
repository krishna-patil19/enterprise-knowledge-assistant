# Watchdog Daemon for Simulated S3 Ingestion
# File: backend/ingestion/watcher.py

import os
import time
import logging
from threading import Timer
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from backend.ingestion.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)

class DebouncedPipelineTrigger:
    """
    Prevents the ingestion pipeline from running multiple times 
    if a batch of files is dropped at once.
    """
    def __init__(self, wait_seconds=2.0):
        self.wait_seconds = wait_seconds
        self.timer = None
        self.pipeline = IngestionPipeline()

    def schedule_run(self):
        if self.timer is not None:
            self.timer.cancel()
        self.timer = Timer(self.wait_seconds, self._run_pipeline)
        self.timer.start()

    def _run_pipeline(self):
        logger.info("File activity settled. Triggering auto-ingestion pipeline...")
        try:
            stats = self.pipeline.run_scan_and_index()
            if stats["indexed"] > 0 or stats["scanned"] == 0: # handle deletes too
                logger.info(f"Auto-ingestion complete. Indexed: {stats['indexed']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")
            else:
                logger.info("Auto-ingestion scan finished. No modified files needed indexing.")
        except Exception as e:
            logger.error(f"Auto-ingestion failed: {e}")

class S3BucketEventHandler(FileSystemEventHandler):
    """
    Listens to filesystem events inside the S3 bucket directory.
    """
    def __init__(self, trigger: DebouncedPipelineTrigger):
        super().__init__()
        self.trigger = trigger

    def on_created(self, event):
        if not event.is_directory:
            logger.info(f"Detected new file: {event.src_path}")
            self.trigger.schedule_run()

    def on_modified(self, event):
        if not event.is_directory:
            logger.info(f"Detected modified file: {event.src_path}")
            self.trigger.schedule_run()

    def on_deleted(self, event):
        if not event.is_directory:
            logger.info(f"Detected deleted file: {event.src_path}")
            self.trigger.schedule_run()

class S3Watcher:
    """
    Daemon manager for the folder observer.
    """
    def __init__(self, s3_root: str):
        self.s3_root = s3_root
        self.observer = Observer()
        self.trigger = DebouncedPipelineTrigger()
        self.handler = S3BucketEventHandler(self.trigger)

    def start(self):
        os.makedirs(self.s3_root, exist_ok=True)
        self.observer.schedule(self.handler, self.s3_root, recursive=True)
        self.observer.start()
        logger.info(f"Started S3 Auto-Ingestion Watcher on: {self.s3_root}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        logger.info("Stopping S3 Auto-Ingestion Watcher...")
        self.observer.stop()
        self.observer.join()
