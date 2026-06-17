# Auto-Ingestion Pipeline for Enterprise Engineering Knowledge Assistant
# File: backend/ingestion/pipeline.py

import os
import json
import hashlib
import logging
import subprocess
import shutil
from typing import Dict, Any, List
from backend.retrieval import database
from backend.ingestion import parsers
from backend.ingestion.relationship_mapper import RelationshipMapper
from backend.services import EmbeddingService
from backend.security.pii_shield import PIIShield

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Ingestion Service.
    Supports multi-mode storage sources:
    1. "local" - Scans files in data/
    2. "s3" - Scans files in a real AWS S3 bucket
    3. "github" - Clones a public github repo and scans it
    """
    
    def __init__(self, s3_root: str = None):
        self.ingestion_source = os.environ.get("S3_INGESTION_SOURCE", "local").lower()
        self.bucket_name = os.environ.get("S3_BUCKET_NAME", "")
        self.github_repo_url = os.environ.get("GITHUB_REPO_URL", "")
        
        # Load available buckets for multi-bucket support
        bucket_names_str = os.environ.get("S3_BUCKET_NAMES", "")
        if bucket_names_str:
            self.available_buckets = [b.strip() for b in bucket_names_str.split(",") if b.strip()]
        else:
            self.available_buckets = [self.bucket_name] if self.bucket_name else []
        
        if s3_root is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            s3_root = os.path.join(base_dir, "data")
            
        self.s3_root = s3_root
        
        # Only create directories if we are in local mode
        if self.ingestion_source == "local":
            for folder in ["sql", "python", "docs", "configs", "pdfs"]:
                os.makedirs(os.path.join(self.s3_root, folder), exist_ok=True)
            
        # Ensure database is initialized
        database.init_db()

    def set_bucket(self, bucket_name: str):
        """Switch the active S3 bucket at runtime."""
        self.bucket_name = bucket_name
        logger.info(f"Switched active S3 bucket to: {bucket_name}")


    def _get_s3_client(self):
        """Creates a Boto3 S3 client."""
        import boto3
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        
        if aws_access_key and aws_secret_key and len(aws_access_key) > 10 and not aws_access_key.startswith("your_"):
            return boto3.client(
                service_name="s3",
                region_name=os.environ.get("AWS_REGION", "ap-south-1"),
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
            )
        else:
            return boto3.client(
                service_name="s3",
                region_name=os.environ.get("AWS_REGION", "ap-south-1"),
            )

    def run_scan_and_index(self) -> Dict[str, Any]:
        """
        Routes the scan request to the proper storage handler.
        """
        if self.ingestion_source == "s3":
            return self._run_scan_s3()
        elif self.ingestion_source == "github":
            return self._run_scan_github()
        else:
            return self._run_scan_local(target_dir=self.s3_root, prefix="")

    def _run_scan_github(self) -> Dict[str, Any]:
        """Clones a GitHub repository to a temp directory and indexes it."""
        if not self.github_repo_url:
            raise ValueError("GITHUB_REPO_URL environment variable is not set.")
            
        logger.info(f"Starting GitHub Ingestion from repo: {self.github_repo_url}")
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        temp_dir = os.path.join(base_dir, ".temp_github_ingest")
        
        # Cleanup existing repo to ensure fresh clone
        if os.path.exists(temp_dir):
            try:
                subprocess.run(["rmdir", "/s", "/q", temp_dir] if os.name == "nt" else ["rm", "-rf", temp_dir], shell=True)
            except Exception as e:
                logger.warning(f"Failed to cleanly remove old github temp dir: {e}")
        
        logger.info(f"Cloning repository {self.github_repo_url} into {temp_dir}...")
        try:
            # Clone with depth 1 to save time and bandwidth
            subprocess.run(["git", "clone", "--depth", "1", self.github_repo_url, temp_dir], 
                           capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git clone failed: {e.stderr}")
            
        # Extract repo name for database path prefixing
        repo_name = self.github_repo_url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
            
        prefix = f"github://{repo_name}/"
        
        # Run the local scanner logic over the cloned directory
        stats = self._run_scan_local(target_dir=temp_dir, prefix=prefix)
        return stats

    def _run_scan_s3(self) -> Dict[str, Any]:
        """Scans real S3 bucket, downloads modified/new files locally for parsing, and indexes them."""
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is not set.")
            
        logger.info(f"Starting real S3 Ingestion from bucket: {self.bucket_name}")
        
        stats = {
            "scanned": 0,
            "indexed": 0,
            "skipped": 0,
            "blocked_pii": 0,
            "errors": 0,
            "details": []
        }
        
        s3 = self._get_s3_client()
        active_files = set()
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        temp_dir = os.path.join(base_dir, ".temp_s3_ingest")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket_name)
            
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue
                        
                    stats["scanned"] += 1
                    file_name = os.path.basename(key)
                    s3_path = f"s3://{self.bucket_name}/{key}"
                    active_files.add(s3_path)
                    
                    file_size = obj['Size']
                    file_etag = obj['ETag'].strip('"')
                    
                    rel_dir = os.path.dirname(key)
                    bucket_folder = rel_dir.replace("\\", "/") if rel_dir else "general"
                    if "/" in bucket_folder:
                        bucket_folder = bucket_folder.split("/")[0]
                    
                    try:
                        existing_file = database.get_file_by_path(s3_path)
                        if existing_file and existing_file["hash"] == file_etag and existing_file["size"] == file_size:
                            stats["skipped"] += 1
                            continue
                            
                        logger.info(f"Indexing S3 file: {key} (Size: {file_size} bytes)")
                        stats["indexed"] += 1
                        
                        ext = os.path.splitext(file_name)[1].lower()
                        temp_file_name = f"temp_{hashlib.md5(key.encode()).hexdigest()}{ext}"
                        temp_file_path = os.path.join(temp_dir, temp_file_name)
                        
                        s3.download_file(self.bucket_name, key, temp_file_path)
                        
                        with open(temp_file_path, "rb") as f:
                            file_bytes = f.read()
                            
                        is_binary = file_name.endswith(".pdf")
                        file_text = ""
                        if not is_binary:
                            try:
                                file_text = file_bytes.decode("utf-8", errors="ignore")
                            except Exception:
                                pass
                                
                        if not is_binary and file_text:
                            if PIIShield.is_customer_data_payload(file_text):
                                stats["blocked_pii"] += 1
                                logger.warning(f"S3 Object blocked by security shield (customer payload): {key}")
                                stats["details"].append({"file": key, "status": "BLOCKED_CUSTOMER_DATA"})
                                try:
                                    os.remove(temp_file_path)
                                except Exception:
                                    pass
                                continue
                                
                        chunks = parsers.parse_file(temp_file_path)
                        database.delete_chunks_for_file(s3_path)
                        
                        file_metadata = {
                            "name": file_name,
                            "folder": bucket_folder,
                            "hash": file_etag,
                            "tables": list(parsers.PythonParser.scan_for_tables(file_text)) if not is_binary else []
                        }
                        
                        if file_name.endswith(".py"):
                            try:
                                import ast
                                tree = ast.parse(file_text)
                                imports = []
                                for node in ast.walk(tree):
                                    if isinstance(node, ast.Import):
                                        for n in node.names:
                                            imports.append(n.name)
                                    elif isinstance(node, ast.ImportFrom) and node.module:
                                        imports.append(node.module)
                                file_metadata["imports"] = imports
                            except Exception:
                                pass
                                
                        database.save_file(s3_path, bucket_folder, file_etag, file_size, file_metadata)
                        
                        for chunk in chunks:
                            chunk_content = chunk["content"]
                            if chunk["type"] != "pdf":
                                chunk_content, censored_any = PIIShield.scan_and_censor(chunk_content)
                                
                            embedding = EmbeddingService.get_embedding(chunk_content)
                            
                            database.save_chunk(
                                chunk_id=chunk["id"],
                                file_path=s3_path,
                                content=chunk_content,
                                chunk_type=chunk["type"],
                                token_count=chunk["token_count"],
                                metadata=chunk["metadata"],
                                embedding=embedding
                            )
                            
                        stats["details"].append({"file": key, "status": "INDEXED", "chunks": len(chunks)})
                        
                        try:
                            os.remove(temp_file_path)
                        except Exception:
                            pass
                            
                    except Exception as e:
                        stats["errors"] += 1
                        logger.error(f"Failed to process S3 object {key}: {str(e)}")
                        stats["details"].append({"file": key, "status": "ERROR", "error": str(e)})
                        
        except Exception as e:
            logger.error(f"Error accessing S3 bucket {self.bucket_name}: {str(e)}")
            stats["errors"] += 1
            stats["details"].append({"file": f"bucket:{self.bucket_name}", "status": "ERROR", "error": str(e)})
            
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
            
        db_files = database.get_all_files()
        for db_file in db_files:
            db_path = db_file["path"]
            if db_path.startswith(f"s3://{self.bucket_name}/") and db_path not in active_files:
                logger.info(f"Removing deleted S3 file from database: {db_path}")
                database.delete_file(db_path)
                
        s3_db_files_count = len([f for f in db_files if f["path"].startswith(f"s3://{self.bucket_name}/")])
        if stats["indexed"] > 0 or len(active_files) != s3_db_files_count:
            RelationshipMapper.resolve_and_save_relationships()
            
        return stats

    def _run_scan_local(self, target_dir: str, prefix: str = "") -> Dict[str, Any]:
        """
        Directory scanner logic.
        Uses `prefix` for database mapping (e.g. "github://repo/" or "" for local).
        """
        logger.info(f"Starting local file scan on: {target_dir}")
        
        stats = {
            "scanned": 0,
            "indexed": 0,
            "skipped": 0,
            "blocked_pii": 0,
            "errors": 0,
            "details": []
        }
        
        active_files = set()
        
        for root, dirs, files in os.walk(target_dir):
            if ".git" in dirs:
                dirs.remove(".git")
                
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                rel_path = os.path.relpath(file_path, target_dir)
                rel_path = rel_path.replace("\\", "/")
                
                db_path = prefix + rel_path if prefix else os.path.abspath(file_path)
                active_files.add(db_path)
                stats["scanned"] += 1
                
                rel_dir = os.path.dirname(rel_path)
                bucket_folder = rel_dir.split("/")[0] if rel_dir else "general"
                
                try:
                    file_size = os.path.getsize(file_path)
                    
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    
                    existing_file = database.get_file_by_path(db_path)
                    if existing_file and existing_file["hash"] == file_hash and existing_file["size"] == file_size:
                        stats["skipped"] += 1
                        continue
                        
                    is_binary = file_name.endswith(".pdf")
                    file_text = ""
                    if not is_binary:
                        try:
                            file_text = file_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            pass
                            
                    if not is_binary and file_text:
                        if PIIShield.is_customer_data_payload(file_text):
                            stats["blocked_pii"] += 1
                            logger.warning(f"File blocked by security shield: {file_name}")
                            stats["details"].append({"file": file_name, "status": "BLOCKED_CUSTOMER_DATA"})
                            continue
                            
                    logger.info(f"Indexing file: {db_path} in folder: {bucket_folder}")
                    stats["indexed"] += 1
                    
                    chunks = parsers.parse_file(file_path)
                    database.delete_chunks_for_file(db_path)
                    
                    file_metadata = {
                        "name": file_name,
                        "folder": bucket_folder,
                        "hash": file_hash,
                        "tables": list(parsers.PythonParser.scan_for_tables(file_text)) if not is_binary else []
                    }
                    
                    if file_name.endswith(".py"):
                        try:
                            import ast
                            tree = ast.parse(file_text)
                            imports = []
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Import):
                                    for n in node.names:
                                        imports.append(n.name)
                                elif isinstance(node, ast.ImportFrom) and node.module:
                                        imports.append(node.module)
                            file_metadata["imports"] = imports
                        except Exception:
                            pass
                            
                    database.save_file(db_path, bucket_folder, file_hash, file_size, file_metadata)
                    
                    for chunk in chunks:
                        chunk_content = chunk["content"]
                        if chunk["type"] != "pdf":
                            chunk_content, censored_any = PIIShield.scan_and_censor(chunk_content)
                            
                        embedding = EmbeddingService.get_embedding(chunk_content)
                        
                        database.save_chunk(
                            chunk_id=chunk["id"],
                            file_path=db_path,
                            content=chunk_content,
                            chunk_type=chunk["type"],
                            token_count=chunk["token_count"],
                            metadata=chunk["metadata"],
                            embedding=embedding
                        )
                        
                    stats["details"].append({"file": file_name, "status": "INDEXED", "chunks": len(chunks)})
                    
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Failed to process file {db_path}: {str(e)}")
                    stats["details"].append({"file": db_path, "status": "ERROR", "error": str(e)})

        db_files = database.get_all_files()
        for db_file in db_files:
            curr_db_path = db_file["path"]
            
            in_scope = curr_db_path.startswith(prefix) if prefix else (not curr_db_path.startswith("s3://") and not curr_db_path.startswith("github://"))
            
            if in_scope and curr_db_path not in active_files:
                logger.info(f"Removing deleted file from database: {curr_db_path}")
                database.delete_file(curr_db_path)

        scoped_db_count = len([f for f in db_files if (f["path"].startswith(prefix) if prefix else (not f["path"].startswith("s3://") and not f["path"].startswith("github://")))])
        if stats["indexed"] > 0 or len(active_files) != scoped_db_count:
            RelationshipMapper.resolve_and_save_relationships()
            
        return stats
