# Auto-Ingestion Pipeline for Enterprise Engineering Knowledge Assistant
# File: backend/ingestion/pipeline.py

import os
import json
import hashlib
import logging
from typing import Dict, Any, List
from backend.retrieval import database
from backend.ingestion import parsers
from backend.ingestion.relationship_mapper import RelationshipMapper
from backend.services import EmbeddingService
from backend.security.pii_shield import PIIShield

logger = logging.getLogger(__name__)

class IngestionPipeline:
    """
    Simulated S3 Ingestion Service.
    Scans files in s3_bucket/, detects modifications via SHA-256 hashes,
    parses, cleanses, embeds, and updates the vector database and relationship mappings.
    """
    
    def __init__(self, s3_root: str = None):
        if s3_root is None:
            # Default to the workspace s3_bucket
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            s3_root = os.path.join(base_dir, "s3_bucket")
            
        self.s3_root = s3_root
        # Auto-create the simulated S3 structure if it doesn't exist
        for folder in ["sql", "python", "docs", "configs", "pdfs"]:
            os.makedirs(os.path.join(self.s3_root, folder), exist_ok=True)
            
        # Ensure database is initialized
        database.init_db()

    def run_scan_and_index(self) -> Dict[str, Any]:
        """
        Scans simulated S3 folders recursively, ingests modified files,
        and regenerates relationships.
        Returns execution statistics.
        """
        logger.info(f"Starting simulated S3 Ingestion from: {self.s3_root}")
        
        from backend.security.pii_shield import PIIShield # Use the correct import path
        
        stats = {
            "scanned": 0,
            "indexed": 0,
            "skipped": 0,
            "blocked_pii": 0,
            "errors": 0,
            "details": []
        }
        
        # Keep track of active files in this scan to detect deleted files
        active_files = set()
        
        # Walk through the simulated S3 folders
        for root, dirs, files in os.walk(self.s3_root):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                # Normalize path for DB compatibility
                norm_path = os.path.abspath(file_path)
                active_files.add(norm_path)
                stats["scanned"] += 1
                
                # Get subdirectory name relative to S3 root
                rel_dir = os.path.dirname(os.path.relpath(file_path, self.s3_root))
                # Fallback to general bucket folder if in root
                bucket_folder = rel_dir if rel_dir else "general"
                
                try:
                    # 1. Read file stats
                    file_size = os.path.getsize(file_path)
                    
                    # Read content to compute hash (PDFs read differently, but let's hash their bytes)
                    with open(file_path, "rb") as f:
                        file_bytes = f.read()
                    file_hash = hashlib.sha256(file_bytes).hexdigest()
                    
                    # 2. Check if file is unchanged in DB
                    existing_file = database.get_file_by_path(norm_path)
                    if existing_file and existing_file["hash"] == file_hash and existing_file["size"] == file_size:
                        stats["skipped"] += 1
                        continue
                        
                    # For non-binary files, run security text scans
                    is_binary = file_name.endswith(".pdf")
                    file_text = ""
                    if not is_binary:
                        try:
                            file_text = file_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            pass
                            
                    # 3. Security check: Block customer payloads
                    if not is_binary and file_text:
                        if PIIShield.is_customer_data_payload(file_text):
                            stats["blocked_pii"] += 1
                            logger.warning(f"File blocked by security shield (customer payload): {file_name}")
                            stats["details"].append({"file": file_name, "status": "BLOCKED_CUSTOMER_DATA"})
                            continue
                            
                    logger.info(f"Indexing file: {file_name} in folder: {bucket_folder}")
                    stats["indexed"] += 1
                    
                    # 4. Parse file using specialized parsers
                    chunks = parsers.parse_file(norm_path)
                    
                    # Clear any existing chunks/embeddings for this file to overwrite freshly
                    database.delete_chunks_for_file(norm_path)
                    
                    # Save the base file entry
                    file_metadata = {
                        "name": file_name,
                        "folder": bucket_folder,
                        "hash": file_hash,
                        "tables": list(parsers.PythonParser.scan_for_tables(file_text)) if not is_binary else []
                    }
                    
                    # Accumulate module-level imports for Python scripts
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
                            
                    database.save_file(norm_path, bucket_folder, file_hash, file_size, file_metadata)
                    
                    # 5. Process and embed individual chunks
                    for chunk in chunks:
                        chunk_content = chunk["content"]
                        
                        # Censor standard PII from text before embedding!
                        if chunk["type"] != "pdf": # PDF texts are parsed separately
                            chunk_content, censored_any = PIIShield.scan_and_censor(chunk_content)
                            
                        # Generate OpenAI embeddings
                        embedding = EmbeddingService.get_embedding(chunk_content)
                        
                        # Save to SQLite + FTS
                        database.save_chunk(
                            chunk_id=chunk["id"],
                            file_path=norm_path,
                            content=chunk_content,
                            chunk_type=chunk["type"],
                            token_count=chunk["token_count"],
                            metadata=chunk["metadata"],
                            embedding=embedding
                        )
                        
                    stats["details"].append({"file": file_name, "status": "INDEXED", "chunks": len(chunks)})
                    
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Failed to process file {file_name}: {str(e)}")
                    stats["details"].append({"file": file_name, "status": "ERROR", "error": str(e)})

        # 6. Detect and clean up deleted files (files in DB but not on disk)
        db_files = database.get_all_files()
        for db_file in db_files:
            db_path = db_file["path"]
            if db_path not in active_files:
                logger.info(f"Removing deleted file from database: {os.path.basename(db_path)}")
                database.delete_file(db_path)

        # 7. Recalculate code/documentation relationships
        if stats["indexed"] > 0 or len(active_files) != len(db_files):
            RelationshipMapper.resolve_and_save_relationships()
            
        return stats
