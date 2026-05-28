# File: backend/retrieval/database.py

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv
import clickhouse_connect

logger = logging.getLogger(__name__)

load_dotenv()

def get_client():
    """Returns a ClickHouse client connection."""
    host = os.getenv('CLICKHOUSE_HOST', 'localhost')
    # If the host contains 'clickhouse.cloud', use secure connection
    secure = True if 'cloud' in host else False
    
    return clickhouse_connect.get_client(
        host=host,
        port=int(os.getenv('CLICKHOUSE_PORT', 8443 if secure else 8123)),
        username=os.getenv('CLICKHOUSE_USER', 'default'),
        password=os.getenv('CLICKHOUSE_PASSWORD', ''),
        secure=secure
    )

def init_db():
    """Initializes the ClickHouse schema (MergeTree engine)."""
    client = get_client()

    client.command("""
    CREATE TABLE IF NOT EXISTS files (
        path String,
        bucket_folder String,
        hash String,
        size Int64,
        metadata String
    ) ENGINE = MergeTree()
    ORDER BY path
    """)

    client.command("""
    CREATE TABLE IF NOT EXISTS chunks (
        id String,
        file_path String,
        content String,
        chunk_type String,
        token_count Int32,
        metadata String
    ) ENGINE = MergeTree()
    ORDER BY id
    """)

    client.command("""
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id String,
        embedding Array(Float32)
    ) ENGINE = MergeTree()
    ORDER BY chunk_id
    """)

    client.command("""
    CREATE TABLE IF NOT EXISTS relationships (
        source_path String,
        target_path String,
        source_chunk_id Nullable(String),
        target_chunk_id Nullable(String),
        rel_type String,
        metadata Nullable(String)
    ) ENGINE = MergeTree()
    ORDER BY (source_path, target_path)
    """)
    logger.info("ClickHouse schema initialized.")

# Helper operations for Ingestion

def save_file(path: str, bucket_folder: str, file_hash: str, size: int, metadata: Dict[str, Any]):
    """Saves a file entry. Deletes existing first to mimic UPSERT."""
    client = get_client()
    client.command("ALTER TABLE files DELETE WHERE path = {p:String}", parameters={'p': path})
    
    # Wait for mutation to complete sync (for POC reliability)
    try:
        client.command("SYSTEM SYNC REPLICA files") 
    except Exception:
        pass # Ignore in standalone/cloud if SYNC REPLICA not applicable
        
    client.insert('files', 
                  [[path, bucket_folder, file_hash, size, json.dumps(metadata)]], 
                  column_names=['path', 'bucket_folder', 'hash', 'size', 'metadata'])

def delete_file(path: str):
    """Deletes a file and all cascade elements."""
    client = get_client()
    client.command("ALTER TABLE files DELETE WHERE path = {p:String}", parameters={'p': path})
    client.command("ALTER TABLE chunks DELETE WHERE file_path = {p:String}", parameters={'p': path})
    client.command("ALTER TABLE relationships DELETE WHERE source_path = {p:String} OR target_path = {p:String}", parameters={'p': path})

def delete_chunks_for_file(path: str):
    client = get_client()
    client.command("ALTER TABLE chunks DELETE WHERE file_path = {p:String}", parameters={'p': path})

def save_chunk(chunk_id: str, file_path: str, content: str, chunk_type: str, token_count: int, metadata: Dict[str, Any], embedding: List[float]):
    """Saves a chunk and its corresponding embedding vector."""
    client = get_client()
    client.insert('chunks', 
                  [[chunk_id, file_path, content, chunk_type, token_count, json.dumps(metadata)]], 
                  column_names=['id', 'file_path', 'content', 'chunk_type', 'token_count', 'metadata'])
    
    client.insert('embeddings', 
                  [[chunk_id, embedding]], 
                  column_names=['chunk_id', 'embedding'])

def clear_all_relationships():
    client = get_client()
    client.command("TRUNCATE TABLE relationships")

def save_relationship(source_path: str, target_path: str, source_chunk_id: Optional[str] = None, target_chunk_id: Optional[str] = None, rel_type: str = "references", metadata: Optional[Dict[str, Any]] = None):
    """Saves a relational mapping between two files/chunks."""
    client = get_client()
    meta_str = json.dumps(metadata) if metadata else None
    
    # Avoid duplicates by checking first
    res = client.query("""
        SELECT 1 FROM relationships 
        WHERE source_path = {sp:String} AND target_path = {tp:String} AND rel_type = {rt:String}
        AND (source_chunk_id = {sc:Nullable(String)} OR (source_chunk_id IS NULL AND {sc:Nullable(String)} IS NULL))
        AND (target_chunk_id = {tc:Nullable(String)} OR (target_chunk_id IS NULL AND {tc:Nullable(String)} IS NULL))
        LIMIT 1
    """, parameters={
        'sp': source_path, 'tp': target_path, 'rt': rel_type, 
        'sc': source_chunk_id, 'tc': target_chunk_id
    })
    
    if res.result_rows:
        return
        
    client.insert('relationships', 
                  [[source_path, target_path, source_chunk_id, target_chunk_id, rel_type, meta_str]], 
                  column_names=['source_path', 'target_path', 'source_chunk_id', 'target_chunk_id', 'rel_type', 'metadata'])

def get_all_files() -> List[Dict[str, Any]]:
    client = get_client()
    res = client.query("SELECT * FROM files")
    return [dict(zip(res.column_names, row)) for row in res.result_rows]

def get_file_by_path(path: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    res = client.query("SELECT * FROM files WHERE path = {p:String} LIMIT 1", parameters={'p': path})
    if res.result_rows:
        return dict(zip(res.column_names, res.result_rows[0]))
    return None

def get_all_embeddings() -> List[Tuple[str, List[float]]]:
    """Fallback if needed, though ClickHouse should do native vector search."""
    client = get_client()
    res = client.query("SELECT chunk_id, embedding FROM embeddings")
    return [(row[0], list(row[1])) for row in res.result_rows]

def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    res = client.query("SELECT * FROM chunks WHERE id = {c:String} LIMIT 1", parameters={'c': chunk_id})
    if res.result_rows:
        data = dict(zip(res.column_names, res.result_rows[0]))
        data['metadata'] = json.loads(data['metadata'])
        return data
    return None

def get_relationships_for_file(file_path: str) -> List[Dict[str, Any]]:
    client = get_client()
    res = client.query("SELECT * FROM relationships WHERE source_path = {p:String} OR target_path = {p:String}", parameters={'p': file_path})
    return [dict(zip(res.column_names, row)) for row in res.result_rows]

def get_db_stats() -> Dict[str, int]:
    client = get_client()
    files_count = client.command("SELECT count() FROM files")
    chunks_count = client.command("SELECT count() FROM chunks")
    rels_count = client.command("SELECT count() FROM relationships")
    return {
        "files": int(files_count),
        "chunks": int(chunks_count),
        "relationships": int(rels_count)
    }

def execute_query(query: str, parameters: Dict[str, Any] = None):
    """Executes a raw read query and returns list of dicts. Used for Hybrid Search."""
    client = get_client()
    res = client.query(query, parameters=parameters)
    return [dict(zip(res.column_names, row)) for row in res.result_rows]
