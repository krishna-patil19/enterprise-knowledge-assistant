# Database Layer for Enterprise Engineering Knowledge Assistant
# File: backend/retrieval/database.py

import sqlite3
import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_assistant.db")

def get_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row_factory for dict-like access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db():
    """Initializes the SQLite schema with all required tables."""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS files (
        path TEXT PRIMARY KEY,
        bucket_folder TEXT NOT NULL,
        hash TEXT NOT NULL,
        size INTEGER NOT NULL,
        metadata TEXT DEFAULT '{}'
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        content TEXT NOT NULL,
        chunk_type TEXT NOT NULL,
        token_count INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}',
        FOREIGN KEY (file_path) REFERENCES files(path)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id TEXT PRIMARY KEY,
        embedding TEXT NOT NULL,
        FOREIGN KEY (chunk_id) REFERENCES chunks(id)
    );
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS relationships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_path TEXT NOT NULL,
        target_path TEXT NOT NULL,
        source_chunk_id TEXT,
        target_chunk_id TEXT,
        rel_type TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    );
    """)
    
    # FTS5 virtual table for full-text keyword search
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
        chunk_id, content, tokenize='porter'
    );
    """)
    
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized at %s", DB_PATH)

# --- File Operations ---

def save_file(path: str, bucket_folder: str, file_hash: str, size: int, metadata: Dict[str, Any]):
    """Saves or updates a file entry."""
    conn = get_connection()
    conn.execute("""
    INSERT OR REPLACE INTO files (path, bucket_folder, hash, size, metadata)
    VALUES (?, ?, ?, ?, ?);
    """, (path, bucket_folder, file_hash, size, json.dumps(metadata)))
    conn.commit()
    conn.close()

def delete_file(path: str):
    """Deletes a file and all cascade elements."""
    conn = get_connection()
    conn.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?);", (path,))
    conn.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?);", (path,))
    conn.execute("DELETE FROM chunks WHERE file_path = ?;", (path,))
    conn.execute("DELETE FROM relationships WHERE source_path = ? OR target_path = ?;", (path, path))
    conn.execute("DELETE FROM files WHERE path = ?;", (path,))
    conn.commit()
    conn.close()

def delete_chunks_for_file(path: str):
    """Deletes chunks and their embeddings for a given file path."""
    conn = get_connection()
    conn.execute("DELETE FROM embeddings WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?);", (path,))
    conn.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?);", (path,))
    conn.execute("DELETE FROM chunks WHERE file_path = ?;", (path,))
    conn.commit()
    conn.close()

def get_all_files() -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM files;")
    files = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return files

def get_file_by_path(path: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM files WHERE path = ?;", (path,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

# --- Chunk Operations ---

def save_chunk(chunk_id: str, file_path: str, content: str, chunk_type: str, token_count: int, metadata: Dict[str, Any], embedding: List[float]):
    """Saves a chunk, its FTS index entry, and its embedding vector."""
    conn = get_connection()
    conn.execute("""
    INSERT OR REPLACE INTO chunks (id, file_path, content, chunk_type, token_count, metadata)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (chunk_id, file_path, content, chunk_type, token_count, json.dumps(metadata)))
    
    # FTS5 index
    conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?;", (chunk_id,))
    conn.execute("INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?);", (chunk_id, content))
    
    # Embedding vector (stored as JSON string)
    conn.execute("""
    INSERT OR REPLACE INTO embeddings (chunk_id, embedding)
    VALUES (?, ?);
    """, (chunk_id, json.dumps(embedding)))
    
    conn.commit()
    conn.close()

def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM chunks WHERE id = ?;", (chunk_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        data = dict(row)
        data['metadata'] = json.loads(data['metadata'])
        return data
    return None

def get_all_embeddings() -> List[Tuple[str, List[float]]]:
    """Returns all embeddings as (chunk_id, vector) pairs."""
    conn = get_connection()
    cursor = conn.execute("SELECT chunk_id, embedding FROM embeddings;")
    results = []
    for row in cursor.fetchall():
        try:
            vec = json.loads(row["embedding"])
            results.append((row["chunk_id"], vec))
        except (json.JSONDecodeError, TypeError):
            continue
    conn.close()
    return results

# --- Relationship Operations ---

def clear_all_relationships():
    conn = get_connection()
    conn.execute("DELETE FROM relationships;")
    conn.commit()
    conn.close()

def save_relationship(source_path: str, target_path: str, source_chunk_id: Optional[str] = None, target_chunk_id: Optional[str] = None, rel_type: str = "references", metadata: Optional[Dict[str, Any]] = None):
    """Saves a relational mapping between two files/chunks."""
    conn = get_connection()
    meta_str = json.dumps(metadata) if metadata else '{}'
    
    # Check for duplicate
    cursor = conn.execute("""
        SELECT 1 FROM relationships
        WHERE source_path = ? AND target_path = ? AND rel_type = ?
        AND (source_chunk_id = ? OR (source_chunk_id IS NULL AND ? IS NULL))
        AND (target_chunk_id = ? OR (target_chunk_id IS NULL AND ? IS NULL))
        LIMIT 1;
    """, (source_path, target_path, rel_type, source_chunk_id, source_chunk_id, target_chunk_id, target_chunk_id))
    
    if cursor.fetchone():
        conn.close()
        return
    
    conn.execute("""
    INSERT INTO relationships (source_path, target_path, source_chunk_id, target_chunk_id, rel_type, metadata)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (source_path, target_path, source_chunk_id, target_chunk_id, rel_type, meta_str))
    conn.commit()
    conn.close()

def get_relationships_for_file(file_path: str) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.execute("SELECT * FROM relationships WHERE source_path = ? OR target_path = ?;", (file_path, file_path))
    rels = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rels

# --- Stats ---

def get_db_stats() -> Dict[str, int]:
    conn = get_connection()
    files_count = conn.execute("SELECT COUNT(*) FROM files;").fetchone()[0]
    chunks_count = conn.execute("SELECT COUNT(*) FROM chunks;").fetchone()[0]
    rels_count = conn.execute("SELECT COUNT(*) FROM relationships;").fetchone()[0]
    conn.close()
    return {
        "files": files_count,
        "chunks": chunks_count,
        "relationships": rels_count
    }
