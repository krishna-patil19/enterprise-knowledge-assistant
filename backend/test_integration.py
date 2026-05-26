# Automated Integration and Retrieval Tests for RAG Assistant
# File: backend/test_integration.py

import os
import sys
import logging
import json

# Ensure parent directory is in python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.retrieval import database
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.search import HybridSearcher
from backend.security.pii_shield import PIIShield

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_integration")

def run_integration_tests():
    logger.info("==============================================")
    logger.info("STARTING RAG ASSISTANT INTEGRATION TESTS")
    logger.info("==============================================")
    
    # 1. Initialize DB & Pipeline
    logger.info("Step 1: Initializing SQLite database and scanning S3 folders...")
    pipeline = IngestionPipeline()
    stats = pipeline.run_scan_and_index()
    logger.info(f"Ingestion scan completed. Stats: {json.dumps(stats, indent=2)}")
    
    # Verify entity counts
    db_stats = database.get_db_stats()
    logger.info(f"Database stats: {json.dumps(db_stats, indent=2)}")
    
    assert db_stats["files"] >= 4, "Should have indexed at least 4 files (sql, python, md, yaml)."
    assert db_stats["chunks"] >= 5, "Should have generated multiple structure-aware chunks."
    assert db_stats["relationships"] >= 1, "Should have mapped relationships between technical assets."
    logger.info("✅ SQLite Database Ingestion Verification: SUCCESS")
    
    # 2. Verify Parsers & Metadata
    logger.info("\nStep 2: Inspecting specialized AST & SQL parser extractions...")
    files = database.get_all_files()
    for f in files:
        f_name = os.path.basename(f["path"])
        f_meta = json.loads(f["metadata"])
        logger.info(f"File: {f_name} | Folder: {f['bucket_folder']} | Metadata Tables: {f_meta.get('tables')}")
        
    # Verify etl_pipeline.py chunks
    conn = database.get_connection()
    cursor = conn.execute("SELECT id, chunk_type, metadata FROM chunks WHERE file_path LIKE '%etl_pipeline.py%';")
    py_chunks = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    logger.info(f"Found {len(py_chunks)} chunks for etl_pipeline.py:")
    for chunk in py_chunks:
        meta = json.loads(chunk["metadata"])
        logger.info(f"  Chunk ID: {chunk['id']} | Type: {meta.get('type')} | Table references: {meta.get('tables')}")
        
    # Verify that AST parser successfully extracted classes/functions
    has_class_chunk = any(json.loads(c["metadata"]).get("type") == "class" for c in py_chunks)
    has_func_chunk = any(json.loads(c["metadata"]).get("type") == "function" for c in py_chunks)
    assert has_class_chunk or has_func_chunk, "AST parser failed to split code by class/function bounds!"
    logger.info("✅ Structure-Aware Parsing Verification: SUCCESS")
    
    # 3. Verify Relationship Mapping Graph
    logger.info("\nStep 3: Verifying computed relationship graphs...")
    relations = database.get_relationships_for_file(files[0]["path"])
    logger.info(f"Found {len(relations)} active linkages in the database:")
    for rel in relations:
        logger.info(f"  Link: {os.path.basename(rel['source_path'])} --[{rel['rel_type']}]--> {os.path.basename(rel['target_path'])}")
        
    assert len(relations) > 0, "No cross-references were mapped by the relationship engine!"
    logger.info("✅ Relationship Mapping verification: SUCCESS")
    
    # 4. Verify PII Shield Protection
    logger.info("\nStep 4: Testing PII and Secret protection scanner...")
    raw_user_query = "Who uses sumState? My personal email is dev.jane@customer-data.com and my OpenAI key is sk-1234567890abcdef1234567890abcdef1234567890"
    censored_query, was_censored = PIIShield.scan_and_censor(raw_user_query)
    
    logger.info(f"Raw query: {raw_user_query}")
    logger.info(f"Sanitized query: {censored_query}")
    
    assert was_censored, "PII scanner failed to detect email or secret!"
    assert "dev.jane@customer-data.com" not in censored_query, "Email was not redacted!"
    assert "sk-1234567890" not in censored_query, "OpenAI secret was not redacted!"
    logger.info("✅ PII Shield Security Verification: SUCCESS")
    
    # 5. Verify Retrieval Pipeline and Scoping
    logger.info("\nStep 5: Testing RAG Query Pipeline & Hybrid Retrieval...")
    searcher = HybridSearcher()
    rag_response = searcher.perform_rag_query("Where is sumState used?")
    
    logger.info(f"Query: {rag_response['query']}")
    logger.info(f"Time Taken: {rag_response['metrics']['duration_seconds']}s")
    logger.info(f"Folders scoped: {rag_response['metrics']['folders_scoped']}")
    logger.info(f"Retrieved {len(rag_response['retrieved_chunks'])} relevant source chunks:")
    for chunk in rag_response["retrieved_chunks"]:
        logger.info(f"  - Chunk: {chunk['id']} (Source: {chunk['file_name']})")
        
    logger.info(f"\nEngine Answer:\n{rag_response['answer']}")
    
    assert len(rag_response["retrieved_chunks"]) > 0, "No chunks retrieved!"
    assert "analytics_queries.sql" in [c["file_name"] for c in rag_response["retrieved_chunks"]], "Failed to retrieve the ClickHouse SQL queries!"
    logger.info("✅ Hybrid Scoping, Retrieval and Reranking Verification: SUCCESS")
    
    logger.info("\n==============================================")
    logger.info("ALL INTEGRATION TESTS COMPLETED SUCCESSFULLY!")
    logger.info("==============================================")

if __name__ == "__main__":
    run_integration_tests()
