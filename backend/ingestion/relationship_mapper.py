# Relationship Mapping Engine for Enterprise Engineering Knowledge Assistant
# File: backend/ingestion/relationship_mapper.py

import os
import json
import logging
from typing import List, Dict, Any, Optional
from backend.retrieval import database

logger = logging.getLogger(__name__)

class RelationshipMapper:
    """
    Builds structured relationship mappings across parsed chunks in the system.
    Detects linkages like: Code -> SQL, Code -> Config, Documentation -> SQL Table, etc.
    """
    
    @classmethod
    def resolve_and_save_relationships(cls):
        """
        Scans all files and chunks in the database and creates relational links.
        This represents the SQL/Python/docs relationship mapping engine.
        """
        logger.info("Starting relationship mapping scan...")
        
        # Fetch all files
        files = database.get_all_files()
        
        # Load metadata and details for each file
        file_details = []
        for f in files:
            f_path = f["path"]
            f_name = os.path.basename(f_path)
            
            # Fetch all chunks for this file
            rows = database.execute_query("SELECT id, content, metadata FROM chunks WHERE file_path = {p:String}", parameters={'p': f_path})
            chunks = []
            for row in rows:
                chunks.append({
                    "id": row["id"],
                    "content": row["content"],
                    "metadata": json.loads(row["metadata"])
                })
            
            file_details.append({
                "path": f_path,
                "name": f_name,
                "folder": f["bucket_folder"],
                "chunks": chunks,
                "metadata": json.loads(f["metadata"])
            })

        # Clear existing relationships first to compute a fresh graph
        database.clear_all_relationships()

        # Let's perform cross-referencing
        for file_a in file_details:
            path_a = file_a["path"]
            name_a = file_a["name"]
            
            # Extract all tables referenced in File A
            tables_a = set()
            for chunk in file_a["chunks"]:
                meta = chunk["metadata"]
                if "tables" in meta:
                    for t in meta["tables"]:
                        tables_a.add(t.lower())
            
            # Also fetch tables from direct file-level metadata if present
            file_meta = file_a["metadata"]
            if "tables" in file_meta:
                for t in file_meta["tables"]:
                    tables_a.add(t.lower())

            for file_b in file_details:
                path_b = file_b["path"]
                name_b = file_b["name"]
                
                # Skip self-mapping
                if path_a == path_b:
                    continue
                
                # --- Link Category 1: Explicit File Mentions (e.g. Markdown Guide references Python script) ---
                for chunk_a in file_a["chunks"]:
                    content_lower = chunk_a["content"].lower()
                    
                    # If File A explicitly mentions File B's name in its content (like Markdown or docstring)
                    if name_b.lower() in content_lower:
                        rel_type = "documentation" if name_a.endswith((".md", ".rst", ".txt")) else "references_file"
                        database.save_relationship(
                            source_path=path_a,
                            target_path=path_b,
                            source_chunk_id=chunk_a["id"],
                            target_chunk_id=None,
                            rel_type=rel_type,
                            metadata={"reason": f"Chunk explicitly mentions file '{name_b}'"}
                        )
                
                # --- Link Category 2: Shared ClickHouse Tables (e.g. Code reads from table, SQL creates it) ---
                # Check for table overlap
                tables_b = set()
                for chunk_b in file_b["chunks"]:
                    meta_b = chunk_b["metadata"]
                    if "tables" in meta_b:
                        for t in meta_b["tables"]:
                            tables_b.add(t.lower())
                
                shared_tables = tables_a.intersection(tables_b)
                for table in shared_tables:
                    # Let's link the specific chunks that mention this table!
                    for chunk_a in file_a["chunks"]:
                        chunk_tables_a = [t.lower() for t in chunk_a["metadata"].get("tables", [])]
                        if table in chunk_tables_a:
                            for chunk_b in file_b["chunks"]:
                                chunk_tables_b = [t.lower() for t in chunk_b["metadata"].get("tables", [])]
                                if table in chunk_tables_b:
                                    database.save_relationship(
                                        source_path=path_a,
                                        target_path=path_b,
                                        source_chunk_id=chunk_a["id"],
                                        target_chunk_id=chunk_b["id"],
                                        rel_type="references_table",
                                        metadata={"table": table, "reason": f"Both chunks reference the database table '{table}'"}
                                    )

                # --- Link Category 3: Configuration mappings (e.g. YAML config lists Python entrypoint) ---
                if name_a.endswith((".yaml", ".yml", ".json")) and "linked_assets" in file_meta:
                    for asset in file_meta["linked_assets"]:
                        if asset.lower() in path_b.lower() or name_b.lower() in asset.lower():
                            database.save_relationship(
                                source_path=path_a,
                                target_path=path_b,
                                source_chunk_id=None,
                                target_chunk_id=None,
                                rel_type="configures_service",
                                metadata={"reason": f"Config file references service/asset '{asset}'"}
                            )
                
                # --- Link Category 4: Python imports of adjacent Python modules ---
                if name_a.endswith(".py") and name_b.endswith(".py"):
                    name_b_no_ext = os.path.splitext(name_b)[0]
                    # Check imports in file A metadata
                    imports_a = file_meta.get("imports", [])
                    # Also look at chunk level imports
                    for chunk_a in file_a["chunks"]:
                        imports_a.extend(chunk_a["metadata"].get("imports", []))
                    
                    if name_b_no_ext in imports_a:
                        database.save_relationship(
                            source_path=path_a,
                            target_path=path_b,
                            source_chunk_id=None,
                            target_chunk_id=None,
                            rel_type="imports_module",
                            metadata={"module": name_b_no_ext, "reason": f"Python code imports module '{name_b_no_ext}'"}
                        )
                        
        logger.info("Relationship mapping completed successfully.")
