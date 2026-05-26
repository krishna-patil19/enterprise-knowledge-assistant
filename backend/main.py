# FastAPI Application for Enterprise Engineering Knowledge Assistant
# File: backend/main.py

import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json

from backend.retrieval import database
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.search import HybridSearcher
from backend.services import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enterprise Engineering Knowledge Assistant API",
    description="Secure, low-token optimized RAG backend with Python AST and SQL relation indexing.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this. For local POC, allow all.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ingestion pipeline instance
pipeline = IngestionPipeline()
searcher = HybridSearcher()

# Models
class QueryRequest(BaseModel):
    query: str
    limit: int = 5

class IndexResponse(BaseModel):
    status: str
    statistics: dict

@app.get("/health")
async def health_check():
    """Returns system status and database statistics."""
    try:
        stats = database.get_db_stats()
        files = [os.path.basename(f["path"]) for f in database.get_all_files()]
        return {
            "status": "healthy",
            "database_stats": stats,
            "indexed_files": files
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=IndexResponse)
async def trigger_indexing():
    """Forces an incremental folder scan and indexes any new/modified files in the simulated S3 bucket."""
    try:
        # Run indexing in threadpool to prevent blocking the async loop
        loop = asyncio.get_running_loop()
        stats = await loop.run_in_executor(None, pipeline.run_scan_and_index)
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Manual indexing trigger failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def execute_query(payload: QueryRequest):
    """Executes a standard structured RAG query and returns complete metadata and answer."""
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, searcher.perform_rag_query, payload.query, payload.limit)
        return result
    except Exception as e:
        logger.error(f"RAG query execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/stream")
async def execute_query_streaming(payload: QueryRequest):
    """
    Executes a high-performance streaming RAG query.
    First yields RAG metadata (chunks, relationships) as an SSE event,
    then streams the answer text token-by-token.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    async def stream_generator():
        # 1. Execute retrieval, scoping, RRF and expansion synchronously in a thread
        loop = asyncio.get_running_loop()
        try:
            # First, perform retrieval synchronously
            sanitized_query, scoped_folders = await loop.run_in_executor(
                None, lambda: (
                    searcher._fts_keyword_search(payload.query),
                    searcher._detect_query_scopes(payload.query)
                )
            )
            
            # Censor query first
            from backend.security.pii_shield import PIIShield
            clean_query, pii_censored = PIIShield.scan_and_censor(payload.query)
            
            # Hybrid search steps
            keyword_res = await loop.run_in_executor(None, searcher._fts_keyword_search, clean_query)
            vector_res = await loop.run_in_executor(None, searcher._vector_cosine_search, clean_query)
            fused = searcher._reciprocal_rank_fusion(keyword_res, vector_res)
            
            candidates = []
            for cid, score in fused:
                chunk = database.get_chunk_by_id(cid)
                if chunk:
                    f_meta = database.get_file_by_path(chunk["file_path"])
                    folder = f_meta["bucket_folder"] if f_meta else "general"
                    boosted_score = score
                    if folder in scoped_folders:
                        boosted_score *= 1.5
                    candidates.append((chunk, boosted_score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            top_candidates = [c[0] for c in candidates[:20]]
            expanded, relations = await loop.run_in_executor(None, searcher._expand_relationships, top_candidates)
            reranked = await loop.run_in_executor(None, searcher._llm_rerank, clean_query, expanded[:15], payload.limit)
            
            # Yield metadata packet immediately so UI can render retrieved files/graph!
            meta_payload = {
                "event": "metadata",
                "data": {
                    "query": clean_query,
                    "retrieved_chunks": [
                        {
                            "id": c["id"],
                            "file_name": os.path.basename(c["file_path"]),
                            "content": c["content"],
                            "type": c["chunk_type"]
                        }
                        for c in reranked
                    ],
                    "relationships": [
                        {
                            "source": os.path.basename(rel["source_path"]),
                            "target": os.path.basename(rel["target_path"]),
                            "type": rel["rel_type"]
                        }
                        for rel in relations
                    ],
                    "metrics": {
                        "pii_filtered": pii_censored,
                        "folders_scoped": list(scoped_folders)
                    }
                }
            }
            yield f"data: {json.dumps(meta_payload)}\n\n"
            await asyncio.sleep(0.05) # Yield control
            
            # 2. Yield LLM streaming completion
            context_str = ""
            for idx, chunk in enumerate(reranked):
                src_name = os.path.basename(chunk["file_path"])
                context_str += f"[Context Chunk {idx + 1}] (Source: {src_name}, Type: {chunk['chunk_type']})\n"
                context_str += f"{chunk['content']}\n\n"
                
            system_prompt = (
                "You are the Enterprise Engineering Knowledge Assistant, an expert technical agent.\n"
                "Answer the user's questions utilizing ONLY the provided context blocks below.\n"
                "If the answer cannot be found in the context, state that you do not have enough information.\n"
                "Cite the source filenames (e.g. analytics_queries.sql, etl_pipeline.py) in your explanation.\n"
                "Format your answers beautifully in clear Markdown with code syntax highlights."
            )
            
            user_prompt = (
                f"Context Chunks:\n{context_str}\n"
                f"User Question: {clean_query}\n"
                f"Answer:"
            )
            
            def get_stream():
                return LLMService.chat_completion_stream(system_prompt, user_prompt)
                
            # Stream tokens
            for token in get_stream():
                token_payload = {
                    "event": "token",
                    "data": token
                }
                yield f"data: {json.dumps(token_payload)}\n\n"
                await asyncio.sleep(0.005) # Tiny wait for async cooperative yields
                
            # Yield end of stream
            yield "data: {\"event\": \"done\"}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming RAG execution failed: {str(e)}")
            yield f"data: {json.dumps({'event': 'error', 'data': str(e)})}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# Note: Frontend is now served by Streamlit (app.py), not via FastAPI static mount.
# The FastAPI endpoints above remain available for programmatic API access.
