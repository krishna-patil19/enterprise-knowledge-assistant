# Hybrid Retrieval & Reranker Engine for Enterprise Engineering Knowledge Assistant
# File: backend/retrieval/search.py

import os
import logging
import re
from typing import List, Dict, Any, Tuple, Set
from backend.retrieval import database
from backend.services import EmbeddingService, LLMService, OPENAI_API_KEY
from backend.security.pii_shield import PIIShield

logger = logging.getLogger(__name__)

class HybridSearcher:
    """
    Advanced RAG Retrieval Engine.
    Executes: Query PII Scrubbing -> Bucket Scoping -> ClickHouse Keyword Search + ClickHouse Vector Cosine Search ->
    Reciprocal Rank Fusion (RRF) -> Relationship expansion -> LLM Reranking -> Compact context assembly.
    """
    
    def __init__(self):
        # Database schema initialized by pipeline, searcher just reads
        pass

    def perform_rag_query(self, raw_query: str, limit: int = 5) -> Dict[str, Any]:
        """
        Executes the entire retrieval pipeline.
        Returns: {
            "query": sanitized_query,
            "answer": llm_generated_answer,
            "retrieved_chunks": top_chunks,
            "relationships": list_of_expanded_relations,
            "metrics": timing_and_tokens
        }
        """
        import time
        start_time = time.time()
        
        # 1. PII Security Check on Query
        query, pii_censored = PIIShield.scan_and_censor(raw_query)
        if pii_censored:
            logger.info("PII censored in user query.")
            
        # 2. Dynamic Bucket Scoping
        scoped_folders = self._detect_query_scopes(query)
        logger.info(f"Query scoped to folders: {scoped_folders}")
        
        # 3. Parallel Search Execution (Keyword + Vector)
        keyword_results = self._fts_keyword_search(query)
        vector_results = self._vector_cosine_search(query)
        
        # 4. Reciprocal Rank Fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(keyword_results, vector_results, k=60)
        
        # Apply folder scoping boosting/filtering
        scored_candidates = []
        for chunk_id, rrf_score in fused_results:
            chunk = database.get_chunk_by_id(chunk_id)
            if not chunk:
                continue
                
            # Fetch bucket folder for this chunk via parent file
            file_meta = database.get_file_by_path(chunk["file_path"])
            folder = file_meta["bucket_folder"] if file_meta else "general"
            
            # Boost score if chunk matches query scope
            boosted_score = rrf_score
            if folder in scoped_folders:
                boosted_score *= 1.5 # 50% boost for scoped folders
                
            scored_candidates.append((chunk, boosted_score))
            
        # Sort candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = [c[0] for c in scored_candidates[:20]] # Keep top 20 for expansion & reranking
        
        # 5. Relationship Expansion (Step 17 in Architecture)
        expanded_chunks, relations_graph = self._expand_relationships(top_candidates)
        
        # 6. Reranking Layer (Step 18 in Architecture)
        # Narrow down our expanded context chunks to the best 5 using a listwise LLM reranker
        reranked_chunks = self._llm_rerank(query, expanded_chunks[:15], limit=limit)
        
        # 7. Compact Context Prompt Assembly
        context_str = ""
        for idx, chunk in enumerate(reranked_chunks):
            # Display source basename
            src_name = os.path.basename(chunk["file_path"])
            context_str += f"[Context Chunk {idx + 1}] (Source: {src_name}, Type: {chunk['chunk_type']})\n"
            context_str += f"{chunk['content']}\n\n"
            
        # Compile prompts
        system_prompt = (
            "You are the Enterprise Engineering Knowledge Assistant, an expert technical agent.\n"
            "Answer the user's questions utilizing ONLY the provided context blocks below.\n"
            "If the answer cannot be found in the context, state that you do not have enough information.\n"
            "Cite the source filenames (e.g. analytics_queries.sql, etl_pipeline.py) in your explanation.\n"
            "Format your answers beautifully in clear Markdown with code syntax highlights."
        )
        
        user_prompt = (
            f"Context Chunks:\n{context_str}\n"
            f"User Question: {query}\n"
            f"Answer:"
        )
        
        # 8. Bedrock/OpenAI Completion Call
        answer = LLMService.chat_completion(system_prompt, user_prompt)
        
        # Token metrics estimation
        retrieval_time = time.time() - start_time
        prompt_tokens = len(system_prompt + user_prompt) // 4
        answer_tokens = len(answer) // 4
        
        # Assemble file mappings for relationships display in UI
        # We fetch details of related files to render the visual graph!
        ui_relationships = []
        for rel in relations_graph:
            ui_relationships.append({
                "source": os.path.basename(rel["source_path"]),
                "target": os.path.basename(rel["target_path"]),
                "type": rel["rel_type"],
                "source_chunk": rel.get("source_chunk_id"),
                "target_chunk": rel.get("target_chunk_id")
            })
            
        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": [
                {
                    "id": c["id"],
                    "file_name": os.path.basename(c["file_path"]),
                    "content": c["content"],
                    "type": c["chunk_type"],
                    "metadata": c["metadata"]
                }
                for c in reranked_chunks
            ],
            "relationships": ui_relationships,
            "metrics": {
                "duration_seconds": round(retrieval_time, 3),
                "prompt_tokens_est": prompt_tokens,
                "answer_tokens_est": answer_tokens,
                "total_tokens_est": prompt_tokens + answer_tokens,
                "pii_filtered": pii_censored,
                "folders_scoped": list(scoped_folders)
            }
        }

    def _detect_query_scopes(self, query: str) -> Set[str]:
        """Classifies terms in query to identify target folder buckets (Bucket Scoping)."""
        scopes = set()
        q_lower = query.lower()
        
        # SQL scoping triggers
        sql_triggers = ["select", "insert", "aggregatingmergetree", "replacingmergetree", "sumstate", "avgstate", "uniqstate", "summerge", "table", "schema", "join", "database", ".sql"]
        # Python scoping triggers
        py_triggers = ["class", "def", "import", "function", "method", "decorator", "pipeline", "etl", "driver", "client.execute", "python", ".py"]
        # Docs scoping triggers
        docs_triggers = ["guide", "how to", "overview", "documentation", "explanation", "monthly aggregation", "tutorial", "architecture", ".md", ".pdf"]
        
        if any(t in q_lower for t in sql_triggers):
            scopes.add("sql")
            scopes.add("configs") # Configs map tables
        if any(t in q_lower for t in py_triggers):
            scopes.add("python")
        if any(t in q_lower for t in docs_triggers):
            scopes.add("docs")
            scopes.add("pdfs")
            
        # Default scope if nothing is detected is to scan everything
        if not scopes:
            scopes = {"sql", "python", "docs", "configs", "pdfs"}
            
        return scopes

    def _fts_keyword_search(self, query: str) -> List[Tuple[str, float]]:
        """Executes full-text keyword search natively in ClickHouse."""
        results = []
        try:
            clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
            if not clean_query:
                return []
                
            words = [w for w in clean_query.split() if len(w) > 2]
            if not words:
                return []
                
            # Use ClickHouse positionCaseInsensitive
            conditions = " OR ".join([f"positionCaseInsensitive(content, '{w}') > 0" for w in words])
            ch_query = f"SELECT id FROM chunks WHERE {conditions} LIMIT 50"
            rows = database.execute_query(ch_query)
            
            for idx, row in enumerate(rows):
                results.append((row["id"], 100.0 / (idx + 1)))
        except Exception as e:
            logger.warning(f"Keyword search failed: {str(e)}")
        return results

    def _vector_cosine_search(self, query: str) -> List[Tuple[str, float]]:
        """Computes dense vector similarity natively inside ClickHouse Cloud using cosineDistance."""
        query_vector = EmbeddingService.get_embedding(query)
        if not query_vector:
            return []
            
        try:
            # ClickHouse has native cosineDistance which returns 0 for identical vectors, 1 for orthogonal
            # So similarity = 1 - cosineDistance
            ch_query = "SELECT chunk_id, (1 - cosineDistance(embedding, {q:Array(Float32)})) as similarity FROM embeddings ORDER BY similarity DESC LIMIT 50"
            rows = database.execute_query(ch_query, parameters={'q': query_vector})
            
            results = []
            threshold = 0.05 if not OPENAI_API_KEY else 0.1
            for row in rows:
                score = float(row['similarity'])
                if score > threshold:
                    results.append((row['chunk_id'], score))
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _reciprocal_rank_fusion(self, keyword_list: List[Tuple[str, float]], vector_list: List[Tuple[str, float]], k: int = 60) -> List[Tuple[str, float]]:
        """Combines search indexes using Reciprocal Rank Fusion (RRF)."""
        rrf_scores = {}
        
        # Process keyword ranking
        for rank, (chunk_id, _) in enumerate(keyword_list):
            score = 1.0 / (k + (rank + 1))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
            
        # Process vector ranking
        for rank, (chunk_id, _) in enumerate(vector_list):
            score = 1.0 / (k + (rank + 1))
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
            
        # Return sorted list
        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_rrf

    def _expand_relationships(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Crawls the relationship graph to fetch adjacent technical nodes."""
        expanded = list(chunks)
        already_added = {c["id"] for c in chunks}
        relations_pulled = []
        
        try:
            for chunk in chunks:
                chunk_id = chunk["id"]
                file_path = chunk["file_path"]
                
                ch_query = """
                SELECT * FROM relationships 
                WHERE source_chunk_id = {c:String} OR target_chunk_id = {c:String}
                OR (source_path = {p:String} AND source_chunk_id IS NULL)
                OR (target_path = {p:String} AND target_chunk_id IS NULL)
                """
                rows = database.execute_query(ch_query, parameters={'c': chunk_id, 'p': file_path})
                
                for row in rows:
                    if row["source_chunk_id"] and row["source_chunk_id"] not in already_added:
                        c = database.get_chunk_by_id(row["source_chunk_id"])
                        if c:
                            expanded.append(c)
                            already_added.add(c["id"])
                            relations_pulled.append(row)
                    if row["target_chunk_id"] and row["target_chunk_id"] not in already_added:
                        c = database.get_chunk_by_id(row["target_chunk_id"])
                        if c:
                            expanded.append(c)
                            already_added.add(c["id"])
                            relations_pulled.append(row)
        except Exception as e:
            logger.error(f"Relationship expansion failed: {e}")
            
        return expanded, relations_pulled

    def _llm_rerank(self, query: str, chunks: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Utilizes a fast LLM listwise prompt-based reranker to narrow context down.
        Falls back to rank-order if offline or failure.
        """
        if not chunks:
            return []
        if len(chunks) <= limit:
            return chunks
            
        # Build candidate summaries for the LLM
        candidates_str = ""
        for idx, chunk in enumerate(chunks):
            src_name = os.path.basename(chunk["file_path"])
            preview = chunk["content"][:200].replace("\n", " ")
            candidates_str += f"[{idx + 1}] Source: {src_name} | Type: {chunk['chunk_type']} | Content: {preview}...\n"
            
        system_prompt = (
            "You are an expert search engine reranker.\n"
            "Analyze the candidates list relative to the user query and output a comma-separated list of "
            "the 5 most relevant chunk indices, in order of decreasing relevance (highest first).\n"
            "Format: 3, 1, 5, 2, 7\n"
            "Only output the numbers. Do not write explanations."
        )
        
        user_prompt = (
            f"Query: {query}\n\n"
            f"Candidate Chunks:\n{candidates_str}\n\n"
            f"Top indices:"
        )
        
        try:
            # Rerank calls mini model for speed and low cost
            response = LLMService.chat_completion(system_prompt, user_prompt)
            indices = [int(x.strip()) - 1 for x in response.split(",") if x.strip().isdigit()]
            
            # Reconstruct list
            reranked = []
            for idx in indices:
                if 0 <= idx < len(chunks):
                    reranked.append(chunks[idx])
            
            # Append remaining to satisfy limit if LLM output was irregular
            for c in chunks:
                if c not in reranked:
                    reranked.append(c)
                    
            return reranked[:limit]
        except Exception as e:
            logger.warning(f"LLM Rerank failed: {str(e)}. Falling back to vector/FTS order.")
            return chunks[:limit]
