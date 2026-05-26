"""
Interactive Backend RAG Pipeline Tester
Shows every step of the retrieval pipeline with detailed output.
"""
import sys
import os
import time
import json
import re

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.retrieval import database
from backend.retrieval.search import HybridSearcher
from backend.services import EmbeddingService, OPENAI_API_KEY
from backend.security.pii_shield import PIIShield

def separator(title):
    print(f"\n{'='*60}")
    print(f"  STEP: {title}")
    print(f"{'='*60}")

def run_query(raw_query: str):
    print(f"\n{'#'*60}")
    print(f"  RAG PIPELINE - FULL BACKEND TRACE")
    print(f"  Query: \"{raw_query}\"")
    print(f"{'#'*60}")
    
    start = time.time()
    
    # --- Step 1: PII Scrubbing ---
    separator("1. PII Security Scan")
    cleaned_query, pii_found = PIIShield.scan_and_censor(raw_query)
    print(f"  Original query : {raw_query}")
    print(f"  Cleaned query  : {cleaned_query}")
    print(f"  PII detected?  : {pii_found}")
    
    # --- Step 2: Bucket Scoping ---
    separator("2. Dynamic Bucket Scoping")
    searcher = HybridSearcher()
    scopes = searcher._detect_query_scopes(cleaned_query)
    print(f"  Scoped folders : {scopes}")
    
    # --- Step 3a: FTS5 Keyword Search ---
    separator("3a. FTS5 Keyword Search (BM25)")
    t1 = time.time()
    keyword_results = searcher._fts_keyword_search(cleaned_query)
    t2 = time.time()
    print(f"  Results found  : {len(keyword_results)}")
    print(f"  Time           : {(t2-t1)*1000:.1f}ms")
    for chunk_id, score in keyword_results[:5]:
        chunk = database.get_chunk_by_id(chunk_id)
        fname = os.path.basename(chunk["file_path"]) if chunk else "?"
        preview = chunk["content"][:80].replace("\n", " ") if chunk else "?"
        print(f"    [{score:6.1f}] {fname} | {chunk_id}")
        print(f"            \"{preview}...\"")
    
    # --- Step 3b: Vector Cosine Search ---
    separator("3b. Vector Cosine Similarity Search (NumPy)")
    t1 = time.time()
    vector_results = searcher._vector_cosine_search(cleaned_query)
    t2 = time.time()
    print(f"  Results found  : {len(vector_results)}")
    print(f"  Time           : {(t2-t1)*1000:.1f}ms")
    for chunk_id, score in vector_results[:5]:
        chunk = database.get_chunk_by_id(chunk_id)
        fname = os.path.basename(chunk["file_path"]) if chunk else "?"
        preview = chunk["content"][:80].replace("\n", " ") if chunk else "?"
        print(f"    [{score:.4f}] {fname} | {chunk_id}")
        print(f"            \"{preview}...\"")
    
    # --- Step 4: Reciprocal Rank Fusion ---
    separator("4. Reciprocal Rank Fusion (RRF)")
    fused = searcher._reciprocal_rank_fusion(keyword_results, vector_results, k=60)
    print(f"  Fused results  : {len(fused)}")
    for chunk_id, score in fused[:5]:
        chunk = database.get_chunk_by_id(chunk_id)
        fname = os.path.basename(chunk["file_path"]) if chunk else "?"
        print(f"    [RRF {score:.5f}] {fname} | {chunk_id}")
    
    # --- Step 5: Scope Boosting ---
    separator("5. Scope Boosting + Candidate Selection")
    scored_candidates = []
    for chunk_id, rrf_score in fused:
        chunk = database.get_chunk_by_id(chunk_id)
        if not chunk:
            continue
        file_meta = database.get_file_by_path(chunk["file_path"])
        folder = file_meta["bucket_folder"] if file_meta else "general"
        boosted = rrf_score * (1.5 if folder in scopes else 1.0)
        scored_candidates.append((chunk, boosted, folder))
    
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = [c[0] for c in scored_candidates[:10]]
    print(f"  Top candidates : {len(top_candidates)}")
    for chunk, score, folder in scored_candidates[:5]:
        fname = os.path.basename(chunk["file_path"])
        boosted_tag = " [BOOSTED]" if folder in scopes else ""
        print(f"    [{score:.5f}] {fname} ({folder}){boosted_tag} | {chunk['id']}")
    
    # --- Step 6: Relationship Expansion ---
    separator("6. Relationship Graph Expansion")
    expanded, relations = searcher._expand_relationships(top_candidates)
    new_chunks = len(expanded) - len(top_candidates)
    print(f"  Original chunks: {len(top_candidates)}")
    print(f"  After expansion: {len(expanded)} (+{new_chunks} from relationships)")
    print(f"  Relations used : {len(relations)}")
    for rel in relations[:5]:
        src = os.path.basename(rel["source_path"])
        tgt = os.path.basename(rel["target_path"])
        print(f"    {src} --[{rel['rel_type']}]--> {tgt}")
    
    # --- Step 7: LLM Reranking ---
    separator("7. LLM Listwise Reranking (GPT-4o)")
    t1 = time.time()
    reranked = searcher._llm_rerank(cleaned_query, expanded[:15], limit=5)
    t2 = time.time()
    print(f"  Final top-5    : {len(reranked)}")
    print(f"  Rerank time    : {(t2-t1)*1000:.1f}ms")
    for idx, chunk in enumerate(reranked):
        fname = os.path.basename(chunk["file_path"])
        preview = chunk["content"][:100].replace("\n", " ")
        print(f"    #{idx+1} {fname} ({chunk['chunk_type']}) | {chunk['id']}")
        print(f"        \"{preview}...\"")
    
    # --- Step 8: Context Assembly & LLM Answer ---
    separator("8. Final LLM Answer Generation")
    print("  Assembling context prompt and calling GPT-4o...")
    
    # Run the full pipeline to get the answer
    result = searcher.perform_rag_query(raw_query, limit=5)
    
    print(f"\n  --- ANSWER ---")
    print(f"  {result['answer']}")
    
    # --- Metrics ---
    separator("METRICS")
    m = result["metrics"]
    total_time = time.time() - start
    print(f"  Total pipeline time  : {total_time:.2f}s")
    print(f"  Prompt tokens (est)  : {m['prompt_tokens_est']}")
    print(f"  Answer tokens (est)  : {m['answer_tokens_est']}")
    print(f"  PII filtered         : {m['pii_filtered']}")
    print(f"  Folders scoped       : {m['folders_scoped']}")


if __name__ == "__main__":
    database.init_db()
    
    while True:
        print("\n" + "-"*60)
        query = input("Enter your question (or 'quit' to exit): ").strip()
        if query.lower() in ('quit', 'exit', 'q'):
            print("Goodbye!")
            break
        if not query:
            continue
        run_query(query)
