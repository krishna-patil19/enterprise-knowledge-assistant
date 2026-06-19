# Streamlit Frontend for Enterprise Engineering Knowledge Assistant
# File: app.py
# Launch: streamlit run app.py

import os
import sys
import time
import streamlit as st

# Ensure project root is on sys.path for backend imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.retrieval import database
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrieval.search import HybridSearcher
from backend.services import LLMService
from backend.security.pii_shield import PIIShield

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Engineering Knowledge Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean, dark, kava.ai-inspired
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root overrides */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* Sidebar styling — clean soft gray */
    section[data-testid="stSidebar"] {
        background-color: #f7f8fa;
        border-right: 1px solid #e2e5ea;
    }
    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        color: #1a1a2e;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption {
        color: #6b7280;
    }

    /* File badges */
    .file-badge {
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 700;
        text-transform: uppercase;
        padding: 2px 7px;
        border-radius: 4px;
        margin-right: 6px;
        letter-spacing: 0.4px;
    }
    .badge-sql { background: #e0f4ff; color: #0077b6; }
    .badge-python { background: #f0e6ff; color: #7c3aed; }
    .badge-docs { background: #e6fbef; color: #16a34a; }
    .badge-configs { background: #fff7e0; color: #d97706; }
    .badge-pdf { background: #ffe6e6; color: #dc2626; }

    /* Source citation cards */
    .source-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
        font-size: 0.82rem;
    }
    .source-card:hover {
        border-color: #4F8BF9;
        background: #f0f5ff;
    }
    .source-header {
        font-weight: 600;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 6px;
        color: #1a1a2e;
    }

    /* Stats metric cards */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 10px;
    }
    div[data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-weight: 700;
    }

    /* Main chat area */
    div[data-testid="stChatMessage"] {
        border-radius: 12px;
    }

    /* Example query buttons */
    div.stButton > button {
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        background: #ffffff;
        color: #374151;
        text-align: left;
        padding: 14px 16px;
        font-size: 0.82rem;
        transition: all 0.2s ease;
        line-height: 1.5;
    }
    div.stButton > button:hover {
        border-color: #4F8BF9;
        background: #f0f5ff;
        color: #1a1a2e;
    }

    /* Sidebar re-index button */
    section[data-testid="stSidebar"] div.stButton > button {
        background: #4F8BF9;
        color: white;
        border: none;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #3b7ae0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pipeline" not in st.session_state:
    st.session_state.pipeline = IngestionPipeline()
if "searcher" not in st.session_state:
    st.session_state.searcher = HybridSearcher()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_file_badge(filename: str) -> str:
    """Returns an HTML badge for the file type."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    badge_map = {
        "sql": ("SQL", "badge-sql"),
        "py": ("PYTHON", "badge-python"),
        "md": ("DOCS", "badge-docs"),
        "yaml": ("CONFIG", "badge-configs"),
        "yml": ("CONFIG", "badge-configs"),
        "json": ("CONFIG", "badge-configs"),
        "pdf": ("PDF", "badge-pdf"),
    }
    label, cls = badge_map.get(ext, ("FILE", "badge-docs"))
    return f'<span class="file-badge {cls}">{label}</span>'


def route_uploaded_file(filename: str) -> str:
    """Determines the data subfolder for an uploaded file by extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    folder_map = {
        "sql": "sql",
        "py": "python",
        "md": "docs",
        "yaml": "configs",
        "yml": "configs",
        "json": "configs",
        "pdf": "pdfs",
    }
    return folder_map.get(ext, "docs")


def stream_rag_answer(query: str):
    """
    Generator that performs the full RAG pipeline and yields answer tokens.
    Also stores metadata (chunks, metrics) into session state for display after streaming.
    """
    searcher = st.session_state.searcher

    # 1. PII Security Check
    clean_query, pii_censored = PIIShield.scan_and_censor(query)

    # 2. Scoping
    scoped_folders = searcher._detect_query_scopes(clean_query)

    # 3. Hybrid Search
    keyword_res = searcher._fts_keyword_search(clean_query)
    vector_res = searcher._vector_cosine_search(clean_query)
    fused = searcher._reciprocal_rank_fusion(keyword_res, vector_res)

    # 4. Scope boosting + candidate assembly
    candidates = []
    for cid, score in fused:
        chunk = database.get_chunk_by_id(cid)
        if chunk:
            f_meta = database.get_file_by_path(chunk["file_path"])
            folder = f_meta["bucket_folder"] if f_meta else "general"
            boosted_score = score * 1.5 if folder in scoped_folders else score
            candidates.append((chunk, boosted_score))
    candidates.sort(key=lambda x: x[1], reverse=True)

    top_candidates = [c[0] for c in candidates[:20]]

    # 5. Relationship expansion
    expanded, relations = searcher._expand_relationships(top_candidates)

    # 6. Rerank
    reranked = searcher._llm_rerank(clean_query, expanded[:15], limit=5)

    # Store metadata for display after streaming
    st.session_state._last_chunks = reranked
    st.session_state._last_pii = pii_censored
    st.session_state._last_scopes = list(scoped_folders)
    st.session_state._last_relations = relations

    # 7. Build context prompt
    context_str = ""
    for idx, chunk in enumerate(reranked):
        src_name = os.path.basename(chunk["file_path"])
        context_str += f"[Context Chunk {idx + 1}] (Source: {src_name}, Type: {chunk['chunk_type']})\n"
        context_str += f"{chunk['content']}\n\n"

    system_prompt = (
        "You are the Engineering Knowledge Assistant, an expert technical agent.\n"
        "Answer the user's questions utilizing ONLY the provided context blocks below.\n"
        "If the answer cannot be found in the context, state that you do not have enough information.\n"
        "Cite the source filenames (e.g. analytics_queries.sql, etl_pipeline.py) in your explanation.\n"
        "Format your answers in clear Markdown with code syntax highlights when appropriate."
    )

    user_prompt = (
        f"Context Chunks:\n{context_str}\n"
        f"User Question: {clean_query}\n"
        f"Answer:"
    )

    # 8. Stream LLM tokens
    for token in LLMService.chat_completion_stream(system_prompt, user_prompt):
        yield token


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🔍 Knowledge Assistant")

    st.divider()

    # --- S3 Bucket Selector (when using S3 mode) ---
    ingestion_source = st.session_state.pipeline.ingestion_source
    if ingestion_source == "s3":
        available_buckets = st.session_state.pipeline.available_buckets
        if len(available_buckets) > 1:
            st.markdown("### 🪣 S3 Bucket")
            current_bucket = st.session_state.pipeline.bucket_name
            selected_bucket = st.selectbox(
                "Select active bucket",
                options=available_buckets,
                index=available_buckets.index(current_bucket) if current_bucket in available_buckets else 0,
                label_visibility="collapsed",
            )
            if selected_bucket != st.session_state.pipeline.bucket_name:
                st.session_state.pipeline.set_bucket(selected_bucket)
                st.rerun()
            st.divider()
        else:
            st.caption(f"📦 Bucket: `{st.session_state.pipeline.bucket_name}`")
            st.divider()

    # --- File Upload ---
    st.markdown("### 📄 Upload Documents")
    
    if ingestion_source == "github":
        github_repo = st.session_state.pipeline.github_repo_url
        st.info(f"**GitHub Sync Enabled**\n\nApp is synced with repository:\n`{github_repo}`\n\nClick **Re-index** to fetch latest commits. File uploads are disabled in GitHub mode.")
    else:
        uploaded_files = st.file_uploader(
            "Drop files here to add to the knowledge base",
            type=["sql", "py", "md", "yaml", "yml", "json", "pdf", "txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
    
        if uploaded_files:
            new_files = []
            
            if ingestion_source == "s3":
                bucket_name = st.session_state.pipeline.bucket_name
                s3_client = st.session_state.pipeline._get_s3_client()
                for uf in uploaded_files:
                    dest_folder = route_uploaded_file(uf.name)
                    s3_key = f"{dest_folder}/{uf.name}" if dest_folder != "general" else uf.name
                    file_bytes = uf.read()
                    
                    upload_needed = True
                    try:
                        head = s3_client.head_object(Bucket=bucket_name, Key=s3_key)
                        s3_etag = head['ETag'].strip('"')
                        import hashlib
                        local_md5 = hashlib.md5(file_bytes).hexdigest()
                        if s3_etag == local_md5 and head['ContentLength'] == len(file_bytes):
                            upload_needed = False
                    except Exception:
                        pass
                        
                    if upload_needed:
                        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=file_bytes)
                        new_files.append(uf.name)
            else:
                s3_root = st.session_state.pipeline.s3_root
                for uf in uploaded_files:
                    dest_folder = route_uploaded_file(uf.name)
                    dest_dir = os.path.join(s3_root, dest_folder)
                    os.makedirs(dest_dir, exist_ok=True)
                    dest_path = os.path.join(dest_dir, uf.name)
    
                    # Only write if file is new or changed
                    file_bytes = uf.read()
                    write_needed = True
                    if os.path.exists(dest_path):
                        with open(dest_path, "rb") as existing:
                            if existing.read() == file_bytes:
                                write_needed = False
    
                    if write_needed:
                        with open(dest_path, "wb") as f:
                            f.write(file_bytes)
                        new_files.append(uf.name)
    
            if new_files:
                with st.spinner(f"Indexing {len(new_files)} new file(s)..."):
                    stats = st.session_state.pipeline.run_scan_and_index()
                st.success(f"✅ Indexed {stats['indexed']} file(s), {stats['skipped']} unchanged")

    st.divider()

    # --- Indexed Files ---
    st.markdown("### 📚 Indexed Documents")

    all_files = database.get_all_files()
    if all_files:
        for f in all_files:
            fname = os.path.basename(f["path"])
            badge_html = get_file_badge(fname)
            st.markdown(f'{badge_html} `{fname}`', unsafe_allow_html=True)
    else:
        st.caption("No documents indexed yet. Upload files above or click re-index.")

    st.divider()

    # --- Re-index Button ---
    if st.button("🔄 Re-index All Documents", use_container_width=True):
        with st.spinner("Scanning data/ and re-indexing..."):
            stats = st.session_state.pipeline.run_scan_and_index()
        st.success(
            f"Scanned: {stats['scanned']} · "
            f"Indexed: {stats['indexed']} · "
            f"Skipped: {stats['skipped']}"
        )
        st.rerun()

    # --- DB Stats ---
    st.divider()
    db_stats = database.get_db_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Files", db_stats["files"])
    c2.metric("Chunks", db_stats["chunks"])
    c3.metric("Links", db_stats["relationships"])


# ---------------------------------------------------------------------------
# MAIN CHAT AREA
# ---------------------------------------------------------------------------

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show source citations for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) referenced", expanded=False):
                for src in msg["sources"]:
                    src_name = os.path.basename(src["file_path"])
                    badge = get_file_badge(src_name)
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-header">{badge} {src_name}</div>'
                        f'<code style="font-size:0.75rem; color:#9ca3af;">{src["chunk_type"]} - Segment Match</code>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.code(src["content"], language=src["chunk_type"] if src["chunk_type"] in ("sql", "python") else None)
                    
                    # Provide full file viewer
                    with st.expander(f"📄 View Full Original File: {src_name}"):
                        try:
                            file_path = src["file_path"]
                            if file_path.startswith("s3://"):
                                path_no_scheme = file_path[5:]
                                parts = path_no_scheme.split('/', 1)
                                bucket = parts[0]
                                key = parts[1] if len(parts) > 1 else ''
                                
                                s3_client = st.session_state.pipeline._get_s3_client()
                                obj = s3_client.get_object(Bucket=bucket, Key=key)
                                full_content = obj['Body'].read().decode("utf-8", errors="ignore")
                            elif file_path.startswith("github://"):
                                path_no_scheme = file_path[9:]
                                parts = path_no_scheme.split('/', 1)
                                key = parts[1] if len(parts) > 1 else ''
                                
                                base_dir = os.path.dirname(os.path.abspath(__file__))
                                local_path = os.path.join(base_dir, ".temp_github_ingest", key)
                                
                                with open(local_path, "r", encoding="utf-8") as f:
                                    full_content = f.read()
                            else:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    full_content = f.read()
                            st.code(full_content, language=src["chunk_type"] if src["chunk_type"] in ("sql", "python") else "markdown")
                        except Exception as e:
                            st.info(f"Could not load file details: {str(e)}")

        if msg["role"] == "assistant" and msg.get("relations"):
            with st.expander("🕸️ Knowledge Graph Context", expanded=False):
                dot_lines = [
                    "digraph G {",
                    '  rankdir=LR;',
                    '  node [shape=box, style=filled, fillcolor="#f0f5ff", color="#4F8BF9", fontname="sans-serif", fontsize=12, penwidth=2];',
                    '  edge [fontname="sans-serif", fontsize=10, color="#9ca3af"];'
                ]
                
                for rel in msg["relations"]:
                    src = os.path.basename(rel["source_path"])
                    tgt = os.path.basename(rel["target_path"])
                    label = rel["rel_type"]
                    dot_lines.append(f'  "{src}" -> "{tgt}" [label="{label}"];')
                    
                dot_lines.append("}")
                st.graphviz_chart("\n".join(dot_lines))

# Welcome state when no messages
if not st.session_state.messages:
    st.markdown("")
    st.markdown("#### Try an example question:")

    examples = [
        "Where is sumState used?",
        "How is raw sales data loaded into ClickHouse?",
        "What partitions are used in sales_aggregates?",
        "Explain the analytics config structure",
    ]

    cols = st.columns(2)
    for idx, q in enumerate(examples):
        with cols[idx % 2]:
            if st.button(q, key=f"example_{idx}", use_container_width=True):
                st.session_state._pending_query = q
                st.rerun()

# Handle example button clicks
if hasattr(st.session_state, "_pending_query") and st.session_state._pending_query:
    query = st.session_state._pending_query
    st.session_state._pending_query = None

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Stream assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(stream_rag_answer(query))

    # Build sources list from last retrieval
    sources = []
    if hasattr(st.session_state, "_last_chunks") and st.session_state._last_chunks:
        sources = [
            {
                "file_path": c["file_path"],
                "content": c["content"],
                "chunk_type": c["chunk_type"],
            }
            for c in st.session_state._last_chunks
        ]

    relations = st.session_state.get("_last_relations", [])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
        "relations": relations,
    })
    st.rerun()

# Chat input
if prompt := st.chat_input("Ask about your engineering documentation..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response
    with st.chat_message("assistant"):
        response = st.write_stream(stream_rag_answer(prompt))

    # Build sources list
    sources = []
    if hasattr(st.session_state, "_last_chunks") and st.session_state._last_chunks:
        sources = [
            {
                "file_path": c["file_path"],
                "content": c["content"],
                "chunk_type": c["chunk_type"],
            }
            for c in st.session_state._last_chunks
        ]

    relations = st.session_state.get("_last_relations", [])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources,
        "relations": relations,
    })
    st.rerun()
