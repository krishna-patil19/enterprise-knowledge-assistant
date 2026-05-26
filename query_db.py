import sqlite3
import os

conn = sqlite3.connect('backend/knowledge_assistant.db')
conn.row_factory = sqlite3.Row

print("=" * 70)
print("  SQLite DATABASE CONTENTS")
print("=" * 70)

# FILES
print("\n--- FILES ---")
for r in conn.execute("SELECT path, bucket_folder, hash FROM files"):
    print(f"  [{r['bucket_folder']:8s}] {os.path.basename(r['path'])}")

# CHUNKS
print("\n--- ALL CHUNKS ---")
for r in conn.execute("SELECT id, file_path, chunk_type, content FROM chunks"):
    fname = os.path.basename(r['file_path'])
    preview = r['content'][:100].replace('\n', ' ')
    print(f"  {r['id']:55s} | {fname:30s} | {r['chunk_type']:12s}")
    print(f"    \"{preview}...\"")
    print()

# EMBEDDINGS
count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
print(f"--- EMBEDDINGS: {count} vectors stored ---\n")

# RELATIONSHIPS
print("--- RELATIONSHIPS ---")
for r in conn.execute("SELECT source_path, target_path, source_chunk_id, target_chunk_id, rel_type, metadata FROM relationships"):
    src = os.path.basename(r['source_path'])
    tgt = os.path.basename(r['target_path'])
    print(f"  {src:30s} --[{r['rel_type']}]--> {tgt}")
    if r['source_chunk_id']:
        print(f"    src_chunk: {r['source_chunk_id']}")
    if r['target_chunk_id']:
        print(f"    tgt_chunk: {r['target_chunk_id']}")

conn.close()
