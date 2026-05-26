import os
import sqlite3
import json

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "knowledge_assistant.db")

def inspect_db():
    if not os.path.exists(DB_FILE):
        print("Database not found!")
        return

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    print("\n" + "="*50)
    print("DATABASE INSPECTION REPORT")
    print("="*50)

    # 1. Files
    files = cursor.execute("SELECT path, bucket_folder FROM files").fetchall()
    print(f"\nINDEXED FILES ({len(files)} total):")
    for f in files:
        print(f"  - [{f['bucket_folder']}] {os.path.basename(f['path'])}")

    # 2. Chunks
    chunks = cursor.execute("SELECT chunk_type, count(*) as count FROM chunks GROUP BY chunk_type").fetchall()
    total_chunks = sum(c['count'] for c in chunks)
    print(f"\nCHUNKS ({total_chunks} total):")
    for c in chunks:
        print(f"  - {c['chunk_type'].upper()}: {c['count']} chunks")

    # 3. Relationships
    rels = cursor.execute("SELECT source_path, target_path, rel_type, metadata FROM relationships").fetchall()
    print(f"\nRELATIONSHIP GRAPH ({len(rels)} links found):")
    if not rels:
        print("  No relationships mapped yet.")
    for r in rels:
        source = os.path.basename(r['source_path'])
        target = os.path.basename(r['target_path'])
        meta = json.loads(r['metadata']) if r['metadata'] else {}
        reason = meta.get('reason', 'N/A')
        
        print(f"  [{r['rel_type'].upper()}]")
        print(f"  {source} -> {target}")
        print(f"  Reason: {reason}\n")

    print("="*50 + "\n")
    conn.close()

if __name__ == "__main__":
    inspect_db()
