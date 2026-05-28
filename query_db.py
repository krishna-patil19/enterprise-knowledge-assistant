import os
from dotenv import load_dotenv
import clickhouse_connect

load_dotenv()

host = os.getenv('CLICKHOUSE_HOST')
port = int(os.getenv('CLICKHOUSE_PORT', 8443))
user = os.getenv('CLICKHOUSE_USER', 'default')
password = os.getenv('CLICKHOUSE_PASSWORD', '')

client = clickhouse_connect.get_client(
    host=host, port=port, username=user, password=password, secure=True
)

print("\n--- FILES IN CLICKHOUSE ---")
res = client.query("SELECT path, bucket_folder FROM files LIMIT 5")
for row in res.result_rows:
    print(f"[{row[1]}] {os.path.basename(row[0])}")

print("\n--- CHUNKS STORED ---")
res = client.query("SELECT id, chunk_type, substring(content, 1, 80) FROM chunks LIMIT 5")
for row in res.result_rows:
    print(f"{row[0]:45s} | {row[1]:10s} | {row[2]}...")

print("\n--- RELATIONSHIPS GRAPH ---")
res = client.query("SELECT source_path, target_path, rel_type FROM relationships LIMIT 10")
for row in res.result_rows:
    src = os.path.basename(row[0])
    tgt = os.path.basename(row[1])
    print(f"{src:25s} --[{row[2]:18s}]--> {tgt}")

print("\n--- TOTAL STATS ---")
f = client.command("SELECT count() FROM files")
c = client.command("SELECT count() FROM chunks")
e = client.command("SELECT count() FROM embeddings")
r = client.command("SELECT count() FROM relationships")
print(f"Files: {f} | Chunks: {c} | Vectors: {e} | Links: {r}")
