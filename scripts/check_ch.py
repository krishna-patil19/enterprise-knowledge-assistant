import os
import sys
from dotenv import load_dotenv
import clickhouse_connect

load_dotenv()

host = os.getenv('CLICKHOUSE_HOST', 'localhost')
port = int(os.getenv('CLICKHOUSE_PORT', 8443))
user = os.getenv('CLICKHOUSE_USER', 'default')
password = os.getenv('CLICKHOUSE_PASSWORD', '')

print(f"Connecting to: {host}:{port} as {user}")

try:
    client = clickhouse_connect.get_client(
        host=host,
        port=port,
        username=user,
        password=password,
        secure=True if 'cloud' in host else False
    )
    version = client.command("SELECT version()")
    print(f"SUCCESS! Connected to ClickHouse version: {version}")
except Exception as e:
    print(f"FAILED to connect: {str(e)}")
    sys.exit(1)
