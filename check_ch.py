import clickhouse_connect
import time

print("Waiting for ClickHouse server to initialize...")
for _ in range(30):
    try:
        client = clickhouse_connect.get_client(host='localhost', port=8123, username='default', password='')
        print("CLICKHOUSE IS UP!")
        exit(0)
    except Exception as e:
        time.sleep(1)

print("Failed to connect to ClickHouse within 30 seconds.")
exit(1)
