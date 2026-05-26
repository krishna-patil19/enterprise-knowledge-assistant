# ETL pipeline for clickhouse analytics data ingestion
# File: s3_bucket/python/etl_pipeline.py
# Repo: analytics-service
# Depends: s3_bucket/sql/analytics_queries.sql

import os
import json
import logging
from datetime import datetime
from clickhouse_driver import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalesDataPipeline:
    """
    SalesDataPipeline manages the ETL process from S3 landing zone to ClickHouse raw_sales.
    """
    def __init__(self, host='localhost', port=9000, database='analytics'):
        self.client = Client(host=host, port=port, database=database)
        logger.info(f"Initialized ClickHouse client for host {host}")

    def load_raw_sales(self, file_path: str):
        """
        Loads raw transactional sales data from local parquet/csv file into ClickHouse raw_sales table.
        Refers to analytics.raw_sales.
        """
        logger.info(f"Starting ingestion of raw sales from {file_path}")
        # Simulated ingestion SQL
        ingestion_sql = """
            INSERT INTO analytics.raw_sales (event_date, product_id, store_id, revenue, units, customer_id)
            VALUES
        """
        try:
            # Simulated parsing and batch insert
            records = [{"event_date": "2026-05-25", "product_id": 101, "store_id": 1, "revenue": 150.50, "units": 2, "customer_id": 8802}]
            logger.info(f"Inserting {len(records)} records into raw_sales")
            # In a real environment: self.client.execute(ingestion_sql, records)
            return True
        except Exception as e:
            logger.error(f"Failed to ingest sales: {str(e)}")
            raise

    def fetch_monthly_report(self, month: str):
        """
        Fetches the aggregated monthly sales report by executing a query over analytics.sales_aggregates_local.
        Uses sumMerge and uniqMerge functions.
        """
        logger.info(f"Generating monthly report for {month}")
        query = """
            SELECT
                product_id,
                sumMerge(revenue_sum) AS total_revenue,
                sumMerge(units_sum) AS total_units,
                uniqMerge(customers_uniq) AS unique_customers
            FROM analytics.sales_aggregates_local
            WHERE toYYYYMM(event_date) = %(month)s
            GROUP BY product_id
            ORDER BY total_revenue DESC
        """
        try:
            results = self.client.execute(query, {'month': month})
            return results
        except Exception as e:
            logger.error(f"Error fetching report: {str(e)}")
            return []

def main():
    pipeline = SalesDataPipeline()
    pipeline.load_raw_sales("s3://company-knowledge/sales_data_2026_05.csv")
    report = pipeline.fetch_monthly_report("202605")
    print(f"Loaded successfully. Reports generated: {len(report)}")

if __name__ == "__main__":
    main()
