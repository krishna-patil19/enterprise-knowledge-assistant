# ClickHouse Ingestion and Aggregation Guide
## Overview
This document guides developers on optimizing time-series transactional aggregation within the `analytics` service using ClickHouse engines. Specifically, it covers the `AggregatingMergeTree` and `ReplacingMergeTree` architectures.

## Ingestion Architecture
Data flows from our S3 landing bucket (`s3://company-knowledge/`) into raw tables, which are then incrementally aggregated into state tables.

### Raw Tables
The entry table for all sales data is `analytics.raw_sales`. It holds atomic, non-aggregated event streams.
* The script [etl_pipeline.py](file:///s3_bucket/python/etl_pipeline.py) handles batch uploads to this table.

### AggregatingMergeTree
To perform low-latency query aggregations over millions of rows, we utilize ClickHouse's `AggregatingMergeTree` table engine.
We define `analytics.sales_aggregates_local` to store pre-aggregated database states.

* **Important Functions:**
  * `sumState(column)`: Saves the intermediate summation state.
  * `sumMerge(state)`: Merges the intermediate states at query time.
  * `uniqState(column)`: Saves the hyperloglog state for cardinality estimations.
  * `uniqMerge(state)`: Merges the state for unique count reports.

Refer to the database schema defined in [analytics_queries.sql](file:///s3_bucket/sql/analytics_queries.sql) for exact fields and primary keys.

## Optimization Strategies
* **Partitioning:** Partition by month using `toYYYYMM(event_date)` to support rapid drops of historic partitions.
* **Low-Token Queries:** When querying monthly reports, utilize pre-aggregated metrics via `sumMerge` instead of reading the atomic `analytics.raw_sales` to minimize data scan costs.
