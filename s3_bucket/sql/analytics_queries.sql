-- CLICKHOUSE ANALYTICS QUERIES
-- File: s3_bucket/sql/analytics_queries.sql
-- Repo: analytics-service
-- Description: Definitions and queries for processing sales and user activity metrics using AggregatingMergeTree.

-- Query 1: Create local sales aggregate table
CREATE TABLE IF NOT EXISTS analytics.sales_aggregates_local ON CLUSTER analytics_cluster
(
    event_date Date,
    product_id UInt32,
    store_id UInt16,
    revenue_sum AggregateFunction(sum, Float64),
    units_sum AggregateFunction(sum, UInt32),
    customers_uniq AggregateFunction(uniq, UInt64)
)
ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(event_date)
PRIMARY KEY (product_id, store_id)
ORDER BY (product_id, store_id, event_date);

-- Query 2: Create MV to populate aggregates
CREATE MATERIALIZED VIEW IF NOT EXISTS analytics.mv_sales_aggregates TO analytics.sales_aggregates_local
AS SELECT
    event_date,
    product_id,
    store_id,
    sumState(revenue) AS revenue_sum,
    sumState(units) AS units_sum,
    uniqState(customer_id) AS customers_uniq
FROM analytics.raw_sales
GROUP BY event_date, product_id, store_id;

-- Query 3: Query monthly revenue and units
SELECT
    product_id,
    sumMerge(revenue_sum) AS total_revenue,
    sumMerge(units_sum) AS total_units,
    uniqMerge(customers_uniq) AS unique_customers
FROM analytics.sales_aggregates_local
WHERE event_date >= '2026-01-01' AND event_date <= '2026-05-31'
GROUP BY product_id
ORDER BY total_revenue DESC;
