CREATE TABLE mock_analytics_table (
    id UUID,
    event_name String,
    revenue Float32
) ENGINE = MergeTree()
ORDER BY id;
