"""SQL queries for funnel analysis using DuckDB.

These demonstrate advanced SQL skills: window functions, CTEs, sessionization,
event sequencing, and cohort analysis — the kind of SQL expected in Analytics
Engineer roles at companies like SeatGeek.
"""

# ---------------------------------------------------------------------------
# 1. Load raw CSV data into DuckDB
# ---------------------------------------------------------------------------
LOAD_RAW_EVENTS = """
CREATE OR REPLACE TABLE raw_events AS
SELECT
    event_time::TIMESTAMP AS event_time,
    event_type,
    product_id::BIGINT AS product_id,
    category_id::BIGINT AS category_id,
    category_code,
    brand,
    price::DOUBLE AS price,
    user_id::BIGINT AS user_id,
    user_session AS session_id
FROM read_csv_auto('{csv_pattern}', header=true, ignore_errors=true);
"""

# ---------------------------------------------------------------------------
# 2. Sessionize events with 30-min inactivity gap
#    Rebuilds sessions from scratch using window functions, which is more
#    robust than trusting the raw session_id field.
# ---------------------------------------------------------------------------
SESSIONIZE_EVENTS = """
CREATE OR REPLACE TABLE sessionized_events AS
WITH ordered_events AS (
    SELECT
        *,
        LAG(event_time) OVER (PARTITION BY user_id ORDER BY event_time) AS prev_event_time
    FROM raw_events
),
session_boundaries AS (
    SELECT
        *,
        CASE
            WHEN prev_event_time IS NULL
                 OR DATEDIFF('minute', prev_event_time, event_time) > 30
            THEN 1
            ELSE 0
        END AS is_new_session
    FROM ordered_events
),
session_ids AS (
    SELECT
        *,
        SUM(is_new_session) OVER (
            PARTITION BY user_id ORDER BY event_time
            ROWS UNBOUNDED PRECEDING
        ) AS computed_session_num
    FROM session_boundaries
)
SELECT
    user_id,
    user_id::VARCHAR || '_' || computed_session_num::VARCHAR AS computed_session_id,
    session_id AS original_session_id,
    event_time,
    event_type,
    product_id,
    category_id,
    category_code,
    brand,
    price
FROM session_ids;
"""

# ---------------------------------------------------------------------------
# 3. Map events to funnel stages and assign ordinal positions
#    Funnel: view(1) → cart(2) → purchase(3)
#    (remove_from_cart is tracked as a negative signal but not a funnel stage)
# ---------------------------------------------------------------------------
FUNNEL_STAGING = """
CREATE OR REPLACE TABLE funnel_events AS
SELECT
    *,
    CASE event_type
        WHEN 'view'    THEN 1
        WHEN 'cart'    THEN 2
        WHEN 'purchase' THEN 3
        ELSE NULL
    END AS funnel_stage,
    CASE event_type
        WHEN 'view'    THEN 'Browse / View'
        WHEN 'cart'    THEN 'Add to Cart'
        WHEN 'purchase' THEN 'Purchase'
        ELSE 'Other'
    END AS funnel_stage_name
FROM sessionized_events
WHERE event_type IN ('view', 'cart', 'purchase');
"""

# ---------------------------------------------------------------------------
# 4. Per-session funnel progression — max stage reached per session
# ---------------------------------------------------------------------------
SESSION_FUNNEL_SUMMARY = """
CREATE OR REPLACE TABLE session_funnel_summary AS
SELECT
    computed_session_id,
    user_id,
    MIN(event_time) AS session_start,
    MAX(event_time) AS session_end,
    DATEDIFF('minute', MIN(event_time), MAX(event_time)) AS session_duration_min,
    COUNT(*) AS total_events,
    COUNT(DISTINCT product_id) AS products_interacted,
    MAX(funnel_stage) AS max_funnel_stage,
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
    SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
    MAX(price) AS max_price,
    AVG(price) AS avg_price,
    COUNT(DISTINCT brand) AS brands_viewed,
    COUNT(DISTINCT category_code) AS categories_viewed
FROM funnel_events
GROUP BY computed_session_id, user_id;
"""

# ---------------------------------------------------------------------------
# 5. Overall funnel conversion rates
# ---------------------------------------------------------------------------
FUNNEL_CONVERSION_RATES = """
SELECT
    funnel_stage_name,
    funnel_stage,
    COUNT(DISTINCT computed_session_id) AS sessions,
    COUNT(DISTINCT user_id) AS users,
    ROUND(
        COUNT(DISTINCT computed_session_id) * 100.0
        / FIRST(total_sessions.cnt), 2
    ) AS pct_of_total_sessions
FROM funnel_events
CROSS JOIN (
    SELECT COUNT(DISTINCT computed_session_id) AS cnt FROM funnel_events
) AS total_sessions
GROUP BY funnel_stage_name, funnel_stage
ORDER BY funnel_stage;
"""

# ---------------------------------------------------------------------------
# 6. Stage-to-stage drop-off rates
# ---------------------------------------------------------------------------
STAGE_DROPOFF = """
WITH stage_counts AS (
    SELECT
        funnel_stage,
        funnel_stage_name,
        COUNT(DISTINCT computed_session_id) AS sessions
    FROM funnel_events
    GROUP BY funnel_stage, funnel_stage_name
)
SELECT
    s.funnel_stage_name AS from_stage,
    LEAD(s.funnel_stage_name) OVER (ORDER BY s.funnel_stage) AS to_stage,
    s.sessions AS from_sessions,
    LEAD(s.sessions) OVER (ORDER BY s.funnel_stage) AS to_sessions,
    ROUND(
        LEAD(s.sessions) OVER (ORDER BY s.funnel_stage) * 100.0 / s.sessions, 2
    ) AS conversion_rate_pct,
    ROUND(
        100.0 - LEAD(s.sessions) OVER (ORDER BY s.funnel_stage) * 100.0 / s.sessions, 2
    ) AS dropoff_rate_pct
FROM stage_counts s
ORDER BY s.funnel_stage;
"""

# ---------------------------------------------------------------------------
# 7. Time-based drop-off patterns (hour of day, day of week)
# ---------------------------------------------------------------------------
TIME_PATTERNS = """
SELECT
    EXTRACT(DOW FROM event_time) AS day_of_week,
    EXTRACT(HOUR FROM event_time) AS hour_of_day,
    event_type,
    COUNT(*) AS event_count,
    COUNT(DISTINCT user_id) AS unique_users
FROM funnel_events
GROUP BY day_of_week, hour_of_day, event_type
ORDER BY day_of_week, hour_of_day, event_type;
"""

# ---------------------------------------------------------------------------
# 8. Weekly cohort retention
# ---------------------------------------------------------------------------
COHORT_RETENTION = """
WITH user_first_week AS (
    SELECT
        user_id,
        DATE_TRUNC('week', MIN(event_time)) AS cohort_week
    FROM funnel_events
    GROUP BY user_id
),
user_activity AS (
    SELECT
        f.user_id,
        u.cohort_week,
        DATE_TRUNC('week', f.event_time) AS activity_week,
        DATEDIFF('week', u.cohort_week, DATE_TRUNC('week', f.event_time)) AS weeks_since_first
    FROM funnel_events f
    JOIN user_first_week u ON f.user_id = u.user_id
)
SELECT
    cohort_week,
    weeks_since_first,
    COUNT(DISTINCT user_id) AS active_users,
    FIRST(cohort_size.cnt) AS cohort_size,
    ROUND(COUNT(DISTINCT user_id) * 100.0 / FIRST(cohort_size.cnt), 2) AS retention_pct
FROM user_activity a
CROSS JOIN (
    SELECT cohort_week, COUNT(DISTINCT user_id) AS cnt
    FROM user_activity
    WHERE weeks_since_first = 0
    GROUP BY cohort_week
) cohort_size
WHERE a.cohort_week = cohort_size.cohort_week
GROUP BY a.cohort_week, weeks_since_first
ORDER BY cohort_week, weeks_since_first;
"""

# ---------------------------------------------------------------------------
# 9. User-level aggregates for ML feature engineering
# ---------------------------------------------------------------------------
USER_FEATURES = """
CREATE OR REPLACE TABLE user_features AS
SELECT
    user_id,
    COUNT(DISTINCT computed_session_id) AS total_sessions,
    COUNT(*) AS total_events,
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS total_views,
    SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS total_carts,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS total_purchases,
    COUNT(DISTINCT product_id) AS unique_products,
    COUNT(DISTINCT brand) AS unique_brands,
    COUNT(DISTINCT category_code) AS unique_categories,
    AVG(price) AS avg_price,
    MAX(price) AS max_price,
    MIN(event_time) AS first_seen,
    MAX(event_time) AS last_seen,
    DATEDIFF('day', MIN(event_time), MAX(event_time)) AS active_days_span,
    CASE WHEN SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) > 0
         THEN 1 ELSE 0
    END AS has_purchased
FROM funnel_events
GROUP BY user_id;
"""
