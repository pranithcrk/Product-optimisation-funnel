"""Phase 2: Feature engineering for conversion prediction.

Builds session-level and user-level features from the funnel data,
creating an ML-ready dataset for the XGBoost conversion model.

Features are designed to answer: "Given what we know about this session
so far, will the user convert (purchase)?"
"""

import duckdb
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "funnel.duckdb"


# ---------------------------------------------------------------------------
# Session-level feature queries (run in DuckDB for speed)
# ---------------------------------------------------------------------------

SESSION_FEATURES_QUERY = """
WITH session_time_features AS (
    SELECT
        computed_session_id,
        user_id,
        MIN(event_time) AS session_start,
        MAX(event_time) AS session_end,
        EXTRACT(HOUR FROM MIN(event_time)) AS start_hour,
        EXTRACT(DOW FROM MIN(event_time)) AS day_of_week,
        CASE WHEN EXTRACT(DOW FROM MIN(event_time)) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend,
        DATEDIFF('second', MIN(event_time), MAX(event_time)) AS session_duration_sec,
        COUNT(*) AS total_events,
        COUNT(DISTINCT product_id) AS unique_products,
        COUNT(DISTINCT brand) AS unique_brands,
        COUNT(DISTINCT category_code) AS unique_categories,

        -- Funnel stage counts
        SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count,
        SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS cart_count,
        SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,

        -- Price features
        AVG(price) AS avg_price,
        MAX(price) AS max_price,
        MIN(price) AS min_price,
        MAX(price) - MIN(price) AS price_range,
        SUM(CASE WHEN event_type = 'cart' THEN price ELSE 0 END) AS cart_value,

        -- Target: did this session result in a purchase?
        CASE WHEN SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) > 0
             THEN 1 ELSE 0 END AS converted

    FROM funnel_events
    GROUP BY computed_session_id, user_id
),
user_history AS (
    -- User-level historical features (what we know about the user before this session)
    SELECT
        s.computed_session_id,
        s.user_id,
        s.session_start,

        -- Count of prior sessions
        (SELECT COUNT(DISTINCT s2.computed_session_id)
         FROM session_funnel_summary s2
         WHERE s2.user_id = s.user_id AND s2.session_start < s.session_start
        ) AS prior_sessions,

        -- Prior purchase count
        (SELECT COALESCE(SUM(s2.purchase_count), 0)
         FROM session_funnel_summary s2
         WHERE s2.user_id = s.user_id AND s2.session_start < s.session_start
        ) AS prior_purchases,

        -- Days since first seen
        (SELECT DATEDIFF('day', MIN(s2.session_start), s.session_start)
         FROM session_funnel_summary s2
         WHERE s2.user_id = s.user_id
        ) AS days_since_first_seen,

        -- Days since last session
        (SELECT DATEDIFF('day', MAX(s2.session_start), s.session_start)
         FROM session_funnel_summary s2
         WHERE s2.user_id = s.user_id AND s2.session_start < s.session_start
        ) AS days_since_last_session

    FROM session_funnel_summary s
)
SELECT
    sf.*,
    COALESCE(uh.prior_sessions, 0) AS prior_sessions,
    COALESCE(uh.prior_purchases, 0) AS prior_purchases,
    COALESCE(uh.days_since_first_seen, 0) AS days_since_first_seen,
    COALESCE(uh.days_since_last_session, 0) AS days_since_last_session,

    -- Derived ratios
    CASE WHEN sf.view_count > 0
         THEN sf.cart_count * 1.0 / sf.view_count ELSE 0 END AS cart_to_view_ratio,
    CASE WHEN sf.total_events > 0
         THEN sf.unique_products * 1.0 / sf.total_events ELSE 0 END AS product_diversity_ratio,
    CASE WHEN COALESCE(uh.prior_sessions, 0) > 0 THEN 1 ELSE 0 END AS is_returning_user

FROM session_time_features sf
LEFT JOIN user_history uh
    ON sf.computed_session_id = uh.computed_session_id;
"""


def build_features():
    """Build ML-ready feature dataset from DuckDB tables."""
    print("Connecting to DuckDB...")
    con = duckdb.connect(str(DB_PATH), read_only=True)

    print("Building session-level features (this may take a moment)...")
    df = con.execute(SESSION_FEATURES_QUERY).fetchdf()
    con.close()

    # Drop ID/timestamp columns not used for modeling
    id_cols = ["computed_session_id", "user_id", "session_start", "session_end"]
    feature_cols = [c for c in df.columns if c not in id_cols and c != "converted"]

    # Fill NaN
    df[feature_cols] = df[feature_cols].fillna(0)

    # Save
    out_path = PROJECT_ROOT / "data" / "processed" / "ml_features.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nFeature dataset: {len(df):,} sessions, {len(feature_cols)} features")
    print(f"Target distribution:\n{df['converted'].value_counts().to_string()}")
    print(f"Conversion rate: {df['converted'].mean():.1%}")
    print(f"Saved → {out_path}")

    return df, feature_cols, id_cols


if __name__ == "__main__":
    build_features()
