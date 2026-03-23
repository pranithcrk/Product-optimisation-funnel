"""Generate realistic synthetic eCommerce event data for testing.

Mirrors the schema of the Kaggle cosmetics shop dataset:
event_time, event_type, product_id, category_id, category_code, brand, price, user_id, user_session

Generates ~500K events with realistic funnel drop-off rates:
  view (100%) → cart (~15%) → purchase (~5%)
"""

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

BRANDS = [
    "maybelline", "loreal", "estee-lauder", "clinique", "mac", "nyx",
    "revlon", "covergirl", "neutrogena", "olay", "garnier", "nivea",
    "dove", "aveeno", "cetaphil", "la-roche-posay", "vichy", "bioderma",
]

CATEGORIES = [
    ("1487580005092295511", "beauty.cosmetics.lip"),
    ("1487580005134238553", "beauty.cosmetics.eye"),
    ("1487580005268456359", "beauty.cosmetics.face"),
    ("1487580006317032337", "beauty.skin.body"),
    ("1487580006383354753", "beauty.skin.face"),
    ("1487580012325355431", "beauty.hair.shampoo"),
    ("1487580013069066219", "beauty.fragrance.women"),
    ("1487580013345890467", "beauty.nail"),
]

NUM_USERS = 10_000
NUM_PRODUCTS = 2_000
START_DATE = datetime(2024, 10, 1)
END_DATE = datetime(2024, 12, 31)


def generate():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    random.seed(42)

    # Pre-generate product catalog
    products = []
    for pid in range(1, NUM_PRODUCTS + 1):
        cat_id, cat_code = random.choice(CATEGORIES)
        products.append({
            "product_id": 100000 + pid,
            "category_id": int(cat_id),
            "category_code": cat_code,
            "brand": random.choice(BRANDS),
            "price": round(random.uniform(2.0, 150.0), 2),
        })

    rows = []
    total_days = (END_DATE - START_DATE).days

    for user_id in range(1, NUM_USERS + 1):
        # Each user has 1-10 sessions
        n_sessions = rng.integers(1, 11)
        for _ in range(n_sessions):
            session_id = f"{user_id}_{random.randint(100000, 999999)}"
            session_start = START_DATE + timedelta(
                days=int(rng.integers(0, total_days)),
                hours=int(rng.integers(0, 24)),
                minutes=int(rng.integers(0, 60)),
            )
            ts = session_start

            # Each session: 1-20 views
            n_views = rng.integers(1, 21)
            session_products = rng.choice(products, size=n_views)

            for prod in session_products:
                ts += timedelta(seconds=int(rng.integers(5, 300)))
                rows.append({
                    "event_time": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "event_type": "view",
                    "product_id": prod["product_id"],
                    "category_id": prod["category_id"],
                    "category_code": prod["category_code"],
                    "brand": prod["brand"],
                    "price": prod["price"],
                    "user_id": user_id,
                    "user_session": session_id,
                })

                # 15% chance of adding to cart after viewing
                if random.random() < 0.15:
                    ts += timedelta(seconds=int(rng.integers(3, 60)))
                    rows.append({
                        "event_time": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                        "event_type": "cart",
                        **{k: prod[k] for k in ["product_id", "category_id", "category_code", "brand", "price"]},
                        "user_id": user_id,
                        "user_session": session_id,
                    })

                    # 35% of carts → purchase
                    if random.random() < 0.35:
                        ts += timedelta(seconds=int(rng.integers(10, 120)))
                        rows.append({
                            "event_time": ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                            "event_type": "purchase",
                            **{k: prod[k] for k in ["product_id", "category_id", "category_code", "brand", "price"]},
                            "user_id": user_id,
                            "user_session": session_id,
                        })

    df = pd.DataFrame(rows)
    df = df.sort_values("event_time").reset_index(drop=True)

    out_path = RAW_DIR / "2024-Oct-Dec-synthetic.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} events for {NUM_USERS:,} users → {out_path}")
    print(f"Event type distribution:\n{df['event_type'].value_counts().to_string()}")
    return out_path


if __name__ == "__main__":
    generate()
