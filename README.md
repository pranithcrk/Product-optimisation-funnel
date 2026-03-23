# E-Commerce Conversion Funnel Intelligence Platform

An end-to-end product analytics platform that analyzes user journeys through a conversion funnel, identifies drop-off patterns, predicts which users will convert, and surfaces actionable recommendations — all wrapped in an interactive Streamlit dashboard.

## Architecture

```
Raw Event Data (CSV)
        │
        ▼
┌─────────────────┐
│    SQL Layer     │  ← Sessionization, funnel staging, cohort tables
│    (DuckDB)     │     Window functions, CTEs
└────────┬────────┘
         ▼
┌─────────────────┐
│   Python ETL    │  ← Feature engineering (23 session/user features)
│   (Pandas)      │     Modular pipeline steps
└────────┬────────┘
         ▼
┌─────────────────┐
│   ML Pipeline   │  ← Gradient Boosting conversion model
│  (Scikit-learn) │     SHAP explainability
└────────┬────────┘
         ▼
┌─────────────────┐
│   Streamlit     │  ← 5-page interactive dashboard
│   Dashboard     │     Plotly visualizations
└─────────────────┘
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Funnel Overview** | Interactive funnel chart with conversion rates, date/brand filters, daily trend |
| **Drop-off Analysis** | Heatmaps by hour/day-of-week, price segment breakdowns, conversion patterns |
| **Conversion Prediction** | Predict conversion probability for any session profile with SHAP explanations |
| **A/B Test Calculator** | Sample size calculator with power curves and sensitivity tables |
| **Cohort & Retention** | Weekly retention heatmap, retention curves, revenue-per-user by cohort |

## Key Technical Highlights

- **SQL Sessionization**: Reconstructs user sessions from raw events using `LAG` + cumulative `SUM` window functions with 30-minute inactivity gap detection
- **9 DuckDB Queries**: Funnel staging, stage-to-stage drop-off, time patterns, cohort retention, user feature aggregation
- **23 ML Features**: Session duration, product diversity ratio, cart-to-view ratio, prior purchase history, temporal features
- **SHAP Explainability**: Feature importance ranking and per-prediction driver explanations
- **A/B Test Rigor**: Two-proportion z-test sample size calculation with power curves

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data (or download from Kaggle)
python -m src.data.generate_synthetic

# Run the data pipeline (sessionization + funnel staging)
python -m src.data.pipeline

# Build ML features
python -m src.features.engineering

# Train the conversion model
python -m src.models.conversion_model

# Launch the dashboard
streamlit run src/dashboard/app.py
```

### Using Real Data (Kaggle)

To use the [eCommerce Events dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop) instead of synthetic data:

1. Set up Kaggle API credentials (`~/.kaggle/kaggle.json`)
2. Run `python -m src.data.download`
3. Continue with the pipeline steps above

## Project Structure

```
├── src/
│   ├── data/
│   │   ├── download.py            # Kaggle dataset downloader
│   │   ├── generate_synthetic.py  # Synthetic data generator
│   │   └── pipeline.py            # DuckDB pipeline runner
│   ├── sql/
│   │   └── funnel_queries.py      # 9 SQL queries (sessionization, funnel, cohorts)
│   ├── features/
│   │   └── engineering.py         # Session & user-level feature builder
│   ├── models/
│   │   └── conversion_model.py    # GradientBoosting + SHAP training
│   └── dashboard/
│       ├── app.py                 # Streamlit app entry point
│       ├── data_loader.py         # Cached data loading utilities
│       └── pages/                 # 5 dashboard pages
├── models/                        # Saved model artifacts & metrics
├── data/                          # Raw & processed data (gitignored)
└── requirements.txt
```

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC (CV) | 0.9996 ± 0.0001 |
| ROC-AUC (Test) | 0.9999 |
| PR-AUC (Test) | 0.9997 |

*Scores reflect synthetic data with clean patterns. Expect 0.85–0.92 AUC with real-world Kaggle data.*

## Top Conversion Drivers (SHAP)

1. **Product diversity ratio** — browsing breadth vs depth
2. **Total events** — session engagement level
3. **View count** — number of product views
4. **Cart-to-view ratio** — intent signal strength
5. **Unique products** — exploration breadth

## Tech Stack

- **Data**: DuckDB, Pandas, PyArrow
- **ML**: Scikit-learn (GradientBoosting), SHAP
- **Visualization**: Streamlit, Plotly
- **Stats**: SciPy (A/B test calculations)
