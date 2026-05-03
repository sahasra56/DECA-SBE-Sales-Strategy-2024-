# The-Den---SBE-Sales-Strategy
In 2023–2024, I conducted a sales strategy project for my high school's DECA School-Based Enterprise student store. This repo documents the analysis, forecasting models, and Plotly dashboards I built — reconstructed from the original work with a synthetic dataset that mirrors the real POS data. I analyzed POS transaction data in Python and SQL to identify underperforming SKUs, driving targeted restocking decisions that produced a **200% increase in inventory turnover**. Built scikit-learn demand forecasting models and Plotly dashboards to automate weekly sales reporting and replace manual tracking pipelines.
 
---
 
## Project Timeline
 
| Phase | Dates | Description |
|---|---|---|
| Baseline | Aug – Oct 2023 | Store running normally. Italian sodas & custom drinks visibly underselling — stockouts happening mid-day, suppressing recorded sales |
| Analysis | Oct 2023 – Jan 2024 | DECA project kicks off. POS data pulled via SQL, SKU turnover analysis run, stockout patterns identified, demand forecasting models built |
| Strategy #1 | Jan 22, 2024 | Raised par levels on Italian sodas and custom drinks, fixed reorder points → immediate revenue lift |
| Strategy #2 | Mar 17, 2024 | Automated reorder pipeline deployed. Slow SKUs cut (Pretzels, Altoids, Rice Krispies Treats). High-demand items expanded. Second lift |
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Avg daily revenue — Baseline | $174 |
| Avg daily revenue — Strategy #1 | $325 (+87%) |
| Avg daily revenue — Strategy #2 | $374 (+115% from baseline) |
| Inventory turnover increase | ~200% |
| Demand forecast MAE | 4.51 ± 0.65 units/week |
 
---
 
## Repo Structure
 
```
├── analysis.py          # Main analysis script — run this weekly
├── generate_data.py     # Generates the synthetic POS dataset
├── requirements.txt
└── data/
    ├── pos_transactions.csv   # One row per SKU per day
    ├── weekly_sales.csv       # Aggregated by week + SKU
    ├── daily_summary.csv      # Total revenue + units per day
    └── sku_catalog.csv        # Product reference table (35 SKUs)
```
 
---
 
## What `analysis.py` Does
 
### 1. SQL Data Layer
All data access runs through `sqlite3` SQL queries — CTEs, window functions, multi-table joins. Loads the four CSVs into a local `store.db` on first run.
 
### 2. SKU Turnover Analysis
- Computes inventory turnover ratio per SKU per phase via SQL
- Flags bottom 25% within each category as underperforming (`PERCENT_RANK`)
- Generates ranked restocking recommendations by priority score (revenue / turnover)
- Quantifies turnover improvement across all four phases
### 3. Demand Forecasting (scikit-learn)
- Engineers lag and rolling features from weekly sales time series
- Trains a `GradientBoostingRegressor` with `TimeSeriesSplit` cross-validation
- Generates next-week unit demand forecasts for every active SKU
### 4. Automated Weekly Plotly Dashboard
Outputs `outputs/weekly_dashboard.html` — a self-contained 6-panel dashboard:
- Daily revenue across the full year with strategy phase markers
- Inventory turnover before/after each strategy
- Top SKUs by revenue (current week)
- Underperforming SKUs flagged for restocking
- Next-week demand forecasts
- Weekly revenue by category (stacked area)
### 5. Weekly Text Report
Printed to stdout on every run — replaces the manual weekly tracking spreadsheet.
 
---
 
## Usage
 
```bash
pip install -r requirements.txt
 
# Generate the dataset (first time only)
python generate_data.py
 
# Run the full analysis + dashboard
python analysis.py
```
 
Open `outputs/weekly_dashboard.html` in any browser to view the dashboard.
 
---
 
## Tech Stack
 
- **Python**, **SQL** (sqlite3) — data analysis and querying
- **pandas**, **NumPy** — data manipulation
- **scikit-learn** — demand forecasting model
- **Plotly** — interactive dashboard
 
