"""
analysis.py
-----------
DECA School-Based Enterprise — Sales Strategy & Demand Forecasting

Matches resume bullet points exactly:
  1. "Analyzed POS transaction data in Python and SQL to identify
     underperforming SKUs, driving targeted restocking decisions that
     produced a 200% increase in inventory turnover"
       → SQL queries via sqlite3 for all data pulls
       → SKU-level inventory turnover analysis
       → Restocking recommendations tied to Strategy #1 and Strategy #2
       → Turnover comparison across phases to quantify the 200% lift

  2. "Built scikit-learn demand forecasting models and Plotly dashboards
     to automate weekly sales reporting and replace manual tracking pipelines"
       → scikit-learn GradientBoostingRegressor with time-series CV
       → Automated weekly Plotly dashboard (run on any new weekly_sales.csv)
       → Replaces manual reporting: one script, full output every week
"""

import sqlite3
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)

# ── Strategy phase boundaries (must match generate_data.py) ────────────────────
PROJECT_START   = "2023-10-23"
STRATEGY_1_DATE = "2024-01-22"
STRATEGY_2_DATE = "2024-03-17"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SQL DATA LAYER
# All data access goes through SQL queries on a local SQLite database.
# This matches: "analyzed POS transaction data in Python and SQL"
# ══════════════════════════════════════════════════════════════════════════════

def build_database(
    pos_path:     str = "data/pos_transactions.csv",
    weekly_path:  str = "data/weekly_sales.csv",
    catalog_path: str = "data/sku_catalog.csv",
    daily_path:   str = "data/daily_summary.csv",
    db_path:      str = "data/store.db",
) -> sqlite3.Connection:
    """
    Load CSVs into a SQLite database and return a connection.
    Having a real DB means all analysis runs through SQL — queryable,
    auditable, and easy to update each week with new exports.
    """
    conn = sqlite3.connect(db_path)

    pd.read_csv(pos_path).to_sql("pos_transactions", conn, if_exists="replace", index=False)
    pd.read_csv(weekly_path).to_sql("weekly_sales",   conn, if_exists="replace", index=False)
    pd.read_csv(catalog_path).to_sql("sku_catalog",   conn, if_exists="replace", index=False)
    pd.read_csv(daily_path).to_sql("daily_summary",   conn, if_exists="replace", index=False)

    print("Database built → data/store.db")
    print(f"  pos_transactions : {pd.read_sql('SELECT COUNT(*) n FROM pos_transactions', conn).iloc[0,0]:,} rows")
    print(f"  weekly_sales     : {pd.read_sql('SELECT COUNT(*) n FROM weekly_sales', conn).iloc[0,0]:,} rows")
    print(f"  sku_catalog      : {pd.read_sql('SELECT COUNT(*) n FROM sku_catalog', conn).iloc[0,0]} SKUs")
    return conn


def query(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    """Execute a SQL query and return a DataFrame."""
    return pd.read_sql(sql, conn)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SKU TURNOVER ANALYSIS
# Identifies underperforming SKUs using inventory turnover ratio,
# then drives restocking decisions for Strategy #1 and Strategy #2.
# Matches: "identify underperforming SKUs, driving targeted restocking
# decisions that produced a 200% increase in inventory turnover"
# ══════════════════════════════════════════════════════════════════════════════

def sku_turnover_by_phase(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    SQL query: compute inventory turnover ratio per SKU per phase.
    Turnover = total units sold / avg daily inventory for that phase.
    Joining pos_transactions with sku_catalog to get inventory levels.
    """
    sql = """
        SELECT
            p.sku,
            p.category,
            p.phase,
            SUM(p.units_sold)                          AS total_units_sold,
            SUM(p.revenue)                             AS total_revenue,
            COUNT(DISTINCT p.date)                     AS active_days,
            ROUND(SUM(p.units_sold) * 1.0 /
                NULLIF(
                    CASE p.phase
                        WHEN 'baseline'        THEN c.inv_pre
                        WHEN 'analysis'        THEN c.inv_pre
                        WHEN 'post_strategy_1' THEN c.inv_s1
                        WHEN 'post_strategy_2' THEN c.inv_s2
                    END, 0
                ), 2
            )                                          AS turnover_ratio
        FROM pos_transactions p
        JOIN sku_catalog c ON p.sku = c.sku
        GROUP BY p.sku, p.category, p.phase
        ORDER BY p.category, p.sku, p.phase
    """
    return query(conn, sql)


def baseline_underperformers(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    SQL query: flag SKUs in the bottom 25% of turnover during baseline.
    These are the candidates that drove Strategy #1 restocking decisions.
    """
    sql = """
        WITH baseline_turnover AS (
            SELECT
                p.sku,
                p.name,
                p.category,
                SUM(pt.units_sold) * 1.0 / NULLIF(p.inv_pre, 0) AS turnover_ratio,
                SUM(pt.revenue)                                   AS total_revenue
            FROM pos_transactions pt
            JOIN sku_catalog p ON pt.sku = p.sku
            WHERE pt.phase IN ('baseline', 'analysis')
            GROUP BY pt.sku
        ),
        ranked AS (
            SELECT *,
                PERCENT_RANK() OVER (
                    PARTITION BY category ORDER BY turnover_ratio
                ) AS turnover_percentile
            FROM baseline_turnover
        )
        SELECT *,
            CASE WHEN turnover_percentile <= 0.25 THEN 1 ELSE 0 END AS underperforming
        FROM ranked
        ORDER BY turnover_percentile ASC
    """
    return query(conn, sql)


def turnover_improvement(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    SQL query: compare turnover ratio before and after each strategy.
    This is how we quantify the 200% inventory turnover increase.
    """
    sql = """
        WITH phase_turnover AS (
            SELECT
                pt.sku,
                c.name,
                pt.category,
                pt.phase,
                ROUND(
                    SUM(pt.units_sold) * 1.0 /
                    NULLIF(
                        CASE pt.phase
                            WHEN 'baseline'        THEN c.inv_pre
                            WHEN 'analysis'        THEN c.inv_pre
                            WHEN 'post_strategy_1' THEN c.inv_s1
                            WHEN 'post_strategy_2' THEN c.inv_s2
                        END, 0
                    ), 2
                ) AS turnover_ratio
            FROM pos_transactions pt
            JOIN sku_catalog c ON pt.sku = c.sku
            GROUP BY pt.sku, pt.phase
        ),
        pivoted AS (
            SELECT
                sku, name, category,
                MAX(CASE WHEN phase = 'baseline'        THEN turnover_ratio END) AS baseline,
                MAX(CASE WHEN phase = 'post_strategy_1' THEN turnover_ratio END) AS after_s1,
                MAX(CASE WHEN phase = 'post_strategy_2' THEN turnover_ratio END) AS after_s2
            FROM phase_turnover
            GROUP BY sku
        )
        SELECT *,
            ROUND((after_s1 - baseline) / NULLIF(baseline, 0) * 100, 1) AS s1_pct_change,
            ROUND((after_s2 - baseline) / NULLIF(baseline, 0) * 100, 1) AS total_pct_change
        FROM pivoted
        WHERE baseline IS NOT NULL
        ORDER BY total_pct_change DESC
    """
    return query(conn, sql)


def restock_recommendations(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    SQL query: rank restocking priorities for Strategy #1.
    High-revenue SKUs with low turnover = most lost revenue from stockouts.
    Priority score = revenue / turnover (higher = more urgent to restock).
    """
    sql = """
        WITH baseline_stats AS (
            SELECT
                pt.sku,
                c.name,
                pt.category,
                ROUND(SUM(pt.units_sold) * 1.0 / NULLIF(c.inv_pre, 0), 2) AS turnover_ratio,
                ROUND(SUM(pt.revenue), 2)                                   AS total_revenue,
                c.inv_pre,
                c.inv_s1,
                (c.inv_s1 - c.inv_pre)                                      AS reorder_increase
            FROM pos_transactions pt
            JOIN sku_catalog c ON pt.sku = c.sku
            WHERE pt.phase IN ('baseline', 'analysis')
            GROUP BY pt.sku
        ),
        ranked AS (
            SELECT *,
                PERCENT_RANK() OVER (
                    PARTITION BY category ORDER BY turnover_ratio
                ) AS turnover_percentile,
                ROUND(total_revenue / NULLIF(turnover_ratio, 0), 2) AS priority_score
            FROM baseline_stats
        )
        SELECT
            sku, name, category,
            turnover_ratio, total_revenue,
            inv_pre     AS current_par,
            inv_s1      AS recommended_par,
            reorder_increase,
            priority_score
        FROM ranked
        WHERE turnover_percentile <= 0.25
        ORDER BY priority_score DESC
    """
    return query(conn, sql)


def strategy2_cuts(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    SQL query: identify SKUs to cut in Strategy #2.
    Bottom performers by turnover post-Strategy #1 that free up budget
    for higher-demand replacements.
    """
    sql = """
        SELECT
            pt.sku,
            c.name,
            pt.category,
            ROUND(SUM(pt.units_sold) * 1.0 / NULLIF(c.inv_s1, 0), 2) AS turnover_ratio,
            ROUND(SUM(pt.revenue), 2)                                   AS revenue_s1,
            COUNT(DISTINCT pt.date)                                     AS days_active
        FROM pos_transactions pt
        JOIN sku_catalog c ON pt.sku = c.sku
        WHERE pt.phase = 'post_strategy_1'
        GROUP BY pt.sku
        ORDER BY turnover_ratio ASC
        LIMIT 5
    """
    return query(conn, sql)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — DEMAND FORECASTING
# scikit-learn GradientBoostingRegressor with time-series cross-validation.
# Matches: "built scikit-learn demand forecasting models"
# ══════════════════════════════════════════════════════════════════════════════

FORECAST_FEATURES = ["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4", "week_of_year"]


def build_forecast_features(weekly_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer lag and rolling features for each SKU time series.
    Using weekly granularity — enough history per SKU for stable features.
    """
    df = weekly_sales.sort_values(["sku", "week"]).copy()
    df["week"] = pd.to_datetime(df["week"])

    g = df.groupby("sku")["units_sold"]
    df["lag_1"]         = g.shift(1)
    df["lag_2"]         = g.shift(2)
    df["lag_4"]         = g.shift(4)
    df["rolling_mean_4"] = g.shift(1).transform(lambda x: x.rolling(4).mean())
    df["rolling_std_4"]  = g.shift(1).transform(lambda x: x.rolling(4).std())
    df["week_of_year"]  = df["week"].dt.isocalendar().week.astype(int)

    return df.dropna(subset=FORECAST_FEATURES)


def train_demand_model(conn: sqlite3.Connection) -> tuple[GradientBoostingRegressor, dict, pd.DataFrame]:
    """
    Train a GradientBoostingRegressor on all SKUs using time-series CV.
    Returns the fitted model, evaluation metrics, and the feature DataFrame.
    """
    # Pull weekly sales via SQL
    weekly = query(conn, "SELECT week, sku, category, units_sold FROM weekly_sales")
    df = build_forecast_features(weekly)

    X = df[FORECAST_FEATURES]
    y = df["units_sold"]

    model = GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores = -cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
    model.fit(X, y)

    metrics = {
        "mean_mae": round(mae_scores.mean(), 2),
        "std_mae":  round(mae_scores.std(), 2),
    }
    print(f"  Demand model MAE: {metrics['mean_mae']} ± {metrics['std_mae']} units/week")
    return model, metrics, df


def forecast_next_week(model: GradientBoostingRegressor, feature_df: pd.DataFrame) -> pd.DataFrame:
    """Generate next-week unit demand forecasts for every active SKU."""
    latest = feature_df.sort_values("week").groupby("sku").last().reset_index()
    latest["forecast_units"] = model.predict(latest[FORECAST_FEATURES]).clip(min=0).round().astype(int)
    return latest[["sku", "week", "forecast_units"]].rename(columns={"week": "as_of_week"})


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — AUTOMATED WEEKLY PLOTLY DASHBOARD
# Runs on fresh data each week — replaces the manual tracking spreadsheet.
# Matches: "Plotly dashboards to automate weekly sales reporting and
#           replace manual tracking pipelines"
# ══════════════════════════════════════════════════════════════════════════════

def build_weekly_dashboard(
    conn:          sqlite3.Connection,
    underperformers: pd.DataFrame,
    turnover_df:   pd.DataFrame,
    forecasts:     pd.DataFrame,
    output_path:   str = "outputs/weekly_dashboard.html",
) -> None:
    """
    Generate a self-contained HTML dashboard with 5 panels:
      1. Daily revenue over the full year with phase bands
      2. Inventory turnover before/after each strategy (the 200% story)
      3. Top 10 SKUs by revenue this week
      4. Underperforming SKUs flagged for restocking
      5. Next-week demand forecasts
    Saves to outputs/weekly_dashboard.html — open in any browser.
    """

    # ── Pull data via SQL ──────────────────────────────────────────────────────
    daily = query(conn, """
        SELECT date, phase, total_revenue
        FROM daily_summary
        ORDER BY date
    """)
    daily["date"] = pd.to_datetime(daily["date"])

    weekly_by_sku = query(conn, """
        SELECT week, sku, category, SUM(revenue) AS revenue, SUM(units_sold) AS units
        FROM weekly_sales
        GROUP BY week, sku, category
        ORDER BY week DESC
    """)
    latest_week = weekly_by_sku["week"].max()
    top_skus_this_week = (
        weekly_by_sku[weekly_by_sku["week"] == latest_week]
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    category_trend = query(conn, """
        SELECT week, category, SUM(revenue) AS revenue
        FROM weekly_sales
        GROUP BY week, category
        ORDER BY week
    """)

    # ── Color palette ──────────────────────────────────────────────────────────
    PHASE_COLORS = {
        "baseline":        "#94a3b8",
        "analysis":        "#60a5fa",
        "post_strategy_1": "#34d399",
        "post_strategy_2": "#f59e0b",
    }
    PHASE_LABELS = {
        "baseline":        "Baseline",
        "analysis":        "Analysis",
        "post_strategy_1": "Strategy #1",
        "post_strategy_2": "Strategy #2",
    }

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Daily Revenue — Full Year",
            "Inventory Turnover Ratio by Phase",
            "Top SKUs by Revenue (Latest Week)",
            "Underperforming SKUs — Restocking Targets",
            "Next-Week Demand Forecast",
            "Weekly Revenue by Category",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    # ── Panel 1: Daily revenue with phase-colored scatter ─────────────────────
    for phase, grp in daily.groupby("phase"):
        fig.add_trace(go.Scatter(
            x=grp["date"], y=grp["total_revenue"],
            mode="markers+lines",
            name=PHASE_LABELS.get(phase, phase),
            line=dict(color=PHASE_COLORS.get(phase, "#888"), width=1.5),
            marker=dict(size=4),
            legendgroup=phase,
        ), row=1, col=1)

    # Strategy annotation lines on panel 1 (via shapes)
    for date_str, label, color in [
        (STRATEGY_1_DATE, "Strategy #1", "#10b981"),
        (STRATEGY_2_DATE, "Strategy #2", "#f59e0b"),
    ]:
        fig.add_shape(
            type="line", x0=date_str, x1=date_str, y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color=color, dash="dash", width=1.5),
        )
        fig.add_annotation(
            x=date_str, y=1, xref="x", yref="paper",
            text=label, showarrow=False,
            font=dict(color=color, size=10),
            xanchor="left", yanchor="top",
        )

    # ── Panel 2: Turnover improvement bar chart ────────────────────────────────
    top_improved = (
        turnover_df.dropna(subset=["after_s2"])
        .sort_values("total_pct_change", ascending=False)
        .head(12)
    )
    for col_name, label, color in [
        ("baseline", "Baseline",    "#94a3b8"),
        ("after_s1", "Strategy #1", "#34d399"),
        ("after_s2", "Strategy #2", "#f59e0b"),
    ]:
        fig.add_trace(go.Bar(
            x=top_improved["name"],
            y=top_improved[col_name],
            name=label,
            marker_color=color,
            legendgroup=col_name,
        ), row=1, col=2)

    # ── Panel 3: Top SKUs this week ────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=top_skus_this_week["revenue"],
        y=top_skus_this_week["sku"],
        orientation="h",
        marker_color="#6366f1",
        name="Revenue (this week)",
        showlegend=False,
        text=top_skus_this_week["revenue"].apply(lambda v: f"${v:.2f}"),
        textposition="outside",
    ), row=2, col=1)

    # ── Panel 4: Underperformers flagged for restocking ────────────────────────
    under = underperformers[underperformers["underperforming"] == 1].head(10)
    fig.add_trace(go.Bar(
        x=under["turnover_ratio"],
        y=under["name"],
        orientation="h",
        marker_color="#ef4444",
        name="Turnover (underperformers)",
        showlegend=False,
        text=under["turnover_ratio"].apply(lambda v: f"{v:.2f}x"),
        textposition="outside",
    ), row=2, col=2)

    # ── Panel 5: Next-week forecast ────────────────────────────────────────────
    top_forecast = forecasts.sort_values("forecast_units", ascending=False).head(12)
    fig.add_trace(go.Bar(
        x=top_forecast["forecast_units"],
        y=top_forecast["sku"],
        orientation="h",
        marker_color="#0ea5e9",
        name="Forecast units",
        showlegend=False,
        text=top_forecast["forecast_units"],
        textposition="outside",
    ), row=3, col=1)

    # ── Panel 6: Weekly revenue by category (stacked area) ────────────────────
    cat_colors = {
        "Italian Soda":  "#f43f5e",
        "Custom Drink":  "#a855f7",
        "Energy Drink":  "#f59e0b",
        "Soft Drink":    "#0ea5e9",
        "Baked Good":    "#84cc16",
        "Candy & Gum":   "#ec4899",
        "Chips":         "#f97316",
    }
    for cat, grp in category_trend.groupby("category"):
        fig.add_trace(go.Scatter(
            x=grp["week"], y=grp["revenue"],
            name=cat, stackgroup="one",
            mode="lines",
            fillcolor=cat_colors.get(cat, "#888"),
            line=dict(color=cat_colors.get(cat, "#888"), width=0.5),
            legendgroup=f"cat_{cat}",
        ), row=3, col=2)

    # ── Layout ─────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"Student Store Weekly Dashboard  ·  Week of {latest_week}",
            font=dict(size=18),
        ),
        height=1100,
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05, x=0),
        font=dict(size=11),
    )
    fig.update_xaxes(tickangle=-30)

    fig.write_html(output_path)
    print(f"  Dashboard saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AUTOMATED WEEKLY REPORT (text summary)
# Printed to stdout each run. Replaces the manual weekly tracking doc.
# ══════════════════════════════════════════════════════════════════════════════

def print_weekly_report(
    conn:            sqlite3.Connection,
    underperformers: pd.DataFrame,
    restock:         pd.DataFrame,
    turnover_df:     pd.DataFrame,
    forecasts:       pd.DataFrame,
    metrics:         dict,
) -> None:
    """Print a full automated weekly report to stdout."""

    # Revenue summary via SQL
    rev = query(conn, """
        SELECT
            phase,
            ROUND(AVG(total_revenue), 2) AS avg_daily_rev,
            ROUND(SUM(total_revenue), 2)  AS total_rev,
            COUNT(*)                      AS days
        FROM daily_summary
        GROUP BY phase
        ORDER BY MIN(date)
    """)

    # Latest week totals
    latest = query(conn, """
        SELECT SUM(revenue) AS revenue, SUM(units_sold) AS units
        FROM weekly_sales
        WHERE week = (SELECT MAX(week) FROM weekly_sales)
    """)

    print("\n" + "=" * 60)
    print("  STUDENT STORE — AUTOMATED WEEKLY REPORT")
    print("=" * 60)

    print("\n── Revenue by Phase ────────────────────────────────────")
    for _, row in rev.iterrows():
        label = {
            "baseline":        "Baseline (Aug–Oct)",
            "analysis":        "Analysis (Nov–Jan)",
            "post_strategy_1": "Strategy #1 (Jan–Mar)",
            "post_strategy_2": "Strategy #2 (Mar–Jun)",
        }.get(row["phase"], row["phase"])
        print(f"  {label:<28} Avg/day: ${row['avg_daily_rev']:.2f}  "
              f"Total: ${row['total_rev']:,.2f}  ({int(row['days'])} days)")

    # Turnover lift
    avg_baseline = turnover_df["baseline"].mean()
    avg_s2       = turnover_df["after_s2"].dropna().mean()
    pct_lift     = (avg_s2 - avg_baseline) / avg_baseline * 100
    print(f"\n── Inventory Turnover Lift ─────────────────────────────")
    print(f"  Avg turnover (baseline) : {avg_baseline:.2f}x")
    print(f"  Avg turnover (S2)       : {avg_s2:.2f}x")
    print(f"  Total lift              : +{pct_lift:.0f}%")

    print(f"\n── This Week ───────────────────────────────────────────")
    print(f"  Revenue    : ${latest['revenue'].iloc[0]:,.2f}")
    print(f"  Units sold : {latest['units'].iloc[0]:,}")

    print(f"\n── Underperforming SKUs (Restocking Targets) ───────────")
    under = underperformers[underperformers["underperforming"] == 1][
        ["name", "category", "turnover_ratio", "total_revenue"]
    ].head(8)
    print(under.to_string(index=False))

    print(f"\n── Strategy #1 Restock Recommendations ────────────────")
    print(restock[["name", "category", "current_par", "recommended_par",
                   "reorder_increase", "priority_score"]].head(8).to_string(index=False))

    print(f"\n── Next-Week Demand Forecast (Top 10) ──────────────────")
    print(f"  Model MAE: {metrics['mean_mae']} ± {metrics['std_mae']} units/week")
    top10 = forecasts.sort_values("forecast_units", ascending=False).head(10)
    print(top10[["sku", "forecast_units"]].to_string(index=False))

    print("\n" + "=" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT — run this script each week for a full automated report
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n── Building database ───────────────────────────────────")
    conn = build_database()

    print("\n── Running SKU turnover analysis (SQL) ─────────────────")
    underperformers = baseline_underperformers(conn)
    restock         = restock_recommendations(conn)
    turnover_df     = turnover_improvement(conn)
    cuts            = strategy2_cuts(conn)

    n_under = underperformers[underperformers["underperforming"] == 1].shape[0]
    print(f"  Underperforming SKUs flagged : {n_under}")
    print(f"  Strategy #1 restock targets  : {len(restock)}")
    print(f"  Strategy #2 cut candidates   : {len(cuts)}")

    print("\n── Training demand forecasting model (scikit-learn) ────")
    model, metrics, feature_df = train_demand_model(conn)
    forecasts = forecast_next_week(model, feature_df)

    print("\n── Generating Plotly dashboard ─────────────────────────")
    build_weekly_dashboard(conn, underperformers, turnover_df, forecasts)

    print("\n── Printing weekly report ──────────────────────────────")
    print_weekly_report(conn, underperformers, restock, turnover_df, forecasts, metrics)

    conn.close()
