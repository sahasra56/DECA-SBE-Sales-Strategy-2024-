"""
generate_data.py
----------------
Generates a full school-year POS dataset for a DECA School-Based Enterprise
(student store) running August 2023 – June 2024.

Project timeline:
  Aug–Oct 2023    : Baseline. Store running normally. Italian sodas & custom
                    drinks underselling — stockouts happening mid-day but
                    nobody had identified the root cause yet.
  Late Oct 2023   : DECA project kicks off. POS data pulled, SKU turnover
                    analysis started, stockout patterns identified.
  Nov–Dec 2023    : Deep analysis. Demand forecasting models built, lost
                    revenue from stockouts quantified, Strategy #1 drafted.
  Jan 22, 2024    : STRATEGY #1 — Restock. Raised par levels on Italian sodas
                    and custom drinks, fixed reorder points. Big jump in revenue
                    as suppressed demand is finally captured.
  Feb–Mar 2024    : Forecasting dashboard goes live. Monitoring weekly data
                    reveals a second opportunity: chips & candy moving faster
                    than inventory can keep up, AND 3 slow SKUs tying up budget.
  Mar 17, 2024    : STRATEGY #2 — Optimize. Automated reorder pipeline deployed.
                    Slow SKUs cut (Pretzels, Altoids, Rice Krispie Treat replaced
                    with higher-demand options). Takis already added Feb 5 as
                    a test — confirmed strong, so inventory doubled. Second lift.
  Mar–Jun 2024    : Sustained post-strategy-2 period. Cleaner inventory, better
                    margins, automated reporting frees up time.

Prices: all end in .00, .25, .50, or .75
"""

import pandas as pd
import numpy as np
import os

np.random.seed(2024)
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ── 1. PRODUCT CATALOG ─────────────────────────────────────────────────────────
# inv_pre     : avg daily inventory before Strategy #1
# inv_s1      : avg daily inventory after Strategy #1 (Jan 22)
# inv_s2      : avg daily inventory after Strategy #2 (Mar 17)
# stockout_rate: prob of hitting stock ceiling pre-strategy (suppresses sales)

PRODUCTS = [
    # Italian Sodas — star category, high margin. Chronically understocked
    # pre-strategy. Stockouts cut off sales mid-day, making them look slow.
    {"sku": "ITS-STR", "name": "Strawberry Italian Soda",    "category": "Italian Soda",
     "price": 3.50, "inv_pre": 4,  "inv_s1": 14, "inv_s2": 14, "base_demand": 4,
     "seasonal": {9:1.2,10:1.1,3:1.1,4:1.2,5:1.3}, "stockout_rate": 0.32,
     "introduced": None, "discontinued": None},
    {"sku": "ITS-PAS", "name": "Passion Fruit Italian Soda", "category": "Italian Soda",
     "price": 3.50, "inv_pre": 3,  "inv_s1": 12, "inv_s2": 12, "base_demand": 3,
     "seasonal": {9:1.2,10:1.1,4:1.1,5:1.2},       "stockout_rate": 0.32,
     "introduced": None, "discontinued": None},
    {"sku": "ITS-BLU", "name": "Blueberry Italian Soda",     "category": "Italian Soda",
     "price": 3.50, "inv_pre": 3,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {4:1.1,5:1.2},                    "stockout_rate": 0.28,
     "introduced": None, "discontinued": None},
    {"sku": "ITS-PEA", "name": "Peach Italian Soda",         "category": "Italian Soda",
     "price": 3.50, "inv_pre": 3,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {5:1.2,6:1.3},                    "stockout_rate": 0.28,
     "introduced": None, "discontinued": None},
    {"sku": "ITS-COC", "name": "Coconut Italian Soda",       "category": "Italian Soda",
     "price": 3.50, "inv_pre": 2,  "inv_s1": 8,  "inv_s2": 8,  "base_demand": 1,
     "seasonal": {},                                "stockout_rate": 0.35,
     "introduced": None, "discontinued": None},

    # Custom Drinks — highest margin. Same stockout problem as Italian sodas.
    # Horchata Cold Brew added Nov 6 during analysis phase as a new menu item.
    {"sku": "DRK-MAT", "name": "Matcha Lemonade",            "category": "Custom Drink",
     "price": 4.25, "inv_pre": 4,  "inv_s1": 12, "inv_s2": 12, "base_demand": 3,
     "seasonal": {3:1.1,4:1.2,5:1.3},              "stockout_rate": 0.25,
     "introduced": None, "discontinued": None},
    {"sku": "DRK-STR", "name": "Strawberry Milk",            "category": "Custom Drink",
     "price": 3.75, "inv_pre": 4,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {},                                "stockout_rate": 0.20,
     "introduced": None, "discontinued": None},
    {"sku": "DRK-HOR", "name": "Horchata Cold Brew",         "category": "Custom Drink",
     "price": 4.50, "inv_pre": 3,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {11:1.1,12:1.2,1:1.2,2:1.1},      "stockout_rate": 0.20,
     "introduced": "2023-11-06", "discontinued": None},
    {"sku": "DRK-CHI", "name": "Chia Lemonade",              "category": "Custom Drink",
     "price": 3.75, "inv_pre": 2,  "inv_s1": 8,  "inv_s2": 8,  "base_demand": 1,
     "seasonal": {4:1.1,5:1.2},                    "stockout_rate": 0.30,
     "introduced": None, "discontinued": None},

    # Energy Drinks — reliable year-round. Finals weeks (Dec, May) spike.
    # Red Bull price bumped to $4.00 in March (supply cost increase).
    {"sku": "ENR-RBU", "name": "Red Bull Original",          "category": "Energy Drink",
     "price": 3.75, "inv_pre": 12, "inv_s1": 18, "inv_s2": 20, "base_demand": 4,
     "seasonal": {12:1.2,1:1.1,5:1.2},             "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},
    {"sku": "ENR-RBS", "name": "Red Bull Sugar Free",        "category": "Energy Drink",
     "price": 3.75, "inv_pre": 10, "inv_s1": 14, "inv_s2": 16, "base_demand": 3,
     "seasonal": {12:1.1,5:1.1},                   "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},
    {"sku": "ENR-MTN", "name": "Monster Energy",             "category": "Energy Drink",
     "price": 3.50, "inv_pre": 10, "inv_s1": 14, "inv_s2": 16, "base_demand": 3,
     "seasonal": {},                                "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},
    {"sku": "ENR-CEL", "name": "Celsius",                    "category": "Energy Drink",
     "price": 3.25, "inv_pre": 10, "inv_s1": 16, "inv_s2": 18, "base_demand": 3,
     "seasonal": {3:1.1,4:1.1,5:1.2},              "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},

    # Soft Drinks — water is #1 by units, steady all year.
    {"sku": "SFT-WAT", "name": "Water Bottle",               "category": "Soft Drink",
     "price": 1.25, "inv_pre": 24, "inv_s1": 30, "inv_s2": 30, "base_demand": 7,
     "seasonal": {8:1.2,9:1.2,5:1.2,6:1.2},        "stockout_rate": 0.01,
     "introduced": None, "discontinued": None},
    {"sku": "SFT-COK", "name": "Coca-Cola",                  "category": "Soft Drink",
     "price": 1.75, "inv_pre": 16, "inv_s1": 20, "inv_s2": 20, "base_demand": 5,
     "seasonal": {},                                "stockout_rate": 0.01,
     "introduced": None, "discontinued": None},
    {"sku": "SFT-SPR", "name": "Sprite",                     "category": "Soft Drink",
     "price": 1.75, "inv_pre": 12, "inv_s1": 16, "inv_s2": 16, "base_demand": 4,
     "seasonal": {},                                "stockout_rate": 0.01,
     "introduced": None, "discontinued": None},
    {"sku": "SFT-DRP", "name": "Dr Pepper",                  "category": "Soft Drink",
     "price": 1.75, "inv_pre": 10, "inv_s1": 12, "inv_s2": 12, "base_demand": 3,
     "seasonal": {},                                "stockout_rate": 0.01,
     "introduced": None, "discontinued": None},
    {"sku": "SFT-LEM", "name": "Lemonade",                   "category": "Soft Drink",
     "price": 1.75, "inv_pre": 10, "inv_s1": 14, "inv_s2": 14, "base_demand": 2,
     "seasonal": {4:1.1,5:1.2,6:1.2},              "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},

    # Baked Goods — fresh baked, small batches. Choc chip is #1.
    # Pumpkin Muffin: fall seasonal only (Sep–Nov).
    # Sugar cookie spikes in Dec (holidays) and Feb (Valentine's Day).
    # Rice Krispie Treat cut in Strategy #2 — low margin, slow relative to shelf space.
    {"sku": "SNK-CHC", "name": "Chocolate Chip Cookie",      "category": "Baked Good",
     "price": 2.00, "inv_pre": 12, "inv_s1": 16, "inv_s2": 18, "base_demand": 5,
     "seasonal": {},                                "stockout_rate": 0.08,
     "introduced": None, "discontinued": None},
    {"sku": "SNK-BRW", "name": "Brownie",                    "category": "Baked Good",
     "price": 2.50, "inv_pre": 8,  "inv_s1": 12, "inv_s2": 14, "base_demand": 3,
     "seasonal": {12:1.1,2:1.1},                   "stockout_rate": 0.08,
     "introduced": None, "discontinued": None},
    {"sku": "SNK-SUG", "name": "Sugar Cookie",               "category": "Baked Good",
     "price": 2.00, "inv_pre": 8,  "inv_s1": 10, "inv_s2": 12, "base_demand": 2,
     "seasonal": {12:1.3,2:1.4},                   "stockout_rate": 0.08,
     "introduced": None, "discontinued": None},
    {"sku": "SNK-PMP", "name": "Pumpkin Muffin",             "category": "Baked Good",
     "price": 2.75, "inv_pre": 6,  "inv_s1": 6,  "inv_s2": 6,  "base_demand": 2,
     "seasonal": {9:1.1,10:1.2},                   "stockout_rate": 0.05,
     "introduced": None, "discontinued": "2023-11-22"},
    {"sku": "SNK-RCK", "name": "Rice Krispie Treat",         "category": "Baked Good",
     "price": 1.50, "inv_pre": 8,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {},                                "stockout_rate": 0.05,
     "introduced": None, "discontinued": "2024-03-16"},  # cut in Strategy #2
    {"sku": "SNK-LMB", "name": "Lemon Bar",                  "category": "Baked Good",
     "price": 2.25, "inv_pre": 0,  "inv_s1": 0,  "inv_s2": 10, "base_demand": 3,
     "seasonal": {4:1.1,5:1.2},                    "stockout_rate": 0.05,
     "introduced": "2024-03-17", "discontinued": None},  # added in Strategy #2

    # Candy & Gum — impulse buys at register.
    # Altoids cut in Strategy #2 (slow mover, low margin).
    # Sour Patch and Starburst spike Valentine's Day; Skittles spikes Halloween.
    {"sku": "CND-GUM", "name": "Orbit Gum",                  "category": "Candy & Gum",
     "price": 1.50, "inv_pre": 14, "inv_s1": 16, "inv_s2": 18, "base_demand": 3,
     "seasonal": {},                                "stockout_rate": 0.02,
     "introduced": None, "discontinued": None},
    {"sku": "CND-RBB", "name": "Sour Patch Kids",            "category": "Candy & Gum",
     "price": 2.00, "inv_pre": 10, "inv_s1": 14, "inv_s2": 18, "base_demand": 3,
     "seasonal": {2:1.4,10:1.2},                   "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},
    {"sku": "CND-SKT", "name": "Skittles",                   "category": "Candy & Gum",
     "price": 1.75, "inv_pre": 10, "inv_s1": 12, "inv_s2": 16, "base_demand": 2,
     "seasonal": {10:1.2},                         "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},
    {"sku": "CND-STB", "name": "Starburst",                  "category": "Candy & Gum",
     "price": 1.75, "inv_pre": 10, "inv_s1": 12, "inv_s2": 16, "base_demand": 2,
     "seasonal": {2:1.5},                          "stockout_rate": 0.04,
     "introduced": None, "discontinued": None},
    {"sku": "CND-MNT", "name": "Altoids",                    "category": "Candy & Gum",
     "price": 2.00, "inv_pre": 8,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {},                                "stockout_rate": 0.02,
     "introduced": None, "discontinued": "2024-03-16"},  # cut in Strategy #2
    {"sku": "CND-GBR", "name": "Gummy Bears",                "category": "Candy & Gum",
     "price": 2.00, "inv_pre": 0,  "inv_s1": 0,  "inv_s2": 14, "base_demand": 3,
     "seasonal": {},                                "stockout_rate": 0.04,
     "introduced": "2024-03-17", "discontinued": None},  # added in Strategy #2

    # Chips & Savory — Doritos and Lays workhorses.
    # Takis added Feb 5 as a test run; confirmed strong seller by strategy #2.
    # Pretzels cut in Strategy #2 — bottom performer by turnover ratio.
    {"sku": "CHP-LAY", "name": "Lays Classic",               "category": "Chips",
     "price": 1.50, "inv_pre": 14, "inv_s1": 16, "inv_s2": 18, "base_demand": 4,
     "seasonal": {},                                "stockout_rate": 0.02,
     "introduced": None, "discontinued": None},
    {"sku": "CHP-DRT", "name": "Doritos Nacho",              "category": "Chips",
     "price": 1.50, "inv_pre": 12, "inv_s1": 14, "inv_s2": 18, "base_demand": 4,
     "seasonal": {},                                "stockout_rate": 0.02,
     "introduced": None, "discontinued": None},
    {"sku": "CHP-TAK", "name": "Takis Fuego",                "category": "Chips",
     "price": 1.75, "inv_pre": 0,  "inv_s1": 10, "inv_s2": 18, "base_demand": 4,
     "seasonal": {},                                "stockout_rate": 0.04,
     "introduced": "2024-02-05", "discontinued": None},
    {"sku": "CHP-PRT", "name": "Pretzels",                   "category": "Chips",
     "price": 1.25, "inv_pre": 8,  "inv_s1": 10, "inv_s2": 10, "base_demand": 2,
     "seasonal": {},                                "stockout_rate": 0.02,
     "introduced": None, "discontinued": "2024-03-16"},  # cut — bottom turnover
    {"sku": "CHP-POP", "name": "Popcorn",                    "category": "Chips",
     "price": 1.25, "inv_pre": 8,  "inv_s1": 10, "inv_s2": 12, "base_demand": 2,
     "seasonal": {},                                "stockout_rate": 0.02,
     "introduced": None, "discontinued": None},
]

# Price change: Red Bull bumped $0.25 in March after supply cost increase
PRICE_CHANGES = {
    pd.Timestamp("2024-03-04"): {"ENR-RBU": 4.00, "ENR-RBS": 4.00},
}

products_df = pd.DataFrame(PRODUCTS)

# ── 2. SCHOOL CALENDAR  Aug 21 2023 – Jun 14 2024 ─────────────────────────────

all_bdays = pd.bdate_range("2023-08-21", "2024-06-14")

SCHOOL_BREAKS = set(pd.to_datetime([
    *pd.bdate_range("2023-11-20", "2023-11-24"),  # Thanksgiving
    *pd.bdate_range("2023-12-18", "2024-01-05"),  # Winter break
    "2024-01-15",                                  # MLK Day
    "2024-02-19",                                  # Presidents Day
    *pd.bdate_range("2024-03-25", "2024-03-29"),  # Spring break
    "2024-05-27",                                  # Memorial Day
]))

school_days = pd.DatetimeIndex([d for d in all_bdays if d not in SCHOOL_BREAKS])

PROJECT_START   = pd.Timestamp("2023-10-23")  # DECA project kicks off
STRATEGY_1_DATE = pd.Timestamp("2024-01-22")  # Strategy #1: restock high-demand SKUs
STRATEGY_2_DATE = pd.Timestamp("2024-03-17")  # Strategy #2: optimize + automate + cut slow SKUs

# ── 3. SPECIAL DAYS ────────────────────────────────────────────────────────────

SPIKE_DAYS = set(pd.to_datetime([
    "2023-09-08",   # back-to-school spirit day
    "2023-10-13",   # homecoming week
    "2023-10-27",   # Halloween dress-up day
    "2023-12-15",   # last day before winter break
    "2024-02-14",   # Valentine's Day
    "2024-03-15",   # Pi Day / store fundraiser
    "2024-04-05",   # spring spirit week
    "2024-05-17",   # prom week
    "2024-06-13",   # last day of school
]))

SLOW_DAYS = set(pd.to_datetime([
    "2023-08-21",   # first day — store not fully running
    "2023-09-20",   # all-school assembly, short lunch
    "2023-10-06",   # cold snap, low foot traffic
    "2023-11-17",   # pre-Thanksgiving slowdown
    "2024-01-08",   # first week back, sluggish return
    "2024-01-22",   # strategy #1 rollout — store reorganized, closed early
    "2024-03-01",   # standardized testing week
    "2024-03-04",
    "2024-03-05",
    "2024-03-17",   # strategy #2 rollout — minor disruption
    "2024-04-19",   # spring testing
    "2024-05-03",   # AP exam week
    "2024-05-06",
]))

# ── 4. GENERATE DAILY TRANSACTIONS ─────────────────────────────────────────────

def get_phase(date):
    if date >= STRATEGY_2_DATE:
        return "post_strategy_2"
    elif date >= STRATEGY_1_DATE:
        return "post_strategy_1"
    elif date >= PROJECT_START:
        return "analysis"
    else:
        return "baseline"

records = []

for date in school_days:
    phase = get_phase(date)
    month = date.month
    dow   = date.dayofweek

    dow_mult = {0: 0.85, 1: 1.05, 2: 1.10, 3: 1.05, 4: 0.92}[dow]

    if date in SPIKE_DAYS:
        day_mult = np.random.uniform(1.35, 1.60)
    elif date in SLOW_DAYS:
        day_mult = np.random.uniform(0.50, 0.72)
    else:
        day_mult = np.random.choice(
            [0.68, 0.80, 0.92, 1.00, 1.08, 1.20, 1.35],
            p=[0.05, 0.10, 0.18, 0.32, 0.18, 0.11, 0.06]
        )

    current_prices = {}
    for change_ts, changes in PRICE_CHANGES.items():
        if date >= change_ts:
            current_prices.update(changes)

    for _, prod in products_df.iterrows():
        sku = prod["sku"]

        if prod["introduced"] and date < pd.Timestamp(prod["introduced"]):
            continue
        if prod["discontinued"] and date > pd.Timestamp(prod["discontinued"]):
            continue

        base   = prod["base_demand"]
        season = prod["seasonal"].get(month, 1.0)
        price  = current_prices.get(sku, prod["price"])

        # Inventory level and strategy boost by phase
        if phase == "post_strategy_2":
            inv = prod["inv_s2"]
            if prod["category"] in ("Italian Soda", "Custom Drink"):
                boost = np.random.uniform(2.4, 3.0)
            elif prod["category"] in ("Chips", "Candy & Gum"):
                boost = np.random.uniform(1.25, 1.45)  # second lift from optimized chips/candy
            else:
                boost = np.random.uniform(1.10, 1.25)
        elif phase == "post_strategy_1":
            inv = prod["inv_s1"]
            if prod["category"] in ("Italian Soda", "Custom Drink"):
                boost = np.random.uniform(2.2, 2.8)
            else:
                boost = np.random.uniform(1.05, 1.18)
        else:
            inv = prod["inv_pre"]
            boost = 1.0

        if inv == 0:
            continue

        raw    = base * dow_mult * day_mult * season * boost
        demand = max(0, int(np.random.normal(raw, raw * 0.28 + 0.5)))

        # Pre-strategy-1 stockouts suppress recorded sales
        if phase in ("baseline", "analysis") and np.random.random() < prod["stockout_rate"]:
            demand = min(demand, max(0, inv // 3))

        demand = min(demand, inv)

        if demand == 0:
            continue

        records.append({
            "date":       date.strftime("%Y-%m-%d"),
            "sku":        sku,
            "name":       prod["name"],
            "category":   prod["category"],
            "units_sold": demand,
            "unit_price": price,
            "revenue":    round(demand * price, 2),
            "phase":      phase,
        })

pos_df = pd.DataFrame(records)
pos_df["date"] = pd.to_datetime(pos_df["date"])

# ── 5. VERIFY PRICES ───────────────────────────────────────────────────────────

bad_prices = pos_df[~pos_df["unit_price"].apply(
    lambda p: round(p * 4) == p * 4  # must be multiple of 0.25
)]
if len(bad_prices) > 0:
    print(f"WARNING: {len(bad_prices)} rows with non-.25 prices")
    print(bad_prices[["sku","unit_price"]].drop_duplicates())
else:
    print("✓ All prices end in .00 / .25 / .50 / .75")

# ── 6. SUMMARY ─────────────────────────────────────────────────────────────────

daily = pos_df.groupby("date")["revenue"].sum()
base  = daily[daily.index < PROJECT_START]
anlys = daily[(daily.index >= PROJECT_START) & (daily.index < STRATEGY_1_DATE)]
s1    = daily[(daily.index >= STRATEGY_1_DATE) & (daily.index < STRATEGY_2_DATE)]
s2    = daily[daily.index >= STRATEGY_2_DATE]

print("\n" + "=" * 58)
print("  DATASET SUMMARY")
print("=" * 58)
print(f"\n  Total school days : {len(school_days)}")
print(f"\n  PHASE 1 — Baseline     (Aug 21 – Oct 22, 2023): {len(base)} days")
print(f"    Avg ${base.mean():.2f}  |  Min ${base.min():.2f}  |  Max ${base.max():.2f}")
print(f"\n  PHASE 2 — Analysis     (Oct 23 – Jan 21, 2024): {len(anlys)} days")
print(f"    Avg ${anlys.mean():.2f}  |  Min ${anlys.min():.2f}  |  Max ${anlys.max():.2f}")
print(f"\n  PHASE 3 — Strategy #1  (Jan 22 – Mar 16, 2024): {len(s1)} days")
print(f"    Avg ${s1.mean():.2f}  |  Min ${s1.min():.2f}  |  Max ${s1.max():.2f}")
print(f"\n  PHASE 4 — Strategy #2  (Mar 17 – Jun 14, 2024): {len(s2)} days")
print(f"    Avg ${s2.mean():.2f}  |  Min ${s2.min():.2f}  |  Max ${s2.max():.2f}")
print(f"\n  Total lift (baseline → S2) : +{(s2.mean()-base.mean())/base.mean()*100:.1f}%")
print(f"  S1 lift alone              : +{(s1.mean()-base.mean())/base.mean()*100:.1f}%")
print(f"  Additional S2 lift         : +{(s2.mean()-s1.mean())/s1.mean()*100:.1f}%")
print(f"\n  Total transactions : {len(pos_df):,}")
print(f"  Unique SKUs active : {pos_df['sku'].nunique()}")
print("=" * 58)

# ── 7. SAVE ────────────────────────────────────────────────────────────────────

pos_df.to_csv("data/pos_transactions.csv", index=False)
print(f"\n-> data/pos_transactions.csv  ({len(pos_df):,} rows)")

pos_df["week"] = pos_df["date"].dt.to_period("W").apply(
    lambda r: r.start_time.strftime("%Y-%m-%d")
)
weekly_df = (
    pos_df.groupby(["week", "sku", "category", "phase"])
    .agg(units_sold=("units_sold", "sum"), revenue=("revenue", "sum"))
    .reset_index()
)
weekly_df.to_csv("data/weekly_sales.csv", index=False)
print(f"-> data/weekly_sales.csv      ({len(weekly_df):,} rows)")

daily_df = (
    pos_df.groupby(["date", "phase"])
    .agg(total_units=("units_sold", "sum"), total_revenue=("revenue", "sum"))
    .reset_index()
)
daily_df["day_of_week"] = pd.to_datetime(daily_df["date"]).dt.day_name()
daily_df.to_csv("data/daily_summary.csv", index=False)
print(f"-> data/daily_summary.csv     ({len(daily_df):,} rows)")

catalog_df = products_df[["sku","name","category","price",
                           "inv_pre","inv_s1","inv_s2","base_demand"]].copy()
catalog_df.to_csv("data/sku_catalog.csv", index=False)
print(f"-> data/sku_catalog.csv       ({len(catalog_df)} SKUs)")
