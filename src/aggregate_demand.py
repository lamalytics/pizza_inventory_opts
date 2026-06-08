import pandas as pd   # groupby, aggregation, and Categorical ordering
from pathlib import Path  # cross-platform file path construction

# load_all and build are imported so this script can be run standalone (python3 src/aggregate_demand.py).
# When run_pipeline.py calls aggregate_sku/aggregate_category directly, it passes the already-built fact table.
from load_data import load_all
from build_fact_table import build

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# Explicit weekday ordering used to make CSVs sort Monday–Sunday instead of alphabetically.
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _agg_stats(group_df, group_cols):
    """
    Two-step aggregation that produces demand statistics per (group_cols + time_window).

    Step 1 — collapse to one row per calendar date:
        Summing quantity per date prevents a single multi-pizza order from
        inflating the per-day average. We want "how many of this pizza sold
        on this specific day in this time window", not "how many line items existed".

    Step 2 — average across all occurrences of each weekday:
        e.g. "across all 52 Mondays in 2015, what was the typical Lunch demand?"
        Only days where at least one unit was sold appear (conditional mean).
        The weeks_observed column makes this denominator visible to the reader.
    """

    # --- Step 1: one row per (date, group_cols, time_window) ---
    # group_df is the full fact table (or a subset of it).
    # ["date"] is prepended so each calendar day is its own row before averaging.
    # .sum() on quantity is critical — never .count() on rows, because one order
    # line can have quantity > 1.
    daily = (
        group_df.groupby(["date"] + group_cols + ["time_window"])["quantity"]
        .sum()
        .reset_index()  # flatten MultiIndex back into a flat DataFrame
    )

    # --- Step 2: aggregate across all Mondays, Tuesdays, etc. ---
    # The result of Step 1 already has one row per day, so averaging here gives
    # the mean daily demand across all weeks that had any sales.
    stats = (
        daily.groupby(group_cols + ["time_window"])["quantity"]
        .agg(
            mean_qty="mean",              # average daily quantity sold (on days with sales)
            std_qty="std",                # standard deviation — measures day-to-day variability
            p25_qty=lambda x: x.quantile(0.25),  # 25th percentile: a low-demand day
            p75_qty=lambda x: x.quantile(0.75),  # 75th percentile: used as the prep target
            p90_qty=lambda x: x.quantile(0.90),  # 90th percentile: the "aggressive" prep option
            weeks_observed="count",       # how many days had at least one sale (the denominator)
        )
        .reset_index()
    )

    # std_qty is NaN when a pizza appears on only one day (std of a single value is undefined).
    # Fill with 0 so downstream CV calculations don't silently propagate NaN.
    stats["std_qty"] = stats["std_qty"].fillna(0)

    # Convert day_of_week to an ordered Categorical so that sort_values()
    # produces Mon → Tue → ... → Sun instead of alphabetical order.
    # This only works if group_cols includes "day_of_week".
    stats["day_of_week"] = pd.Categorical(
        stats["day_of_week"], categories=DOW_ORDER, ordered=True
    )

    # Sort by the group columns first, then time_window, so the output CSV is
    # readable without further sorting (e.g. all rows for a pizza are grouped together).
    return stats.sort_values(group_cols + ["time_window"]).reset_index(drop=True)


def aggregate_sku(fact):
    # SKU grain: one row per (pizza × day_of_week × time_window).
    # price is included in the groupby columns so it passes through to the output
    # and can be used for revenue calculations in build_recommendations.py.
    sku_cols = ["pizza_id", "pizza_type_id", "name", "category", "size", "price", "day_of_week"]
    return _agg_stats(fact, sku_cols)


def aggregate_category(fact):
    # Category grain: one row per (category × day_of_week × time_window).
    # Used in the summary report to show peak vs. slowest day ratios per category.
    cat_cols = ["category", "day_of_week"]
    return _agg_stats(fact, cat_cols)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Reload raw data and rebuild the fact table so this script is self-contained.
    orders, order_details, pizzas, pizza_types = load_all()
    fact = build(orders, order_details, pizzas, pizza_types)

    sku_stats = aggregate_sku(fact)
    cat_stats = aggregate_category(fact)

    sku_out = OUTPUT_DIR / "demand_stats_sku.csv"
    cat_out = OUTPUT_DIR / "demand_stats_category.csv"

    # index=False omits the pandas row index; it has no analytical meaning here.
    sku_stats.to_csv(sku_out, index=False)
    cat_stats.to_csv(cat_out, index=False)

    print(f"demand_stats_sku.csv      — {len(sku_stats):,} rows")
    print(f"demand_stats_category.csv — {len(cat_stats):,} rows")


if __name__ == "__main__":
    main()
