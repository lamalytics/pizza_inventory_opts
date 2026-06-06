import pandas as pd
from pathlib import Path

from load_data import load_all
from build_fact_table import build

OUTPUT_DIR = Path(__file__).parent.parent / "output"
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _agg_stats(group_df, group_cols):
    daily = (
        group_df.groupby(["date"] + group_cols + ["time_window"])["quantity"]
        .sum()
        .reset_index()
    )
    stats = (
        daily.groupby(group_cols + ["time_window"])["quantity"]
        .agg(
            mean_qty="mean",
            std_qty="std",
            p25_qty=lambda x: x.quantile(0.25),
            p75_qty=lambda x: x.quantile(0.75),
            p90_qty=lambda x: x.quantile(0.90),
            weeks_observed="count",
        )
        .reset_index()
    )
    stats["std_qty"] = stats["std_qty"].fillna(0)
    stats["day_of_week"] = pd.Categorical(
        stats["day_of_week"], categories=DOW_ORDER, ordered=True
    )
    return stats.sort_values(group_cols + ["time_window"]).reset_index(drop=True)


def aggregate_sku(fact):
    sku_cols = ["pizza_id", "pizza_type_id", "name", "category", "size", "price", "day_of_week"]
    return _agg_stats(fact, sku_cols)


def aggregate_category(fact):
    cat_cols = ["category", "day_of_week"]
    return _agg_stats(fact, cat_cols)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    orders, order_details, pizzas, pizza_types = load_all()
    fact = build(orders, order_details, pizzas, pizza_types)

    sku_stats = aggregate_sku(fact)
    cat_stats = aggregate_category(fact)

    sku_out = OUTPUT_DIR / "demand_stats_sku.csv"
    cat_out = OUTPUT_DIR / "demand_stats_category.csv"
    sku_stats.to_csv(sku_out, index=False)
    cat_stats.to_csv(cat_out, index=False)

    print(f"demand_stats_sku.csv      — {len(sku_stats):,} rows")
    print(f"demand_stats_category.csv — {len(cat_stats):,} rows")


if __name__ == "__main__":
    main()
