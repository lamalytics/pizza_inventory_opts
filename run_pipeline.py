import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import load_data
import build_fact_table
import aggregate_demand
import build_recommendations
import generate_report
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    print("=== Step 1: Validate data ===")
    orders, order_details, pizzas, pizza_types = load_data.load_all()
    errors = load_data.validate(orders, order_details, pizzas, pizza_types)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    print("  Validation passed.\n")

    print("=== Step 2: Build fact table ===")
    fact = build_fact_table.build(orders, order_details, pizzas, pizza_types)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fact.to_csv(OUTPUT_DIR / "fact_table.csv", index=False)
    print(f"  {len(fact):,} rows, revenue ${fact['revenue'].sum():,.2f}\n")

    print("=== Step 3: Aggregate demand ===")
    sku_stats = aggregate_demand.aggregate_sku(fact)
    cat_stats = aggregate_demand.aggregate_category(fact)
    sku_stats.to_csv(OUTPUT_DIR / "demand_stats_sku.csv", index=False)
    cat_stats.to_csv(OUTPUT_DIR / "demand_stats_category.csv", index=False)
    print(f"  SKU stats: {len(sku_stats):,} rows")
    print(f"  Category stats: {len(cat_stats):,} rows\n")

    print("=== Step 4: Build recommendations ===")
    recs = build_recommendations.build_recommendations(sku_stats)
    recs.to_csv(OUTPUT_DIR / "recommendations_raw.csv", index=False)
    print(f"  {len(recs):,} rows")
    print(f"  Confidence: {recs['confidence'].value_counts().to_dict()}\n")

    print("=== Step 5: Generate reports ===")
    generate_report.write_csv(recs)
    generate_report.write_markdown(fact, recs, cat_stats)
    print()

    print("Pipeline complete.")
    print("  Reports: reports/inventory_recommendations.csv")
    print("           reports/summary_report.md")


if __name__ == "__main__":
    main()
