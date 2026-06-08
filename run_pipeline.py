import sys            # sys.exit() to halt the pipeline if validation fails
from pathlib import Path  # cross-platform file path construction

# Add the src/ directory to Python's module search path so the five pipeline
# modules can be imported by name (e.g. "import load_data") without installing them.
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import each pipeline step as a module so we can call individual functions
# rather than spawning subprocesses. This keeps all five steps in the same
# Python process, which is faster and makes the in-memory fact table reusable.
import load_data
import build_fact_table
import aggregate_demand
import build_recommendations
import generate_report
import pandas as pd  # used to read back intermediate CSVs when needed

# OUTPUT_DIR is resolved relative to this file (project root / output/).
OUTPUT_DIR = Path(__file__).parent / "output"


def main():
    # -------------------------------------------------------------------------
    # Step 1: Validate raw CSVs
    # -------------------------------------------------------------------------
    print("=== Step 1: Validate data ===")

    # load_all() reads the four source CSVs and returns them as DataFrames.
    orders, order_details, pizzas, pizza_types = load_data.load_all()

    # validate() checks for null dates, bad quantities, and referential integrity.
    # It returns a list of error strings — empty list means all checks passed.
    errors = load_data.validate(orders, order_details, pizzas, pizza_types)
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        # Exit immediately with code 1 so no downstream step runs on bad data.
        sys.exit(1)
    print("  Validation passed.\n")

    # -------------------------------------------------------------------------
    # Step 2: Build the fact table
    # -------------------------------------------------------------------------
    print("=== Step 2: Build fact table ===")

    # build() joins all four tables and adds derived columns:
    # day_of_week, week_number, time_window (Lunch/Dinner/Other), and revenue.
    fact = build_fact_table.build(orders, order_details, pizzas, pizza_types)

    OUTPUT_DIR.mkdir(exist_ok=True)  # create output/ if it does not exist

    # Write to disk so it can be inspected independently or reused by other tools.
    # index=False omits the pandas row index from the file.
    fact.to_csv(OUTPUT_DIR / "fact_table.csv", index=False)
    print(f"  {len(fact):,} rows, revenue ${fact['revenue'].sum():,.2f}\n")

    # -------------------------------------------------------------------------
    # Step 3: Aggregate demand statistics
    # -------------------------------------------------------------------------
    print("=== Step 3: Aggregate demand ===")

    # aggregate_sku() produces one row per (pizza × day_of_week × time_window)
    # with mean, std, P25, P75, P90, and weeks_observed.
    sku_stats = aggregate_demand.aggregate_sku(fact)

    # aggregate_category() produces the same stats at the category level
    # (Chicken / Classic / Supreme / Veggie) for the summary report.
    cat_stats = aggregate_demand.aggregate_category(fact)

    sku_stats.to_csv(OUTPUT_DIR / "demand_stats_sku.csv",      index=False)
    cat_stats.to_csv(OUTPUT_DIR / "demand_stats_category.csv", index=False)

    print(f"  SKU stats: {len(sku_stats):,} rows")
    print(f"  Category stats: {len(cat_stats):,} rows\n")

    # -------------------------------------------------------------------------
    # Step 4: Build prep recommendations
    # -------------------------------------------------------------------------
    print("=== Step 4: Build recommendations ===")

    # build_recommendations() takes the SKU-level stats and adds:
    # prep_target (ceil of P75), confidence (High/Medium/Low),
    # revenue impact columns, and an action flag.
    recs = build_recommendations.build_recommendations(sku_stats)

    recs.to_csv(OUTPUT_DIR / "recommendations_raw.csv", index=False)

    print(f"  {len(recs):,} rows")
    # Print confidence distribution as a quick sanity check.
    print(f"  Confidence: {recs['confidence'].value_counts().to_dict()}\n")

    # -------------------------------------------------------------------------
    # Step 5: Generate stakeholder reports
    # -------------------------------------------------------------------------
    print("=== Step 5: Generate reports ===")

    # write_csv() selects and renames columns for the manager-facing spreadsheet.
    generate_report.write_csv(recs)

    # write_reorder_guide() computes per-pizza reorder points and order quantities
    # based on the LEAD_DAYS and DELIVERIES_PER_WEEK constants in generate_report.py.
    generate_report.write_reorder_guide(recs)

    # write_markdown() computes all summary statistics and writes the narrative report.
    generate_report.write_markdown(fact, recs, cat_stats)
    print()

    print("Pipeline complete.")
    print("  Reports: reports/inventory_recommendations.csv")
    print("           reports/reorder_guide.csv")
    print("           reports/summary_report.md")


# Guard ensures main() only runs when this file is executed directly,
# not when it is imported by another script or test.
if __name__ == "__main__":
    main()
