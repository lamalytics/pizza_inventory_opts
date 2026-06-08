import numpy as np   
import pandas as pd  
from pathlib import Path

# load_all() returns the four raw dataframes so no need to re-import CSVs
# when this module is used as a library by run_pipeline.py.
from load_data import load_all

# OUTPUT_DIR points two levels up from src/ to the project root, then into output/.
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build(orders, order_details, pizzas, pizza_types):
    # --- Join sequence ---
    # Start from order_details because it is the fact table grain:
    # one row per pizza line item per order.
    # Each merge is a left join so that every line item is preserved even if a
    # matching key is missing.

    # Step 1: attach order-level fields (date, time) to each line item.
    df = order_details.merge(orders, on="order_id", how="left")

    # Step 2: attach pizza-level fields (pizza_type_id, size, price) to each line item.
    df = df.merge(pizzas, on="pizza_id", how="left")

    # Step 3: attach type-level fields (name, category, ingredients) to each line item.
    df = df.merge(pizza_types, on="pizza_type_id", how="left")

    # --- Derived columns ---

    # returns full weekday name strings to be used as groupby category keys.
    df["day_of_week"] = df["date"].dt.day_name()

    # ISO week number (1–53). Used to count how many distinct weeks each pizza was sold,
    # which becomes the denominator for the conditional-mean calculation downstream.
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)

    # Parse the time column (stored as "HH:MM:SS" strings) into datetime objects
    # so we can extract the hour integer for time-window assignment.
    hour = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour

    # Assign each order line to an operational time window based on the hour of day.
    # np.select evaluates conditions in order and returns the first matching value.
    # Lunch  = 11:00–14:59 (covers the midday service period)
    # Dinner = 17:00–21:59 (covers the evening service period)
    # Other  = everything else (early morning, mid-afternoon, late night)
    df["time_window"] = np.select(
        [(hour >= 11) & (hour <= 14), (hour >= 17) & (hour <= 21)],
        ["Lunch", "Dinner"],
        default="Other",
    )

    # Revenue per line item = quantity ordered × unit price.
    # Must use quantity (not 1) because ~2% of line items have quantity > 1.
    df["revenue"] = df["quantity"] * df["price"]

    # Hard assertion: the join should produce exactly one output row per input row
    # in order_details. Any deviation means a join key mismatch caused row duplication
    # or row loss, which would corrupt every downstream calculation.
    assert len(df) == 48620, f"Expected 48620 rows, got {len(df)}"

    return df


def main():
    # Create the output directory if it does not already exist.
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load the four raw CSVs (validation is handled separately by load_data.py).
    orders, order_details, pizzas, pizza_types = load_all()

    # Build the enriched, joined fact table.
    df = build(orders, order_details, pizzas, pizza_types)

    out = OUTPUT_DIR / "fact_table.csv"
    # index=False omits the pandas row index from the CSV; it carries no meaning here.
    df.to_csv(out, index=False)

    print(f"fact_table.csv written — {len(df):,} rows")
    # Revenue total is printed as a quick sanity check; expected ~$817K for 2015.
    print(f"  revenue total:       ${df['revenue'].sum():,.2f}")
    # time_window counts reveal how much volume falls outside Lunch/Dinner.
    # "Other" should be a small minority; a large value signals a parsing issue.
    print(f"  time_window counts:  {df['time_window'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
