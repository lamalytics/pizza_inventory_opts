import sys          # used to exit with a non-zero code if validation fails
import pandas as pd  # all CSV loading and dataframe validation
from pathlib import Path  # cross-platform file path construction

# DATA_DIR points two levels up from this file (src/ → project root) then into data/
DATA_DIR = Path(__file__).parent.parent / "data"

PIZZA_TYPES_ENCODING = "latin-1"  # for quotations encoding ""


def load_all():
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["date"])

    # Enforces integer type on ingest.
    # Without this, pandas may infer float if any value is missing, causing silent errors in sums.
    order_details = pd.read_csv(DATA_DIR / "order_details.csv")

    pizzas = pd.read_csv(DATA_DIR / "pizzas.csv")

    # Must use latin-1 here
    pizza_types = pd.read_csv(DATA_DIR / "pizza_types.csv", encoding=PIZZA_TYPES_ENCODING)

    return orders, order_details, pizzas, pizza_types


def validate(orders, order_details, pizzas, pizza_types):
    # Collect all validation failures before returning so the caller sees every problem at once.
    errors = []

    # Null dates would break all downstream .dt operations and groupby logic.
    if orders["date"].isna().any():
        errors.append("orders: null dates found")

    # A quantity of zero or less would corrupt demand totals silently.
    if order_details["quantity"].min() < 1:
        errors.append("order_details: quantity < 1 found")

    # Set difference: every order_id in order_details must exist in orders.
    # A missing order_id means we have line items with no parent order (orphaned rows).
    missing_orders = set(order_details["order_id"]) - set(orders["order_id"])
    if missing_orders:
        errors.append(f"order_details: {len(missing_orders)} order_ids not in orders")

    # Every pizza_id in order_details must exist in pizzas so price and size joins don't produce NaNs.
    missing_pizzas = set(order_details["pizza_id"]) - set(pizzas["pizza_id"])
    if missing_pizzas:
        errors.append(f"order_details: {len(missing_pizzas)} pizza_ids not in pizzas: {missing_pizzas}")

    # Every pizza_type_id in pizzas must exist in pizza_types so category and ingredient joins work.
    missing_types = set(pizzas["pizza_type_id"]) - set(pizza_types["pizza_type_id"])
    if missing_types:
        errors.append(f"pizzas: {len(missing_types)} pizza_type_ids not in pizza_types: {missing_types}")

    return errors


def main():
    print("Loading CSVs...")
    orders, order_details, pizzas, pizza_types = load_all()

    # Print row counts as a sanity check before validation runs.
    print(f"  orders:        {len(orders):,} rows")
    print(f"  order_details: {len(order_details):,} rows")
    print(f"  pizzas:        {len(pizzas):,} rows")
    print(f"  pizza_types:   {len(pizza_types):,} rows")

    errors = validate(orders, order_details, pizzas, pizza_types)

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        # Exit code 1 signals failure to any calling script or CI runner.
        sys.exit(1)
    else:
        print("\nAll validations passed.")


# Only run main() when this file is executed directly (e.g. python3 src/load_data.py).
# When other scripts import load_all() or validate(), main() does not run automatically.
if __name__ == "__main__":
    main()
