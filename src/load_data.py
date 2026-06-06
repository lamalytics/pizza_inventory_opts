import sys
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
# for quotations encoding ""
PIZZA_TYPES_ENCODING = "latin-1" 


def load_all():
    orders = pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["date"])
    order_details = pd.read_csv(DATA_DIR / "order_details.csv", dtype={"quantity": int})
    pizzas = pd.read_csv(DATA_DIR / "pizzas.csv")
    pizza_types = pd.read_csv(DATA_DIR / "pizza_types.csv", encoding=PIZZA_TYPES_ENCODING)
    return orders, order_details, pizzas, pizza_types


def validate(orders, order_details, pizzas, pizza_types):
    errors = []

    if orders["date"].isna().any():
        errors.append("orders: null dates found")
    if orders["date"].min().date().year != 2015 or orders["date"].max().date().year != 2015:
        errors.append(f"orders: date range outside 2015 ({orders['date'].min()} – {orders['date'].max()})")

    if order_details["quantity"].min() < 1:
        errors.append("order_details: quantity < 1 found")

    missing_orders = set(order_details["order_id"]) - set(orders["order_id"])
    if missing_orders:
        errors.append(f"order_details: {len(missing_orders)} order_ids not in orders")

    missing_pizzas = set(order_details["pizza_id"]) - set(pizzas["pizza_id"])
    if missing_pizzas:
        errors.append(f"order_details: {len(missing_pizzas)} pizza_ids not in pizzas: {missing_pizzas}")

    missing_types = set(pizzas["pizza_type_id"]) - set(pizza_types["pizza_type_id"])
    if missing_types:
        errors.append(f"pizzas: {len(missing_types)} pizza_type_ids not in pizza_types: {missing_types}")

    return errors


def main():
    print("Loading CSVs...")
    orders, order_details, pizzas, pizza_types = load_all()

    print(f"  orders:        {len(orders):,} rows")
    print(f"  order_details: {len(order_details):,} rows")
    print(f"  pizzas:        {len(pizzas):,} rows")
    print(f"  pizza_types:   {len(pizza_types):,} rows")

    errors = validate(orders, order_details, pizzas, pizza_types)

    if errors:
        print("\nVALIDATION FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\nAll validations passed.")


if __name__ == "__main__":
    main()
