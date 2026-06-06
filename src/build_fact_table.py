import numpy as np
import pandas as pd
from pathlib import Path

from load_data import load_all

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build(orders, order_details, pizzas, pizza_types):
    df = order_details.merge(orders, on="order_id", how="left")
    df = df.merge(pizzas, on="pizza_id", how="left")
    df = df.merge(pizza_types, on="pizza_type_id", how="left")

    df["day_of_week"] = df["date"].dt.day_name()
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)

    hour = pd.to_datetime(df["time"], format="%H:%M:%S").dt.hour
    df["time_window"] = np.select(
        [(hour >= 11) & (hour <= 14), (hour >= 17) & (hour <= 21)],
        ["Lunch", "Dinner"],
        default="Other",
    )

    df["revenue"] = df["quantity"] * df["price"]

    assert len(df) == 48620, f"Expected 48620 rows, got {len(df)}"
    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    orders, order_details, pizzas, pizza_types = load_all()
    df = build(orders, order_details, pizzas, pizza_types)

    out = OUTPUT_DIR / "fact_table.csv"
    df.to_csv(out, index=False)

    print(f"fact_table.csv written — {len(df):,} rows")
    print(f"  revenue total:       ${df['revenue'].sum():,.2f}")
    print(f"  time_window counts:  {df['time_window'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
