import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build_recommendations(sku_stats):
    df = sku_stats.copy()

    df["prep_target"] = np.ceil(df["p75_qty"]).astype(int)
    df["prep_target_p90"] = np.ceil(df["p90_qty"]).astype(int)

    df["cv"] = df["std_qty"] / df["mean_qty"].replace(0, np.nan)

    high = (df["weeks_observed"] >= 40) & (df["cv"] < 1.0)
    medium = (df["weeks_observed"] >= 20) & ~high
    df["confidence"] = np.select([high, medium], ["High", "Medium"], default="Low")

    df["revenue_at_target"] = df["prep_target"] * df["price"]
    df["revenue_at_p90"] = df["prep_target_p90"] * df["price"]
    df["revenue_upside"] = df["revenue_at_p90"] - df["revenue_at_target"]

    very_low = df["mean_qty"] < 0.5
    volatile = df["cv"] > 1.5
    df["action"] = np.select(
        [very_low, volatile],
        ["Review - Very Low Volume", "Volatile - Monitor Closely"],
        default="Maintain",
    )

    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sku_stats = pd.read_csv(OUTPUT_DIR / "demand_stats_sku.csv")
    recs = build_recommendations(sku_stats)

    out = OUTPUT_DIR / "recommendations_raw.csv"
    recs.to_csv(out, index=False)

    assert (recs["prep_target"] >= 1).all(), "Found prep_target < 1"
    conf_counts = recs["confidence"].value_counts().to_dict()
    assert len(conf_counts) > 1, "All rows have the same confidence level — check thresholds"

    print(f"recommendations_raw.csv — {len(recs):,} rows")
    print(f"  confidence distribution: {conf_counts}")
    print(f"  action distribution:     {recs['action'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
