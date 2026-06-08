import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def build_recommendations(sku_stats):
    # Work on a copy so the original sku_stats DataFrame is not mutated.
    # This matters when run_pipeline.py passes the same object to multiple steps.
    df = sku_stats.copy()

    # --- Prep targets ---
    # P75 = ceiling of the 75th percentile daily demand. Covers most shifts;
    # spike days handled via same-day restocking. np.ceil ensures we never
    # prep a fractional unit (2.3 → 3).
    df["prep_target"] = np.ceil(df["p75_qty"]).astype(int)
    df["prep_target_p90"] = np.ceil(df["p90_qty"]).astype(int)

    # --- CV ---
    # std / mean. replace(0) guards against div-by-zero on zero-mean rows.
    df["cv"] = df["std_qty"] / df["mean_qty"].replace(0, np.nan)

    # --- Confidence scoring ---
    # High:   ≥ 40 weeks of data + low variability (CV < 1.0)
    # Medium: ≥ 20 weeks (but not High)
    # Low:    < 20 weeks — thin data, treat as rough estimates
    high   = (df["weeks_observed"] >= 40) & (df["cv"] < 1.0)
    medium = (df["weeks_observed"] >= 20) & ~high
    df["confidence"] = np.select([high, medium], ["High", "Medium"], default="Low")

    # --- Revenue impact estimates ---
    df["revenue_at_target"] = df["prep_target"] * df["price"]
    df["revenue_at_p90"]    = df["prep_target_p90"] * df["price"]
    df["revenue_upside"]    = df["revenue_at_p90"] - df["revenue_at_target"]

    # --- Action flags ---
    # NOTE: originally had mean_qty < 0.5 for very_low — never fired because
    # conditional mean excludes zero-sales days (min is always 1.0). Caught this
    # when 100% of rows came back "Maintain". Switched to weeks_observed as the
    # thin-data / low-volume signal instead.
    #
    # Similarly, volatile was cv > 1.5 but max CV in this dataset is ~0.80
    # (conditional demand is naturally smoother). Adjusted to 0.5 — still conservative
    # but actually meaningful.
    very_low = df["weeks_observed"] < 8    # sold in fewer than 8 of the possible slots
    volatile = df["cv"] > 0.5             # std > 50% of mean — notable variability

    df["action"] = np.select(
        [very_low, volatile],
        ["Review - Very Low Volume", "Volatile - Monitor Closely"],
        default="Maintain",
    )

    return df


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Read the SKU-level demand stats produced by aggregate_demand.py.
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
