import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

REPORT_COLS = {
    "name": "Pizza Name",
    "category": "Category",
    "size": "Size",
    "day_of_week": "Day",
    "time_window": "Time Window",
    "prep_target": "Recommended Prep Qty",
    "mean_qty": "Avg Sold (Historical)",
    "std_qty": "Std Dev",
    "p90_qty": "High-Demand Day Qty (90th Pct)",
    "weeks_observed": "Weeks with Sales",
    "confidence": "Confidence",
    "revenue_at_target": "Revenue at Prep Target ($)",
    "revenue_upside": "Revenue Upside if Prepped to P90 ($)",
    "action": "Action Flag",
}


def write_csv(recs):
    report = recs[list(REPORT_COLS.keys())].rename(columns=REPORT_COLS).copy()
    report["Avg Sold (Historical)"] = report["Avg Sold (Historical)"].round(1)
    report["Std Dev"] = report["Std Dev"].round(1)
    report["High-Demand Day Qty (90th Pct)"] = report["High-Demand Day Qty (90th Pct)"].round(1)
    report["Revenue at Prep Target ($)"] = report["Revenue at Prep Target ($)"].round(2)
    report["Revenue Upside if Prepped to P90 ($)"] = report["Revenue Upside if Prepped to P90 ($)"].round(2)

    out = REPORTS_DIR / "inventory_recommendations.csv"
    report.to_csv(out, index=False)
    print(f"  inventory_recommendations.csv — {len(report):,} rows")
    return report


def write_markdown(fact, recs, cat_stats):
    total_orders = fact["order_id"].nunique()
    total_revenue = fact["revenue"].sum()
    top_cats = (
        fact.groupby("category")["revenue"].sum()
        .sort_values(ascending=False)
        .head(3)
    )
    top_cats_pct = (top_cats / total_revenue * 100).round(1)

    peak_window = (
        fact.groupby(["day_of_week", "time_window"])["quantity"]
        .sum()
        .idxmax()
    )
    peak_day, peak_tw = peak_window

    top10 = (
        recs.groupby(["pizza_type_id", "name"])["mean_qty"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    conf_dist = recs["confidence"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
    total_upside = recs["revenue_upside"].sum()

    cat_peak = (
        cat_stats.groupby("category")
        .apply(lambda g: g.loc[g["mean_qty"].idxmax(), ["day_of_week", "time_window", "mean_qty"]])
        .reset_index()
    )
    cat_slow = (
        cat_stats.groupby("category")
        .apply(lambda g: g.loc[g["mean_qty"].idxmin(), ["day_of_week", "time_window", "mean_qty"]])
        .reset_index()
    )

    # Concrete example action for the operational problem definition
    top_sku_friday = (
        recs[
            (recs["day_of_week"] == "Friday") & (recs["time_window"] == "Dinner")
        ]
        .sort_values("prep_target", ascending=False)
        .iloc[0]
    )
    example_action = (
        f"Prep {top_sku_friday['prep_target']} units of "
        f"{top_sku_friday['name']} ({top_sku_friday['size']}) before Friday Dinner service."
    )

    lines = [
        "# Pizza Inventory Recommendations — 2015 Analysis",
        "",
        "## Operational Problem Definition",
        "",
        "**Stakeholder:** Kitchen manager / ops lead",
        "",
        "**Decision supported:** How many of each pizza to prep at the start of each shift "
        "(Lunch and Dinner) to avoid both stockouts and waste, given a same-day restocking "
        "option with a 4-hour lead time.",
        "",
        "**Day-to-day use:** At the start of each shift, the manager opens "
        "`inventory_recommendations.csv`, filters to today's day of week and the upcoming "
        "time window, and reads the *Recommended Prep Qty* for each pizza. If Lunch runs low "
        "before 15:00, a restock order is triggered to cover Dinner.",
        "",
        f"**Example action:** {example_action}",
        "",
        "## Assumptions",
        "",
        "- **4-hour same-day restocking:** The operation can receive replenished stock within "
        "4 hours of placing an order, making mid-day restocking feasible between Lunch and Dinner.",
        "- **2015 as representative baseline:** Demand patterns from 2015 are used as a proxy "
        "for future demand. No year-over-year trend adjustment is applied.",
        "- **Conditional mean:** Prep targets are based only on days when a pizza was sold "
        "(zero-sales days are excluded). The *Weeks with Sales* column shows how many weeks "
        "each figure is based on.",
        "- **75th percentile as prep target:** Prepping to the 75th percentile covers demand "
        "on approximately 3 of 4 historical days. Same-day restocking handles the remaining tail.",
        "- **No ingredient cost data:** Recommendations optimize for demand coverage, not margin. "
        "Waste reduction through cost weighting is a future improvement.",
        "",
        "## Executive Summary",
        "",
        f"- **Total orders in 2015:** {total_orders:,}",
        f"- **Total revenue:** ${total_revenue:,.2f}",
        "- **Top 3 categories by revenue:**",
    ]
    for cat, pct in top_cats_pct.items():
        lines.append(f"  - {cat}: {pct}% (${top_cats[cat]:,.2f})")

    lines += [
        "",
        "## Peak Demand Windows",
        "",
        f"Highest order volume occurs on **{peak_day} {peak_tw}**.",
        "Use this window as the baseline for maximum inventory prep.",
        "",
        "## Top 10 Pizzas by Average Weekly Volume",
        "",
        "| Pizza | Avg Units/Day-Window |",
        "|-------|----------------------|",
    ]
    for _, row in top10.iterrows():
        lines.append(f"| {row['name']} | {row['mean_qty']:.1f} |")

    lines += [
        "",
        "## Category Prep Guidance",
        "",
        "Ratio of peak-day demand to slowest-day demand per category.",
        "",
        "| Category | Peak Day | Slowest Day | Peak-to-Slow Ratio |",
        "|----------|----------|-------------|---------------------|",
    ]
    for cat in cat_peak["category"]:
        peak_row = cat_peak[cat_peak["category"] == cat].iloc[0]
        slow_row = cat_slow[cat_slow["category"] == cat].iloc[0]
        if slow_row["mean_qty"] > 0:
            ratio = peak_row["mean_qty"] / slow_row["mean_qty"]
        else:
            ratio = float("inf")
        lines.append(
            f"| {cat} | {peak_row['day_of_week']} {peak_row['time_window']} "
            f"| {slow_row['day_of_week']} {slow_row['time_window']} | {ratio:.1f}x |"
        )

    lines += [
        "",
        "## Confidence Distribution",
        "",
        "Confidence reflects how consistently a pizza sells on a given day and time window.",
        "",
        f"- **High confidence:** {conf_dist['High']:,} SKU-day-window combinations",
        f"- **Medium confidence:** {conf_dist['Medium']:,} SKU-day-window combinations",
        f"- **Low confidence:** {conf_dist['Low']:,} SKU-day-window combinations",
        "",
        "Low-confidence rows are based on fewer than 20 weeks of sales data — treat those "
        "prep targets as estimates and monitor closely.",
        "",
        "## Revenue Opportunity",
        "",
        f"Prepping all SKUs to the 90th-percentile quantity on peak days would generate an "
        f"additional **${total_upside:,.2f}** in revenue across the year, compared to prepping "
        "to the 75th percentile (the current recommended target).",
        "",
        "## Restocking Note",
        "",
        "Same-day restocking is available with a 4-hour lead time. The recommended workflow:",
        "",
        "1. **Morning prep (open):** Prep to the Lunch recommended quantity.",
        "2. **Mid-day decision point (around 14:00):** If Lunch inventory ran low, trigger a "
        "restock order before 15:00 to receive stock before the Dinner window opens at 17:00.",
        "3. **Dinner prep:** Prep to the Dinner recommended quantity on receipt of restocked goods.",
        "",
        "The 'Other' time window (orders outside 11:00-14:59 and 17:00-21:59) accounts for a "
        "small share of daily volume. No separate prep run is needed — carry-over from Dinner prep "
        "is sufficient.",
        "",
        "> **Note on XL/XXL sizes:** Only *The Greek Pizza* is offered in XL and XXL. "
        "Size-mix analysis for those sizes is based on a single SKU and should be read accordingly.",
        "",
        "## MVP Approach & Justification",
        "",
        "**Method:** Rule-based prep recommendation using the 75th percentile of historical "
        "daily demand per SKU, day of week, and time window.",
        "",
        "**Why this is sufficient for an MVP:**",
        "",
        "- Interpretable — any manager can read a percentile-based target without statistical training.",
        "- Requires only one year of transaction data, no external inputs.",
        "- Outperforms a naive average by absorbing demand spikes through the same-day restock "
        "trigger, while avoiding over-prep on typical days.",
        "- Confidence scoring (High/Medium/Low) tells the operator exactly which recommendations "
        "to trust and which to treat as estimates.",
        "",
        "A demand forecast model (e.g., time-series regression) would improve precision but adds "
        "complexity without a meaningful operational advantage while same-day restocking is available.",
        "",
        "## Analysis Limitations",
        "",
        "- **Single year of data:** Seasonal trends, year-over-year growth, and one-off events "
        "(holidays, local events) are not captured.",
        "- **No zero-sales imputation:** Days when a pizza was not ordered at all are excluded "
        "from the average. For low-volume items, this inflates the apparent mean.",
        "- **No ingredient cost data:** Cannot identify which stockouts cost the most margin or "
        "which over-preps generate the most waste by dollar value.",
        "- **No weather or promotional data:** External demand drivers are invisible to the model.",
        "- **Static thresholds:** The 75th-percentile rule and confidence cutoffs (20/40 weeks) "
        "are fixed. They have not been calibrated against actual stockout or waste records.",
        "",
        "## Future Improvements",
        "",
        "- **Incorporate ingredient costs:** Weight prep recommendations by margin to minimize "
        "high-cost waste and prioritize high-margin stockout prevention.",
        "- **Rolling 8-week forecast:** Replace the full-year average with a recent-window "
        "model to capture seasonality and drift.",
        "- **Weather and events data:** Layer in external signals (rain, local sports games, "
        "holidays) that correlate with demand spikes.",
        "- **Actual vs. recommended tracking:** Log real prep quantities and sales to calibrate "
        "confidence thresholds and detect when the 2015 baseline becomes stale.",
        "- **Promotion-aware adjustments:** Flag days with active promotions so the manager "
        "can apply a manual multiplier to the base recommendation.",
    ]

    out = REPORTS_DIR / "summary_report.md"
    out.write_text("\n".join(lines))
    print(f"  summary_report.md written")


def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    fact = pd.read_csv(OUTPUT_DIR / "fact_table.csv", parse_dates=["date"])
    recs = pd.read_csv(OUTPUT_DIR / "recommendations_raw.csv")
    cat_stats = pd.read_csv(OUTPUT_DIR / "demand_stats_category.csv")

    print("Generating reports...")
    write_csv(recs)
    write_markdown(fact, recs, cat_stats)
    print("Done.")


if __name__ == "__main__":
    main()
