import numpy as np    # np.ceil for rounding reorder quantities up to whole units
import pandas as pd   # reading CSVs and dataframe operations for computing report values
from pathlib import Path  # cross-platform file path construction

OUTPUT_DIR  = Path(__file__).parent.parent / "output"   # intermediate pipeline outputs
REPORTS_DIR = Path(__file__).parent.parent / "reports"  # final stakeholder deliverables

# Industry standard for food distributor lead time is 2-3 days (order-to-delivery).
# 3 days is used as the planning baseline; increase to 4 for a more conservative buffer.
# This constant flows into the reorder guide — change it here and re-run the pipeline.
LEAD_DAYS = 3

# Deliveries per week determines how to split the weekly demand into per-order quantities.
# Twice-weekly (e.g. Monday + Thursday) is standard for most restaurant distributors.
DELIVERIES_PER_WEEK = 2

# Maps internal column names (from recommendations_raw.csv) to plain-English labels
# for the stakeholder-facing CSV. Defined here so any rename is made in one place.
REPORT_COLS = {
    "name":               "Pizza Name",
    "category":           "Category",
    "size":               "Size",
    "day_of_week":        "Day",
    "time_window":        "Time Window",
    "prep_target":        "Recommended Prep Qty",
    "mean_qty":           "Avg Sold (Historical)",
    "std_qty":            "Std Dev",
    "p90_qty":            "High-Demand Day Qty (90th Pct)",
    "weeks_observed":     "Weeks with Sales",
    "confidence":         "Confidence",
    "revenue_at_target":  "Revenue at Prep Target ($)",
    "revenue_upside":     "Revenue Upside if Prepped to P90 ($)",
    "action":             "Action Flag",
}


def write_csv(recs):
    # Select only the columns listed in REPORT_COLS (drops internal-only columns like cv,
    # prep_target_p90, revenue_at_p90) then rename to plain-English headers.
    # .copy() prevents SettingWithCopyWarning when we mutate the renamed columns below.
    report = recs[list(REPORT_COLS.keys())].rename(columns=REPORT_COLS).copy()

    # Round each column to the precision appropriate for a manager reading a spreadsheet.
    # Averages and std devs to 1 decimal place — more precision implies false accuracy.
    report["Avg Sold (Historical)"]          = report["Avg Sold (Historical)"].round(1)
    report["Std Dev"]                        = report["Std Dev"].round(1)
    report["High-Demand Day Qty (90th Pct)"] = report["High-Demand Day Qty (90th Pct)"].round(1)
    # Dollar amounts to 2 decimal places (standard currency display).
    report["Revenue at Prep Target ($)"]               = report["Revenue at Prep Target ($)"].round(2)
    report["Revenue Upside if Prepped to P90 ($)"]     = report["Revenue Upside if Prepped to P90 ($)"].round(2)
    # Note: "Recommended Prep Qty" is already an integer from build_recommendations.py — no rounding needed.

    out = REPORTS_DIR / "inventory_recommendations.csv"
    report.to_csv(out, index=False)  # index=False omits the row number column
    print(f"  inventory_recommendations.csv — {len(report):,} rows")
    return report


def write_reorder_guide(recs):
    """
    Produces reports/reorder_guide.csv — a per-pizza ordering reference for a
    3-4 day supplier lead time.

    With lead times measured in days (not hours), the manager can no longer
    react same-day between shifts. Instead they need to know:
      1. Reorder point  — how many units on hand triggers a new order.
      2. Order quantity — how many units to order per delivery cycle.

    Both are derived from the per-shift prep_target already computed in
    build_recommendations.py, aggregated up to the pizza (not shift) level.
    """

    # Sum prep_target across all day × window rows for each pizza.
    # This gives the total number of units that should be prepped across a full week
    # (across every day and time window that pizza appears in the recommendations).
    weekly = (
        recs.groupby(["pizza_id", "name", "category", "size", "price"])
        .agg(
            weekly_prep_total=("prep_target", "sum"),   # total units across all shifts in a week
            avg_shift_qty=("mean_qty", "mean"),         # average units sold per individual shift
            confidence=("confidence", lambda x: x.mode()[0]),  # most common confidence level for this pizza
        )
        .reset_index()
    )

    # Daily prep units = weekly total spread evenly across 7 days.
    # Used as the basis for the reorder point calculation.
    weekly["daily_prep_units"] = (weekly["weekly_prep_total"] / 7).round(1)

    # Reorder point = how many units must remain in stock before placing a new order.
    # Formula: daily_prep_units × LEAD_DAYS
    # Rationale: if stock drops to this level today, you have just enough to cover
    # demand until the next delivery arrives in LEAD_DAYS days.
    # np.ceil ensures we round up — never leave a fractional unit as the safety threshold.
    weekly["reorder_point"] = np.ceil(weekly["daily_prep_units"] * LEAD_DAYS).astype(int)

    # Order quantity = how many units to request per delivery cycle.
    # Formula: weekly_prep_total / DELIVERIES_PER_WEEK
    # Rationale: if you receive DELIVERIES_PER_WEEK shipments per week, each order
    # needs to cover (7 / DELIVERIES_PER_WEEK) days of demand.
    # np.ceil rounds up so you never order less than a day's worth.
    weekly["order_qty_per_cycle"] = np.ceil(
        weekly["weekly_prep_total"] / DELIVERIES_PER_WEEK
    ).astype(int)

    # Rename columns to plain English for the stakeholder-facing file.
    guide = weekly.rename(columns={
        "name":               "Pizza Name",
        "category":           "Category",
        "size":               "Size",
        "price":              "Unit Price ($)",
        "weekly_prep_total":  "Total Units Needed Per Week",
        "daily_prep_units":   "Avg Units Per Day",
        "confidence":         "Confidence",
        "reorder_point":      f"Reorder Point (at {LEAD_DAYS}-day lead time)",
        "order_qty_per_cycle": f"Order Qty Per Cycle ({DELIVERIES_PER_WEEK}x/week)",
    }).drop(columns=["pizza_id", "avg_shift_qty"])

    # Sort by weekly volume descending so high-priority items appear at the top.
    guide = guide.sort_values("Total Units Needed Per Week", ascending=False).reset_index(drop=True)

    out = REPORTS_DIR / "reorder_guide.csv"
    guide.to_csv(out, index=False)
    print(f"  reorder_guide.csv — {len(guide):,} pizzas")
    return guide


def write_markdown(fact, recs, cat_stats):
    # --- Pre-compute all values used in the report ---

    # Count unique orders (not line items) for the executive summary.
    total_orders = fact["order_id"].nunique()

    # Sum revenue across all 48,620 line items.
    total_revenue = fact["revenue"].sum()

    # Group revenue by category, sort descending, keep the top 3.
    top_cats = (
        fact.groupby("category")["revenue"].sum()
        .sort_values(ascending=False)
        .head(3)
    )
    # Express each category's revenue as a percentage of total revenue, rounded to 1dp.
    top_cats_pct = (top_cats / total_revenue * 100).round(1)

    # Find the (day_of_week, time_window) combination with the highest total quantity sold.
    # .sum() aggregates across all 52 weeks; .idxmax() returns the index of the maximum,
    # which is a tuple (day_of_week, time_window) because of the two-level groupby.
    peak_window = (
        fact.groupby(["day_of_week", "time_window"])["quantity"]
        .sum()
        .idxmax()
    )
    # Unpack the tuple into two named variables for use in f-strings below.
    peak_day, peak_tw = peak_window

    # Top 10 pizzas by total mean demand across all day-window combinations.
    # Summing mean_qty across all (day, window) rows gives a rough total weekly
    # demand signal that can be used to rank pizzas by overall popularity.
    top10 = (
        recs.groupby(["pizza_type_id", "name"])["mean_qty"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    # Count how many SKU-day-window rows fall into each confidence level.
    # .reindex() ensures all three levels appear even if one has zero rows.
    conf_dist = recs["confidence"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)

    # Total revenue upside = sum of (P90 revenue - P75 revenue) across all rows.
    # This is the annual revenue left on the table by prepping conservatively.
    total_upside = recs["revenue_upside"].sum()

    # For each category, find the (day, window) row with the highest mean demand (peak)
    # and the row with the lowest mean demand (slowest).
    # .apply(lambda g: ...) runs per-group; g.loc[...idxmax()...] selects the peak row.
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

    # Build a concrete example action for the operational problem definition section.
    # We pick the highest prep_target SKU on Friday Dinner — the busiest shift for most categories.
    top_sku_friday = (
        recs[
            (recs["day_of_week"] == "Friday") & (recs["time_window"] == "Dinner")
        ]
        .sort_values("prep_target", ascending=False)
        .iloc[0]  # first row after sorting = the SKU with the highest prep target
    )
    example_action = (
        f"Prep {top_sku_friday['prep_target']} units of "
        f"{top_sku_friday['name']} ({top_sku_friday['size']}) before Friday Dinner service."
    )

    # --- Build the markdown document as a list of strings, then join with newlines ---
    # This approach makes it easy to append or insert sections without string concatenation.

    lines = [
        "# Pizza Inventory Recommendations — 2015 Analysis",
        "",
        "## Operational Problem Definition",
        "",
        "**Stakeholder:** Kitchen manager / ops lead",
        "",
        "**Decision supported:** How many of each pizza to prep at the start of each shift "
        "(Lunch and Dinner) to avoid stockouts, and when and how much to order from the "
        f"supplier given a {LEAD_DAYS}-day delivery lead time.",
        "",
        "**Day-to-day use:** At each shift, the manager checks `inventory_recommendations.csv` "
        "for today's *Recommended Prep Qty* per pizza. Separately, they check current stock "
        "levels against `reorder_guide.csv` — if any pizza's stock is at or below its "
        f"*Reorder Point*, a supplier order is placed today to arrive in {LEAD_DAYS} days.",
        "",
        f"**Example action:** {example_action}",
        "",
        "## Assumptions",
        "",
        f"- **{LEAD_DAYS}-day supplier lead time (industry standard):** Food distributors "
        "typically deliver 2-3 times per week with a 2-3 day order-to-delivery window. "
        f"This analysis uses {LEAD_DAYS} days as the baseline. Change LEAD_DAYS in "
        "generate_report.py and re-run the pipeline to adjust.",
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
    # Dynamically append each top category so the list updates automatically if data changes.
    for cat, pct in top_cats_pct.items():
        lines.append(f"  - {cat}: {pct}% (${top_cats[cat]:,.2f})")

    lines += [
        "",
        "## Peak Demand Windows",
        "",
        # peak_day and peak_tw were computed from idxmax() above.
        f"Highest order volume occurs on **{peak_day} {peak_tw}**.",
        "Use this window as the baseline for maximum inventory prep.",
        "",
        "## Top 10 Pizzas by Average Weekly Volume",
        "",
        "| Pizza | Avg Units/Day-Window |",
        "|-------|----------------------|",
    ]
    # Append one markdown table row per top-10 pizza.
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
        # Guard against division by zero for a category with zero slowest-day demand.
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
        # conf_dist was reindexed to guarantee all three keys exist.
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
        "## Restocking & Ordering Workflow",
        "",
        f"With a {LEAD_DAYS}-day supplier lead time, same-day restocking between shifts is not "
        "possible. The recommended workflow uses two separate cycles: a daily prep cycle "
        "and a twice-weekly ordering cycle.",
        "",
        "**Daily prep cycle (per shift):**",
        "",
        "1. **Lunch prep (before 11:00):** Pull from on-hand stock to reach the *Recommended "
        "Prep Qty* for each pizza (see `inventory_recommendations.csv`, filtered to today's "
        "day + Lunch).",
        "2. **Dinner prep (before 17:00):** Repeat using the Dinner targets. If stock for any "
        "pizza falls below its *Reorder Point* after Dinner prep, flag it for the next order.",
        "",
        f"**Twice-weekly ordering cycle ({DELIVERIES_PER_WEEK}x/week, e.g. Monday + Thursday):**",
        "",
        "1. Check current stock levels against the *Reorder Point* column in `reorder_guide.csv`.",
        "2. For any pizza at or below its reorder point, place an order for the *Order Qty "
        "Per Cycle* shown in the same file.",
        f"3. The order arrives in {LEAD_DAYS} days — enough time to cover demand until delivery "
        "without running out.",
        "",
        "> **Why twice weekly?** Ordering more frequently reduces the stock you need to hold "
        "on hand; ordering less frequently requires a larger safety buffer. Twice weekly is the "
        "industry norm for perishable ingredients and balances delivery cost against spoilage risk.",
        "",
        "The 'Other' time window (orders outside 11:00–14:59 and 17:00–21:59) is a small share "
        "of daily volume. Carry-over from Dinner prep covers it — no separate reorder target needed.",
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
    # "\n".join(lines) stitches the list back into a single string with one newline per item.
    out.write_text("\n".join(lines))
    print(f"  summary_report.md written")


def main():
    # Create the reports directory if it does not yet exist.
    REPORTS_DIR.mkdir(exist_ok=True)

    # Load the three intermediate files written by earlier pipeline steps.
    # parse_dates=["date"] restores the datetime type lost when fact_table.csv was written.
    fact      = pd.read_csv(OUTPUT_DIR / "fact_table.csv", parse_dates=["date"])
    recs      = pd.read_csv(OUTPUT_DIR / "recommendations_raw.csv")
    cat_stats = pd.read_csv(OUTPUT_DIR / "demand_stats_category.csv")

    print("Generating reports...")
    write_csv(recs)
    write_markdown(fact, recs, cat_stats)
    print("Done.")


if __name__ == "__main__":
    main()
