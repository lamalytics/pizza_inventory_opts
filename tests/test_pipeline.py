"""
QA suite for the pizza inventory pipeline.

Run after the pipeline has been executed:
    python3 run_pipeline.py
    python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"


# ---------------------------------------------------------------------------
# Fixtures — load each artifact once per test session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fact():
    return pd.read_csv(OUTPUT / "fact_table.csv", parse_dates=["date"])


@pytest.fixture(scope="session")
def sku_stats():
    return pd.read_csv(OUTPUT / "demand_stats_sku.csv")


@pytest.fixture(scope="session")
def cat_stats():
    return pd.read_csv(OUTPUT / "demand_stats_category.csv")


@pytest.fixture(scope="session")
def recs_raw():
    return pd.read_csv(OUTPUT / "recommendations_raw.csv")


@pytest.fixture(scope="session")
def recs_report():
    return pd.read_csv(REPORTS / "inventory_recommendations.csv")


# ---------------------------------------------------------------------------
# Step 2 — fact table
# ---------------------------------------------------------------------------


class TestFactTable:
    def test_row_count(self, fact):
        assert len(fact) == 48620, f"Expected 48620 rows, got {len(fact)}"

    def test_revenue_in_range(self, fact):
        total = fact["revenue"].sum()
        assert 750_000 < total < 900_000, f"Revenue ${total:,.2f} outside expected range"

    def test_no_null_revenue(self, fact):
        assert fact["revenue"].isna().sum() == 0

    def test_time_window_values(self, fact):
        allowed = {"Lunch", "Dinner", "Other"}
        assert set(fact["time_window"].unique()).issubset(allowed)

    def test_other_window_is_minority(self, fact):
        other_pct = (fact["time_window"] == "Other").mean()
        assert other_pct < 0.20, f"'Other' time window is {other_pct:.1%} — unexpectedly large"

    def test_day_of_week_all_seven(self, fact):
        days = set(fact["day_of_week"].unique())
        assert len(days) == 7, f"Missing days of week: {days}"

    def test_quantity_positive(self, fact):
        assert fact["quantity"].min() >= 1

    def test_date_range_2015(self, fact):
        assert fact["date"].dt.year.unique().tolist() == [2015]


# ---------------------------------------------------------------------------
# Step 3 — demand stats
# ---------------------------------------------------------------------------


class TestDemandStats:
    def test_sku_stats_has_rows(self, sku_stats):
        assert len(sku_stats) > 0

    def test_cat_stats_four_categories(self, cat_stats):
        cats = set(cat_stats["category"].unique())
        assert cats == {"Chicken", "Classic", "Supreme", "Veggie"}

    def test_mean_qty_positive(self, sku_stats):
        assert (sku_stats["mean_qty"] > 0).all()

    def test_p75_gte_p25(self, sku_stats):
        assert (sku_stats["p75_qty"] >= sku_stats["p25_qty"]).all()

    def test_p90_gte_p75(self, sku_stats):
        assert (sku_stats["p90_qty"] >= sku_stats["p75_qty"]).all()

    def test_weeks_observed_nonzero(self, sku_stats):
        assert (sku_stats["weeks_observed"] > 0).all()

    def test_spot_check_popular_pizza_friday_dinner(self, sku_stats):
        """A high-volume pizza on Friday Dinner should have meaningful average demand."""
        friday_dinner = sku_stats[
            (sku_stats["day_of_week"] == "Friday") & (sku_stats["time_window"] == "Dinner")
        ]
        assert len(friday_dinner) > 0
        max_mean = friday_dinner["mean_qty"].max()
        assert max_mean >= 2, f"Top Friday Dinner mean is only {max_mean:.1f} — suspiciously low"


# ---------------------------------------------------------------------------
# Step 4 — recommendations
# ---------------------------------------------------------------------------


class TestRecommendations:
    def test_prep_target_at_least_one(self, recs_raw):
        assert (recs_raw["prep_target"] >= 1).all()

    def test_all_three_confidence_levels_present(self, recs_raw):
        conf = set(recs_raw["confidence"].unique())
        assert conf == {"High", "Medium", "Low"}, f"Confidence levels: {conf}"

    def test_revenue_upside_nonnegative(self, recs_raw):
        assert (recs_raw["revenue_upside"] >= 0).all()

    def test_revenue_at_p90_gte_target(self, recs_raw):
        assert (recs_raw["revenue_at_p90"] >= recs_raw["revenue_at_target"]).all()

    def test_action_values_valid(self, recs_raw):
        allowed = {"Maintain", "Review - Very Low Volume", "Volatile - Monitor Closely"}
        assert set(recs_raw["action"].unique()).issubset(allowed)

    def test_cv_defined_for_nonzero_mean(self, recs_raw):
        nonzero_mean = recs_raw[recs_raw["mean_qty"] > 0]
        assert nonzero_mean["cv"].notna().all()

    def test_high_confidence_meets_criteria(self, recs_raw):
        high = recs_raw[recs_raw["confidence"] == "High"]
        assert (high["weeks_observed"] >= 40).all()
        assert (high["cv"] < 1.0).all()


# ---------------------------------------------------------------------------
# Step 5 — report CSV
# ---------------------------------------------------------------------------


class TestReportCSV:
    def test_file_exists(self):
        assert (REPORTS / "inventory_recommendations.csv").exists()

    def test_expected_columns_present(self, recs_report):
        expected = {
            "Pizza Name", "Category", "Size", "Day", "Time Window",
            "Recommended Prep Qty", "Avg Sold (Historical)", "Confidence",
            "Action Flag",
        }
        missing = expected - set(recs_report.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_prep_qty_is_integer_like(self, recs_report):
        col = recs_report["Recommended Prep Qty"]
        assert (col == col.round(0)).all(), "Prep quantities should be whole numbers"

    def test_friday_dinner_high_demand_prep_meaningful(self, recs_report, recs_raw):
        """High-demand Friday Dinner SKUs should have a prep target of at least 2.

        Right-skewed demand means p75 < mean is expected and correct — spike days
        pull the mean up, but the 75th-percentile target is the right anchor.
        This test checks the meaningful invariant: SKUs with average demand >= 3
        should get a prep target of at least 2.
        """
        fd = recs_raw[
            (recs_raw["day_of_week"] == "Friday") & (recs_raw["time_window"] == "Dinner")
        ]
        high_demand = fd[fd["mean_qty"] >= 3]
        assert (high_demand["prep_target"] >= 2).all(), (
            "Some high-demand Friday Dinner SKUs have prep_target < 2"
        )

    def test_summary_report_exists(self):
        assert (REPORTS / "summary_report.md").exists()

    def test_summary_report_has_sections(self):
        content = (REPORTS / "summary_report.md").read_text()
        for section in [
            "Operational Problem Definition",
            "Assumptions",
            "Executive Summary",
            "Peak Demand Windows",
            "Top 10 Pizzas",
            "Category Prep Guidance",
            "Confidence Distribution",
            "Revenue Opportunity",
            "Restocking Note",
            "MVP Approach",
            "Analysis Limitations",
            "Future Improvements",
        ]:
            assert section in content, f"Missing section in summary_report.md: {section}"

    def test_summary_report_operational_framing(self):
        content = (REPORTS / "summary_report.md").read_text()
        assert "Stakeholder" in content
        assert "Example action" in content
        assert "Day-to-day use" in content

    def test_summary_report_assumptions_present(self):
        content = (REPORTS / "summary_report.md").read_text()
        assert "75th percentile" in content
        assert "4-hour" in content
        assert "Conditional mean" in content

    def test_summary_report_limitations_present(self):
        content = (REPORTS / "summary_report.md").read_text()
        assert "Limitation" in content or "limitation" in content

    def test_summary_report_future_improvements_present(self):
        content = (REPORTS / "summary_report.md").read_text()
        assert "Future Improvements" in content
