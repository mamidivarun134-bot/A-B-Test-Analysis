"""
significance_test.py
---------------------
Hypothesis-testing utilities for analysing A/B test results:

- Two-proportion z-test (e.g. conversion rate control vs. treatment)
- Chi-square test of independence (categorical outcome vs. group)
- Welch's t-test (e.g. revenue per user, unequal variances assumed)
- Mann-Whitney U test (non-parametric alternative for skewed metrics)
- Confidence interval for the difference in two proportions
- A convenience summary function for quick group-level metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


@dataclass
class TestResult:
    """Container for a hypothesis test result."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    significant: bool
    details: dict

    def __repr__(self) -> str:
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.test_name}: statistic={self.statistic:.4f}, "
            f"p-value={self.p_value:.4g} ({verdict} at alpha={self.alpha})"
        )


def two_proportion_z_test(
    conversions: "tuple[int, int]",
    n_obs: "tuple[int, int]",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Two-proportion z-test, e.g. comparing conversion rates of control
    vs. treatment groups.

    Parameters
    ----------
    conversions : (n_control_conversions, n_treatment_conversions)
    n_obs : (n_control_total, n_treatment_total)
    alpha : significance level
    alternative : 'two-sided', 'larger', or 'smaller'
        ('larger' tests whether group 2's rate is larger than group 1's)

    Returns
    -------
    TestResult
    """
    count = np.array(conversions)
    nobs = np.array(n_obs)

    # statsmodels' proportions_ztest alternative convention refers to
    # prop[0] vs prop[1]; we keep 'two-sided' as default for clarity.
    stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative=alternative)

    rate_control = conversions[0] / n_obs[0]
    rate_treatment = conversions[1] / n_obs[1]

    return TestResult(
        test_name="Two-proportion z-test",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=p_value < alpha,
        details={
            "rate_control": rate_control,
            "rate_treatment": rate_treatment,
            "absolute_lift": rate_treatment - rate_control,
            "relative_lift_pct": (
                (rate_treatment - rate_control) / rate_control * 100
                if rate_control > 0
                else float("nan")
            ),
        },
    )


def chi_square_test(
    contingency_table: pd.DataFrame, alpha: float = 0.05
) -> TestResult:
    """
    Chi-square test of independence between group membership and a
    categorical outcome (e.g. converted vs. not converted).

    Parameters
    ----------
    contingency_table : pd.DataFrame
        Rows = groups, columns = outcome categories (raw counts).
    alpha : significance level

    Returns
    -------
    TestResult
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    return TestResult(
        test_name="Chi-square test of independence",
        statistic=float(chi2),
        p_value=float(p_value),
        alpha=alpha,
        significant=p_value < alpha,
        details={
            "degrees_of_freedom": dof,
            "expected_frequencies": expected.tolist(),
        },
    )


def welch_t_test(
    sample_a: "np.ndarray | pd.Series",
    sample_b: "np.ndarray | pd.Series",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Welch's t-test (does not assume equal variances) — useful for comparing
    a continuous metric such as revenue-per-user between two groups.

    Parameters
    ----------
    sample_a, sample_b : array-like
        Raw observations for each group (e.g. revenue per user).
    alpha : significance level
    alternative : 'two-sided', 'less', or 'greater'

    Returns
    -------
    TestResult
    """
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)

    stat, p_value = stats.ttest_ind(
        sample_a, sample_b, equal_var=False, alternative=alternative
    )

    return TestResult(
        test_name="Welch's t-test",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=p_value < alpha,
        details={
            "mean_a": float(sample_a.mean()),
            "mean_b": float(sample_b.mean()),
            "std_a": float(sample_a.std(ddof=1)),
            "std_b": float(sample_b.std(ddof=1)),
            "n_a": int(sample_a.size),
            "n_b": int(sample_b.size),
        },
    )


def mann_whitney_u_test(
    sample_a: "np.ndarray | pd.Series",
    sample_b: "np.ndarray | pd.Series",
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Mann-Whitney U test: a non-parametric alternative to the t-test,
    useful when the metric of interest (e.g. revenue) is highly skewed
    or zero-inflated, so normality assumptions are questionable.

    Parameters
    ----------
    sample_a, sample_b : array-like
    alpha : significance level
    alternative : 'two-sided', 'less', or 'greater'

    Returns
    -------
    TestResult
    """
    sample_a = np.asarray(sample_a, dtype=float)
    sample_b = np.asarray(sample_b, dtype=float)

    stat, p_value = stats.mannwhitneyu(sample_a, sample_b, alternative=alternative)

    return TestResult(
        test_name="Mann-Whitney U test",
        statistic=float(stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=p_value < alpha,
        details={
            "median_a": float(np.median(sample_a)),
            "median_b": float(np.median(sample_b)),
            "n_a": int(sample_a.size),
            "n_b": int(sample_b.size),
        },
    )


def confidence_interval_proportion_diff(
    conversions: "tuple[int, int]",
    n_obs: "tuple[int, int]",
    confidence: float = 0.95,
) -> "tuple[float, float]":
    """
    Compute a confidence interval for the difference in two proportions
    (treatment - control) using the normal approximation.

    Parameters
    ----------
    conversions : (n_control_conversions, n_treatment_conversions)
    n_obs : (n_control_total, n_treatment_total)
    confidence : confidence level (e.g. 0.95 for 95%)

    Returns
    -------
    (lower_bound, upper_bound) : tuple of float
    """
    p1 = conversions[0] / n_obs[0]
    p2 = conversions[1] / n_obs[1]
    diff = p2 - p1

    se = math_sqrt = np.sqrt(
        p1 * (1 - p1) / n_obs[0] + p2 * (1 - p2) / n_obs[1]
    )
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    return (diff - z * se, diff + z * se)


def summarize_group_metrics(
    df: pd.DataFrame,
    group_col: str = "group",
    conversion_col: str = "converted",
    revenue_col: Optional[str] = "revenue",
) -> pd.DataFrame:
    """
    Convenience function to compute a quick per-group summary table:
    sample size, conversion count/rate, and (optionally) average revenue
    metrics.

    Parameters
    ----------
    df : pd.DataFrame
        The raw experiment data.
    group_col : str
        Column identifying the experiment arm (e.g. 'control'/'treatment').
    conversion_col : str
        Binary column indicating conversion (1) or not (0).
    revenue_col : str, optional
        Column with revenue values (0 for non-converters). If None,
        revenue metrics are skipped.

    Returns
    -------
    pd.DataFrame
        Summary table indexed by group.
    """
    grouped = df.groupby(group_col)
    summary = grouped[conversion_col].agg(n="count", conversions="sum")
    summary["conversion_rate"] = summary["conversions"] / summary["n"]

    if revenue_col is not None and revenue_col in df.columns:
        summary["avg_revenue_per_user"] = grouped[revenue_col].mean()
        converters = df[df[conversion_col] == 1]
        summary["avg_revenue_per_converter"] = converters.groupby(group_col)[
            revenue_col
        ].mean()

    return summary
