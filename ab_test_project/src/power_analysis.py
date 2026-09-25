"""
power_analysis.py
------------------
Statistical power and sample-size utilities for a two-sample
(A/B) proportions test, e.g. conversion-rate experiments.

All functions rely on `statsmodels.stats.power.NormalIndPower`, which
implements the standard normal-approximation power calculations used
for two-proportion z-tests.
"""
from __future__ import annotations

import math
from typing import Optional

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def _effect_size(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions."""
    return proportion_effectsize(p1, p2)


def required_sample_size_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
    alternative: str = "two-sided",
) -> int:
    """
    Compute the required per-group sample size to detect a difference
    between a baseline conversion rate and a target rate.

    Parameters
    ----------
    baseline_rate : float
        Expected conversion rate of the control group (0-1).
    minimum_detectable_effect : float
        The absolute difference in conversion rate you want to be able
        to detect (e.g. 0.02 for a 2 percentage-point lift).
    alpha : float
        Significance level (Type I error rate). Default 0.05.
    power : float
        Desired statistical power (1 - Type II error rate). Default 0.8.
    ratio : float
        Ratio of treatment group size to control group size. Default 1.0
        (equal-sized groups).
    alternative : str
        'two-sided', 'larger', or 'smaller'.

    Returns
    -------
    int
        Required sample size per group (rounded up).
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1 (exclusive).")
    if minimum_detectable_effect <= 0:
        raise ValueError("minimum_detectable_effect must be positive.")

    target_rate = baseline_rate + minimum_detectable_effect
    if not (0 < target_rate < 1):
        raise ValueError(
            "baseline_rate + minimum_detectable_effect must remain between 0 and 1."
        )

    effect_size = _effect_size(target_rate, baseline_rate)
    analysis = NormalIndPower()
    n_control = analysis.solve_power(
        effect_size=abs(effect_size),
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    return math.ceil(n_control)


def achieved_power_proportions(
    baseline_rate: float,
    observed_rate: float,
    n_control: int,
    n_treatment: Optional[int] = None,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> float:
    """
    Given actual (or planned) group sizes, compute the statistical power
    achieved for detecting the difference between baseline_rate and
    observed_rate.

    Parameters
    ----------
    baseline_rate : float
        Control group conversion rate.
    observed_rate : float
        Treatment group conversion rate.
    n_control : int
        Sample size of the control group.
    n_treatment : int, optional
        Sample size of the treatment group. Defaults to n_control
        (equal-sized groups) if not provided.
    alpha : float
        Significance level.
    alternative : str
        'two-sided', 'larger', or 'smaller'.

    Returns
    -------
    float
        Statistical power (probability of detecting a true effect), in [0, 1].
    """
    if n_treatment is None:
        n_treatment = n_control

    effect_size = _effect_size(observed_rate, baseline_rate)
    ratio = n_treatment / n_control
    analysis = NormalIndPower()
    power = analysis.power(
        effect_size=abs(effect_size),
        nobs1=n_control,
        alpha=alpha,
        ratio=ratio,
        alternative=alternative,
    )
    return float(power)


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "two-sided",
    tolerance: float = 1e-4,
) -> float:
    """
    Estimate the smallest absolute lift in conversion rate that could be
    reliably detected (at the given alpha/power) with a fixed sample size
    per group. Uses binary search over the effect size.

    Parameters
    ----------
    baseline_rate : float
        Control group conversion rate.
    n_per_group : int
        Sample size available in each group.
    alpha : float
        Significance level.
    power : float
        Desired statistical power.
    alternative : str
        'two-sided', 'larger', or 'smaller'.
    tolerance : float
        Precision of the binary search on absolute lift.

    Returns
    -------
    float
        Minimum detectable absolute difference in conversion rate.
    """
    if not (0 < baseline_rate < 1):
        raise ValueError("baseline_rate must be between 0 and 1 (exclusive).")

    analysis = NormalIndPower()
    lo, hi = 0.0, 1.0 - baseline_rate - 1e-6

    def achieved(lift: float) -> float:
        target = baseline_rate + lift
        es = abs(_effect_size(target, baseline_rate))
        return analysis.power(
            effect_size=es, nobs1=n_per_group, alpha=alpha, ratio=1.0, alternative=alternative
        )

    # Binary search for the lift that achieves the target power.
    for _ in range(100):
        mid = (lo + hi) / 2
        if achieved(mid) < power:
            lo = mid
        else:
            hi = mid
        if hi - lo < tolerance:
            break

    return round(hi, 4)
