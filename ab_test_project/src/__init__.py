"""
ab_test_project.src

Utility package for analysing A/B test (marketing experiment) results.

Modules:
    power_analysis      -- sample size / statistical power calculations
    significance_test    -- hypothesis testing utilities (proportions & means)
"""

from .power_analysis import (
    required_sample_size_proportions,
    achieved_power_proportions,
    minimum_detectable_effect,
)
from .significance_test import (
    two_proportion_z_test,
    chi_square_test,
    welch_t_test,
    mann_whitney_u_test,
    confidence_interval_proportion_diff,
    summarize_group_metrics,
)

__all__ = [
    "required_sample_size_proportions",
    "achieved_power_proportions",
    "minimum_detectable_effect",
    "two_proportion_z_test",
    "chi_square_test",
    "welch_t_test",
    "mann_whitney_u_test",
    "confidence_interval_proportion_diff",
    "summarize_group_metrics",
]
