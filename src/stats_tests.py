"""
Hypothesis tests and confidence intervals.

Every function returns an interval, not just a p value. A point estimate with no
interval tells a stakeholder nothing about whether to act, and a p value on its
own tells them nothing about how big the effect is. The interval is the finding.
"""

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep


def two_proportion_test(successes_control, n_control, successes_treatment, n_treatment, alpha=0.05):
    """
    The workhorse for conversion style metrics.

    Takes counts, not a dataframe. That is deliberate: it means this runs on
    fourteen million rows just as fast as on four, because the sufficient
    statistics are four integers.
    """
    count = np.array([successes_treatment, successes_control])
    nobs = np.array([n_treatment, n_control])

    z_stat, p_value = proportions_ztest(count=count, nobs=nobs, alternative="two-sided")

    p_control = successes_control / n_control
    p_treatment = successes_treatment / n_treatment
    absolute_lift = p_treatment - p_control
    relative_lift = absolute_lift / p_control if p_control > 0 else np.nan

    ci_low, ci_high = confint_proportions_2indep(
        count1=successes_treatment,
        nobs1=n_treatment,
        count2=successes_control,
        nobs2=n_control,
        method="wald",
        compare="diff",
        alpha=alpha,
    )

    rel_ci_low = ci_low / p_control if p_control > 0 else np.nan
    rel_ci_high = ci_high / p_control if p_control > 0 else np.nan

    return {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "successes_control": int(successes_control),
        "successes_treatment": int(successes_treatment),
        "rate_control": float(p_control),
        "rate_treatment": float(p_treatment),
        "absolute_lift": float(absolute_lift),
        "relative_lift": float(relative_lift),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "ci_absolute": (float(ci_low), float(ci_high)),
        "ci_relative": (float(rel_ci_low), float(rel_ci_high)),
        "alpha": alpha,
    }


def welch_t_test(values_control, values_treatment, alpha=0.05):
    """
    For continuous metrics such as revenue per user.
    Welch rather than Student, because equal variance between arms is an
    assumption we have no reason to make and no way to justify.
    """
    t_stat, p_value = stats.ttest_ind(values_treatment, values_control, equal_var=False)

    mean_c = float(np.mean(values_control))
    mean_t = float(np.mean(values_treatment))
    diff = mean_t - mean_c

    var_t = np.var(values_treatment, ddof=1) / len(values_treatment)
    var_c = np.var(values_control, ddof=1) / len(values_control)
    se = np.sqrt(var_t + var_c)

    dof = (var_t + var_c) ** 2 / (
        var_t ** 2 / (len(values_treatment) - 1) + var_c ** 2 / (len(values_control) - 1)
    )
    crit = stats.t.ppf(1 - alpha / 2, dof)

    return {
        "mean_control": mean_c,
        "mean_treatment": mean_t,
        "absolute_lift": float(diff),
        "relative_lift": float(diff / mean_c) if mean_c != 0 else np.nan,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "ci_absolute": (float(diff - crit * se), float(diff + crit * se)),
        "alpha": alpha,
    }


def bootstrap_ci(values_control, values_treatment, statistic=np.mean, n_boot=2000, alpha=0.05, seed=42):
    """
    Distribution free interval. Reach for this when the metric is heavily skewed,
    which revenue always is, and the normal approximation is doing too much work.
    """
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        c = rng.choice(values_control, size=len(values_control), replace=True)
        t = rng.choice(values_treatment, size=len(values_treatment), replace=True)
        diffs[i] = statistic(t) - statistic(c)

    return {
        "point_estimate": float(np.mean(diffs)),
        "ci": (float(np.percentile(diffs, 100 * alpha / 2)),
               float(np.percentile(diffs, 100 * (1 - alpha / 2)))),
        "n_boot": n_boot,
    }