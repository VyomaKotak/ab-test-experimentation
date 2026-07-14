"""
Power analysis and sample size calculation for proportion based A/B tests.

Written out explicitly rather than as a single library call, because in an
interview you will be asked where the numbers come from and "statsmodels did it"
is not an answer.

The order matters. This module exists to be used before the data does. A sample
size chosen after seeing the result is not a sample size, it is a rationalisation.
"""

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size(baseline_rate, mde_relative, alpha=0.05, power=0.80, ratio=1.0):
    """
    Users needed per arm to detect a relative lift of mde_relative.

    baseline_rate : the rate we expect in control
    mde_relative  : the smallest relative lift worth acting on, e.g. 0.10
    ratio         : n_treatment / n_control. Criteo runs 85/15, so ratio is 5.67

    Returns a dict rather than a number, so every assumption travels with the answer.
    """
    treatment_rate = baseline_rate * (1 + mde_relative)
    effect_size = proportion_effectsize(treatment_rate, baseline_rate)

    analysis = NormalIndPower()
    n_control = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative="two-sided",
    )

    n_control = int(np.ceil(n_control))
    n_treatment = int(np.ceil(n_control * ratio))

    return {
        "baseline_rate": baseline_rate,
        "treatment_rate": treatment_rate,
        "absolute_mde": treatment_rate - baseline_rate,
        "relative_mde": mde_relative,
        "effect_size_cohens_h": float(effect_size),
        "alpha": alpha,
        "power": power,
        "ratio": ratio,
        "n_control": n_control,
        "n_treatment": n_treatment,
        "n_total": n_control + n_treatment,
    }


def achieved_power(n_control, n_treatment, baseline_rate, target_rate, alpha=0.05):
    """
    The power we actually had, given the sample we actually got.

    This is the question that matters when the result is null. "We found no
    effect" and "we could not have found the effect" look identical in a p value
    and mean completely different things.
    """
    effect_size = proportion_effectsize(target_rate, baseline_rate)
    analysis = NormalIndPower()
    return float(
        analysis.solve_power(
            effect_size=effect_size,
            nobs1=n_control,
            alpha=alpha,
            ratio=n_treatment / n_control,
            alternative="two-sided",
        )
    )


def minimum_detectable_effect(n_control, n_treatment, baseline_rate, alpha=0.05, power=0.80):
    """
    Invert the question. Given the traffic we have, what is the smallest lift we
    could reliably detect? Ask this before agreeing to run someone's experiment.
    """
    from scipy.optimize import brentq

    def gap(mde):
        target = baseline_rate * (1 + mde)
        return achieved_power(n_control, n_treatment, baseline_rate, target, alpha) - power

    try:
        return float(brentq(gap, 1e-5, 5.0))
    except ValueError:
        return None


def sample_size_curve(baseline_rate, mde_range, alpha=0.05, power=0.80, ratio=1.0):
    """Sample size as a function of the effect we want to detect."""
    return [
        {
            "relative_mde": mde,
            "n_total": required_sample_size(baseline_rate, mde, alpha, power, ratio)["n_total"],
        }
        for mde in mde_range
    ]


def runtime_days(n_total, daily_eligible_users):
    """Translate a sample size into the only unit a business cares about: time."""
    return float(np.ceil(n_total / daily_eligible_users))