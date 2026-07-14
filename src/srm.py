"""
Sample Ratio Mismatch detection.

If the observed split between arms differs from the intended split by more than
chance allows, the experiment is broken and nothing downstream can rescue it.
Stop. Do not read the metrics. Find the bug.

Common causes in production: bot filtering that removes traffic asymmetrically,
a redirect that fails on one platform, or an exposure event that only fires after
a render so slow-loading treatment users never get logged. All three produce a
biased sample. None of them are visible in the conversion numbers themselves.

The threshold is p < 0.001, not 0.05. A false alarm costs an afternoon of
investigation. A missed SRM costs a wrong shipping decision. The asymmetry
justifies being strict.

Note on this dataset: Criteo runs an intentional 85 / 15 split, so the naive
"is it 50 / 50" check would fire immediately and be wrong. The test must always
be against the intended ratio, never against balance.
"""

from scipy import stats

SRM_ALPHA = 0.001


def srm_check(observed_counts, expected_ratios, alpha=SRM_ALPHA):
    """
    observed_counts : {"control": 2096037, "treatment": 11883555}
    expected_ratios : {"control": 0.15, "treatment": 0.85}
    """
    arms = list(observed_counts.keys())
    observed = [observed_counts[a] for a in arms]
    total = sum(observed)

    ratio_sum = sum(expected_ratios[a] for a in arms)
    expected = [total * (expected_ratios[a] / ratio_sum) for a in arms]

    chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    passed = bool(p_value >= alpha)

    return {
        "arms": arms,
        "observed": observed,
        "expected": [round(e, 1) for e in expected],
        "observed_ratio": [round(o / total, 5) for o in observed],
        "expected_ratio": [round(expected_ratios[a] / ratio_sum, 5) for a in arms],
        "chi2": float(chi2),
        "p_value": float(p_value),
        "alpha": alpha,
        "passed": passed,
        "verdict": (
            "No sample ratio mismatch. The randomisation behaved as designed. Safe to proceed."
            if passed
            else "SRM DETECTED. Do not interpret any metric. Investigate assignment and "
                 "logging before reading a single result."
        ),
    }


def srm_by_segment(df, group_col, segment_col, expected_ratios, alpha=SRM_ALPHA):
    """
    An SRM that appears only inside one segment is the classic fingerprint of a
    platform specific bug. Run this even when the overall check passes, because
    two opposing segment level failures can cancel out into a clean total.
    """
    results = {}
    for segment, chunk in df.groupby(segment_col):
        counts = chunk[group_col].value_counts().to_dict()
        if len(counts) < 2:
            continue
        results[segment] = srm_check(counts, expected_ratios, alpha)
    return results