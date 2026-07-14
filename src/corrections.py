"""
Multiple testing correction.

One test at alpha 0.05 carries a five percent false positive risk. Three tests
carry roughly fourteen percent. Six carry twenty six. Any experiment with a
primary metric, a secondary and a couple of guardrails is already in that
territory before anyone has sliced by platform.

Benjamini Hochberg is the default here rather than Bonferroni. Bonferroni
controls the probability of any false positive at all, which is the right target
when a single false claim is catastrophic. BH controls the expected proportion of
false claims among the ones you declare significant, which is what a product team
is actually managing. With a small metric family and one metric that genuinely
matters, Bonferroni spends power on secondaries we do not need certainty about.
"""

from statsmodels.stats.multitest import multipletests


def apply_correction(results_dict, method="fdr_bh", alpha=0.05):
    """
    results_dict : {metric_name: result_dict_with_a_p_value}
    method       : "bonferroni", "holm", or "fdr_bh"
    """
    metrics = list(results_dict.keys())
    p_values = [results_dict[m]["p_value"] for m in metrics]

    reject, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method=method)

    corrected = {}
    for i, metric in enumerate(metrics):
        entry = dict(results_dict[metric])
        entry["p_value_raw"] = entry["p_value"]
        entry["p_value_adjusted"] = float(p_adjusted[i])
        entry["significant_raw"] = bool(p_values[i] < alpha)
        entry["significant_adjusted"] = bool(reject[i])
        entry["correction_method"] = method
        corrected[metric] = entry

    return corrected


def correction_summary(corrected):
    """
    A table showing which conclusions survive correction.

    The column to read is "changed". Every True in it is a result a naive
    analysis would have reported as real. That is the moment a wrong decision
    gets made, and it is invisible unless you look for it.
    """
    return [
        {
            "metric": metric,
            "p_raw": round(r["p_value_raw"], 6),
            "p_adjusted": round(r["p_value_adjusted"], 6),
            "significant_before": r["significant_raw"],
            "significant_after": r["significant_adjusted"],
            "changed": r["significant_raw"] != r["significant_adjusted"],
        }
        for metric, r in corrected.items()
    ]


def compare_methods(results_dict, alpha=0.05):
    """
    Run all three corrections side by side. Useful in the writeup, because it
    shows the choice of method was a decision rather than a default.
    """
    out = {}
    for method in ["bonferroni", "holm", "fdr_bh"]:
        corrected = apply_correction(results_dict, method=method, alpha=alpha)
        out[method] = {
            m: {
                "p_adjusted": round(r["p_value_adjusted"], 6),
                "significant": r["significant_adjusted"],
            }
            for m, r in corrected.items()
        }
    return out