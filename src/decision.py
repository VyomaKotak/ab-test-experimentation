"""
Turn statistical output into a business decision.

Statistical significance is not a shipping decision. With fourteen million users
almost anything is significant, and the p value stops carrying information. The
question that survives at that scale is whether the effect is big enough to be
worth having, which is what the confidence interval and the practical
significance threshold are for.

Five verdicts, and the distinction between the last two is the one that most
experiment reports get wrong:

  SHIP                 significant, whole CI clears the practical bar, guardrails intact
  SHIP WITH CAUTION    significant and positive, but the lower bound sits below the bar
  DO NOT SHIP          significant in the wrong direction, or a guardrail breached
  DO NOT SHIP (null)   not significant, and the test had the power to find the effect,
                       so the effect is probably not there
  EXTEND OR ABANDON    not significant, but underpowered, so we learned nothing at all
"""


def make_decision(
    primary_result,
    guardrail_results=None,
    practical_significance_relative=0.05,
    achieved_power=None,
    power_threshold=0.80,
    guardrail_direction="increase_is_bad",
):
    guardrail_results = guardrail_results or {}

    ci_low, ci_high = primary_result["ci_relative"]
    significant = primary_result.get("significant_adjusted", primary_result["p_value"] < 0.05)

    breached = []
    for name, res in guardrail_results.items():
        sig = res.get("significant_adjusted", res["p_value"] < 0.05)
        moved_wrong_way = (
            res["relative_lift"] > 0 if guardrail_direction == "increase_is_bad"
            else res["relative_lift"] < 0
        )
        if sig and moved_wrong_way:
            breached.append(name)

    if breached:
        verdict = "DO NOT SHIP"
        reason = (
            f"Guardrail metrics moved in the harmful direction: {', '.join(breached)}. "
            "A win on the primary metric does not buy the right to degrade these. "
            "Roll back and understand the mechanism before trying again."
        )

    elif significant and ci_low > 0 and ci_low >= practical_significance_relative:
        verdict = "SHIP"
        reason = (
            f"Statistically significant, and the entire confidence interval "
            f"[{ci_low:.1%}, {ci_high:.1%}] sits above the practical significance threshold of "
            f"{practical_significance_relative:.1%}. Even the pessimistic end of the range is "
            "a lift worth paying for. No guardrail was harmed."
        )

    elif significant and ci_low > 0:
        verdict = "SHIP WITH CAUTION"
        reason = (
            f"Statistically significant and positive, confidence interval "
            f"[{ci_low:.1%}, {ci_high:.1%}]. But the lower bound falls below the practical "
            f"threshold of {practical_significance_relative:.1%}, which means the true effect "
            "may be too small to recover the cost of building and maintaining the change. "
            "Whether that is acceptable depends on what the change costs, and that is a product "
            "call rather than a statistical one. The statistics have done all they can here."
        )

    elif significant and ci_high < 0:
        verdict = "DO NOT SHIP"
        reason = (
            f"The treatment performs significantly worse than control, confidence interval "
            f"[{ci_low:.1%}, {ci_high:.1%}]. Roll back."
        )

    elif achieved_power is not None and achieved_power < power_threshold:
        verdict = "EXTEND OR ABANDON"
        reason = (
            f"No significant effect, but achieved power was only {achieved_power:.0%}, below the "
            f"{power_threshold:.0%} we designed for. This is not evidence that the treatment does "
            "nothing. It is an absence of evidence either way. Either extend the test to reach the "
            "planned sample size, or accept that an effect of the size we care about is not "
            "detectable with the traffic available and stop spending on it."
        )

    else:
        verdict = "DO NOT SHIP"
        reason = (
            f"No significant effect. The confidence interval [{ci_low:.1%}, {ci_high:.1%}] includes "
            "zero, and the test was adequately powered to detect the lift we cared about. We can be "
            "reasonably confident the treatment does not produce it. Keep control and put the "
            "engineering effort somewhere with a better expected return."
        )

    return {
        "verdict": verdict,
        "reason": reason,
        "relative_lift": primary_result["relative_lift"],
        "absolute_lift": primary_result["absolute_lift"],
        "ci_relative": (ci_low, ci_high),
        "p_value_adjusted": primary_result.get("p_value_adjusted"),
        "guardrails_breached": breached,
        "achieved_power": achieved_power,
        "practical_threshold": practical_significance_relative,
    }