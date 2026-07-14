"""
Simulate a fintech onboarding experiment.

The Criteo data is real but it is already finished. You cannot demonstrate power
analysis on a dataset that already has its outcomes in it, because the sample
size was decided by someone else years ago. So the project carries a second,
simulated experiment where we control the ground truth and can therefore:

  1. run the design phase honestly, before any data exists
  2. verify the analysis pipeline recovers an effect of a size we planted
  3. inject a sample ratio mismatch deliberately and confirm the check catches it

Scenario: a digital bank redesigns its onboarding screen.
  Primary   : account_funded, did the user deposit within seven days
  Secondary : kyc_completed
  Guardrail : support_ticket_raised, we do not want this going up
"""

import numpy as np
import pandas as pd


def simulate_experiment(
    n_users=80_000,
    control_share=0.5,
    baseline_funded=0.22,
    true_relative_lift=0.06,
    baseline_kyc=0.61,
    kyc_relative_lift=0.01,
    baseline_tickets=0.035,
    ticket_relative_lift=0.0,
    seed=42,
    inject_srm=False,
):
    rng = np.random.default_rng(seed)

    if inject_srm:
        # Break the split on purpose, by a margin small enough that no dashboard
        # would flag it and large enough to invalidate the experiment.
        control_share = control_share * 0.94

    group = rng.choice(["control", "treatment"], size=n_users,
                       p=[control_share, 1 - control_share])
    is_treatment = (group == "treatment").astype(int)

    funded_p = np.where(is_treatment == 1,
                        baseline_funded * (1 + true_relative_lift), baseline_funded)
    kyc_p = np.where(is_treatment == 1,
                     baseline_kyc * (1 + kyc_relative_lift), baseline_kyc)
    ticket_p = np.where(is_treatment == 1,
                        baseline_tickets * (1 + ticket_relative_lift), baseline_tickets)

    df = pd.DataFrame({
        "user_id": np.arange(n_users),
        "group": group,
        "is_treatment": is_treatment,
        "account_funded": rng.binomial(1, funded_p),
        "kyc_completed": rng.binomial(1, kyc_p),
        "support_ticket_raised": rng.binomial(1, ticket_p),
        "signup_day": rng.integers(1, 15, size=n_users),
        "device": rng.choice(["ios", "android", "web"], size=n_users, p=[0.42, 0.44, 0.14]),
    })

    metadata = {
        "true_relative_lift_primary": true_relative_lift,
        "baseline_funded": baseline_funded,
        "srm_injected": inject_srm,
        "seed": seed,
    }
    return df, metadata


if __name__ == "__main__":
    from pathlib import Path
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    df, meta = simulate_experiment()
    df.to_parquet("data/processed/simulated_onboarding.parquet", index=False)
    print(f"Simulated {len(df):,} users. Ground truth: {meta}")
    print(df.groupby("group")[["account_funded", "kyc_completed", "support_ticket_raised"]].mean())