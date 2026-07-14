# A/B Test Analysis and Experiment Design

An end to end experimentation case study covering the full lifecycle of an online
controlled experiment: design, power analysis, sanity checking, inference,
multiple testing correction, and a recommendation a product team could act on.

Most A/B testing portfolio projects run a t test on two columns and stop. The
parts that matter in practice are the parts either side of that test. Deciding how
much data you need before you collect any. Checking that the randomisation actually
worked. Correcting for the fact that you looked at more than one metric. Translating
a p value into a decision, and being honest about when a null result means "no
effect" versus "we could not have seen it". Those are the steps here.

## The two experiments

**Criteo incrementality test.** A real randomised experiment from a live advertising
platform. 14 million users, an intentional 85 / 15 allocation, a 0.29 percent
conversion rate. Sector: advertising and e-commerce. Pulled from Hugging Face
(`criteo/criteo-uplift`), no manual download required.

**Fintech onboarding redesign.** Simulated, with a planted ground truth effect.
This exists because the design phase happens before data does, and you cannot
demonstrate a power analysis on a dataset whose sample size someone else fixed
years ago. It also lets us inject a sample ratio mismatch on purpose and confirm
the check catches it.

## The pipeline

| Stage | What happens | Module |
|---|---|---|
| Design | Set the MDE from business cost, compute sample size and runtime | `src/power.py` |
| Sanity | Chi square SRM test, overall and by segment, plus covariate balance | `src/srm.py` |
| Inference | Two proportion z test, Welch t test, bootstrap intervals | `src/stats_tests.py` |
| Correction | Benjamini Hochberg across the metric family | `src/corrections.py` |
| Decision | Ship, ship with caution, do not ship, extend | `src/decision.py` |

## Headline results

**Fintech onboarding redesign**
- Sample size computed in advance for a 6 percent MDE at 80 percent power
- SRM check passed
- Funding rate lift: +6.0 percent, 95 percent CI [1.8 percent, 10.4 percent]
- Guardrail on support tickets: unchanged
- Verdict: SHIP, with the lower bound caveat recorded explicitly

**Criteo incrementality test**
- Adequately powered to detect a lift well below 1 percent on conversion
- SRM check passed against the intended 85 / 15 allocation
- Conversion and visit both lift significantly under treatment

## Running it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_dataset.py # pulls 311 MB from Hugging Face, once
python -m src.simulate

pytest tests/ -v
jupyter notebook
```

Notebooks are numbered and run in order.

## Data handling

The Criteo file is 311 MB compressed, roughly 3 GB in memory as a pandas frame.
`scripts/download_dataset.py` streams it in chunks, writes a Parquet copy, and
accumulates exact counts as it goes. Every statistical test in this project consumes
those counts rather than the frame, because a two proportion z test needs four
integers, not fourteen million rows. The notebooks run in seconds on a laptop.

Nothing under `data/` is committed. A portfolio repository is not a data store.

## Design decisions worth defending

**Why the SRM test is against the intended ratio, not against balance.** Criteo
splits 85 / 15 on purpose, because withholding ads from control users costs money.
A naive check for a fifty fifty split fires immediately here with a p value near
zero, and would conclude the experiment is broken when it is working exactly as
designed. `tests/test_srm.py` locks that behaviour in.

**Why the SRM threshold is 0.001 and not 0.05.** A false alarm costs an afternoon
of investigation. A missed SRM costs a wrong shipping decision. The asymmetry
justifies being strict.

**Why Benjamini Hochberg and not Bonferroni.** Bonferroni controls the probability
of any false positive at all, which is right when a single false claim is
catastrophic. BH controls the expected proportion of false claims among those
declared significant, which is what a product team is actually managing. With a
small metric family and one metric that genuinely matters, Bonferroni spends power
on secondaries we do not need certainty about, and that power comes out of the
primary.

**Why the decision framework separates "no effect" from "underpowered".** A null
result from a well powered test is evidence that the effect is not there. A null
result from an underpowered test is evidence of nothing at all. They look identical
in a p value and they mean completely different things. Collapsing them is the most
common error in experiment reporting, and `src/decision.py` refuses to.

**Why significance is not the shipping criterion.** At 14 million users, almost
anything real is significant, and the p value stops carrying information. The
question that survives at that scale is whether the effect is large enough to be
worth having, which is what the confidence interval and the practical significance
threshold are for.

## Citation

Diemert, E., Betlei, A., Renaudin, C. and Amini, M. (2018) 'A Large Scale Benchmark
for Uplift Modeling', *Proceedings of the AdKDD and TargetAd Workshop, KDD*, London,
20 August. New York: ACM.

Dataset licensed CC BY-NC-SA 4.0.