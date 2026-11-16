# ShopTheLook: A/B test plan

> **Status: plan only.** No experiment has been run. The baseline rates, traffic and
> adoption below are **planning assumptions** chosen to be plausible for a large Indian
> fashion app. They are not measured data. The numbers are computed by
> `analysis/power_analysis.py` (outputs in `results/ab_planning/`). A worked
> example of the analysis on **synthetic** data is in
> `analysis/simulated_ab_test.ipynb`, clearly labelled as a simulation.

## 1. Hypothesis

> Giving shoppers a **Shop the Look** entry point (upload an outfit photo → similar products per garment)
> will **increase the share of users who add an item to cart**. Shoppers who struggle to describe
> an item in words will find it faster than with text search alone.

* **H0:** user-level ATC conversion is the same in treatment and control.
* **H1:** it differs (two-sided test; we ship only on a positive, significant result).

## 2. Variants

| | Control (A) | Treatment (B) |
|---|---|---|
| Experience | Today's app: **text search only** (search bar, filters, autocomplete) | Same, **plus** a camera icon in the search bar and a "Shop the Look" card on the home feed. Tapping either opens upload → detected garments → top-5 similar products per garment |
| Ranking of normal search | unchanged | unchanged |
| Share of eligible traffic | 50% | 50% |

## 3. Randomization

| | Decision | Why |
|---|---|---|
| Unit | **User** (logged-in user ID; device ID for logged-out users, merged on login) | The primary metric is user-level. Session-level randomization would show the same person both experiences, which contaminates the result and ignores learning ("I know I can upload a photo now"). |
| Eligibility | Users on app versions that support the camera entry point, in fashion categories, during the test window | Users who can't see the feature shouldn't dilute either arm |
| Assignment | Deterministic hash of `user_id + experiment_salt` → bucket 0–999. 0–499 = A, 500–999 = B | Stable across sessions and devices, reproducible, no lookup table |
| Sample-ratio check | χ² test on arm counts daily. **p < 0.001 ⇒ stop and debug** (logging or assignment bug), do not read results | An SRM invalidates every downstream metric |

## 4. Metrics

| Role | Metric | Notes |
|---|---|---|
| **Primary** | **User-level ATC conversion**: share of assigned users with ≥ 1 add-to-cart during the test | Defined identically in both arms. The North Star (ATC rate *within look sessions*) only exists in B, so it can't be compared across arms. |
| Secondary | Search-to-cart conversion · time to first ATC · orders per user · GMV per user | Explain the mechanism; not decision-making on their own |
| Feature health (B only) | Entry-point CTR, upload completion, detection success, recommendation CTR, North Star | Diagnose *why* the primary moved or didn't |
| Guardrails | Return rate (30-day lag) · p95 latency ≤ 1.5 s · bounce rate · crash rate · overall ATC not negative | See `METRICS.md` §3 for thresholds |

## 5. Minimum detectable effect, sample size, duration

**Planning assumptions** (inputs to `analysis/power_analysis.py`):

| Input | Value | Reasoning |
|---|---|---|
| Baseline user-level ATC conversion (control) | **8.0%** over the test window | Assumption. Replace with the real baseline from the last 4 weeks before launch |
| Minimum detectable effect | **+3% relative** (8.00% → 8.24%, +0.24 pp) | The smallest lift worth the build and maintenance cost of a CV pipeline (assumption) |
| Significance α (two-sided) / power | 0.05 / 0.80 | standard |
| New eligible users entering the test per day | **100,000** (both arms combined) | Assumption: a slice of traffic allocated to this experiment |
| Adoption (share of B users who use the entry point) | **12%** | Assumption. Visual search is an opt-in behaviour |

**Results** (`results/ab_planning/power_plan.json`):

| Output | Value |
|---|---|
| Users needed **per arm** | **203,322** (statsmodels cross-check: 203,315) |
| Total users | 406,644 |
| Days to reach sample at 100k/day | 5 |
| **Planned duration** | **14 days**: we run two full weekly cycles anyway (weekday/weekend and payday shopping patterns differ) |
| MDE actually achieved at 14 days (700k users per arm) | **+1.6% relative** |
| Implied lift needed **among adopters** | +25% relative (= 3% / 12% adoption) |

**Formula** (unpooled two-proportion z-test):

n per arm = (z₁₋α/₂ + z₁₋β)² · [p₀(1−p₀) + p₁(1−p₁)] / (p₁ − p₀)²,  with p₁ = p₀ · (1 + MDE)

**Dilution is the main risk to power.** Only ~12% of treatment users will try the feature, so
a +25% lift among adopters shows up as only ~+3% across the arm. We therefore also report a
**triggered analysis**: users who *would have seen* the entry point (logged in both arms at the
moment of first search) are compared. This removes users the feature could never affect and
increases sensitivity without breaking randomization.

**Sensitivity:** users per arm for other baselines and MDEs are in
`results/ab_planning/sample_size_grid.csv`. For example, at a 4% baseline and a 3% MDE you
need 424,617 per arm.

**Variance reduction (optional):** CUPED adjusts each user's outcome by their ATC count in the
28 days before the test. How much it helps depends entirely on how well pre-period behaviour
predicts in-test behaviour. In our synthetic simulation it cut variance by only ~2%, so we treat
it as a pre-registered secondary readout and measure its gain on real data. It is never a way to
rescue a failed primary.

**Monte Carlo check of this plan** (`analysis/simulated_ab_test.ipynb`, synthetic): at 203,322
users per arm and a true +3% lift, 78% of 2,000 simulated experiments were significant (target
80%; MC standard error ≈ 0.9 pp), and 5.0% were significant with no true effect. The analysis uses
a pooled z-test, which is slightly more conservative than the unpooled sizing formula, so round the
sample up by ~5% if 80% must be guaranteed. The 14-day plan already gives ~3.4× this sample.

## 6. Analysis plan (pre-registered)

1. **Validity first:** SRM χ² test, no logging gaps, both arms on the same app versions.
2. **Primary:** difference in proportions (B − A), 95% CI, two-sided z-test. Relative lift with a delta-method CI.
3. **Guardrails:** one-sided non-inferiority tests against the thresholds in `METRICS.md`.
   Return rate is read **30 days after** the last order in the window, so the final call waits for it.
4. **Novelty check:** plot the daily lift. If it decays steadily across the two weeks, extend the test
   or keep a 5% long-term holdout before calling the result.
5. **Segments** (exploratory, Bonferroni-adjusted): new vs returning, garment group, photo source, device tier.
6. **No peeking:** we look at the primary only at day 14. Daily dashboards show guardrails and SRM only.

## 7. Decision rules

| Outcome | Condition | Decision |
|---|---|---|
| **Ship** | Primary lift > 0 with 95% CI excluding 0, **and** no guardrail breached | Roll out 10% → 50% → 100% over 3 weeks, keep a 5% holdout for 8 weeks to measure long-term effect and returns |
| **Iterate** | Primary not significant **but** feature-health metrics are promising (e.g. adopters' ATC clearly higher, recommendation CTR ≥ text-search CTR), **or** a guardrail is breached for a fixable reason (latency, one bad category) | Fix the bottleneck the funnel points to (entry-point visibility if adoption is low, retrieval if CTR is low, catalog gaps if one category fails) and re-test |
| **Kill** | Primary significantly negative, **or** return rate of look-attributed orders up > 1 pp with CI excluding 0, **or** adoption < 3% after the iteration round | Remove the entry point. Keep the detection + embedding stack for internal uses (catalog tagging, "similar items" on product pages) |

## 8. Risks to the experiment itself

* **Network effects / interference:** minimal. Recommendations don't depend on other users' actions.
* **Inventory interference:** treatment could deplete stock of popular "lookalike" items that control users would also buy. This is small at 50/50 on a large catalog, and we monitor out-of-stock rates.
* **Seasonality / sales events:** don't run during End of Reason Sale-type events. Promo traffic behaves differently and inflates ATC.
* **Logging:** "ATC from a recommended item" needs a `source=shop_the_look` attribution parameter on the cart event. Verify it in a 1% dogfood before launch.
