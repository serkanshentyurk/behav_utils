# Summary Statistics Reference

## Overview

behav_utils provides a registry of summary statistics for 2-AFC behavioural data. Each stat is a function that takes `(choices, stimuli, categories)` arrays and returns a scalar or dict.

All stats handle NaN choices (no-response trials) automatically. All stats work on both single-session (1D) and multi-session (2D) arrays.

### Usage

```python
from behav_utils.analysis import compute_summary_stats, list_available_stats

# See what's available
print(list_available_stats())

# Compute specific stats
stats = compute_summary_stats(
    choices, stimuli, categories,
    stat_names=['accuracy', 'recency', 'psychometric'],
    return_dict=True,
)

# Compute via data class
stats = session.stats(['accuracy', 'recency'])

# Feature matrix (all stats × all sessions)
df = animal.feature_matrix()
```

### Adding Custom Stats

```python
from behav_utils.analysis.summary_stats import register_stat

@register_stat('my_stat')
def compute_my_stat(choices, stimuli, categories):
    valid = ~np.isnan(choices)
    return float(np.mean(choices[valid]))
```

---

## Performance Statistics

### accuracy
**Overall proportion correct.**

Formula: `mean(choice == category)` over valid trials.

Range: [0, 1]. Chance = 0.5 for 2-AFC.

Tracks: learning (increases naive → expert).

### hard_accuracy
**Accuracy on hard trials only** (|stimulus| < `hard_threshold`).

Trials near the category boundary are hardest to discriminate. This measures performance on these ambiguous trials specifically.

Default threshold: 0.3 (configurable).

### easy_accuracy
**Accuracy on easy trials only** (|stimulus| ≥ `hard_threshold`).

Trials far from boundary. Should approach ceiling early in training.

### hard_easy_ratio
**Ratio of hard to easy accuracy.**

Formula: `hard_accuracy / easy_accuracy`.

Low ratio = flat performance across difficulty (chance-level or strong bias). High ratio approaching 1.0 = good discrimination even near boundary.

---

## Psychometric Parameters

### psychometric
**Fitted cumulative Gaussian parameters.** Returns a dict:

| Key | Description | Interpretation |
|-----|-------------|----------------|
| `pse` | Point of subjective equality (μ) | Stimulus value at 50% choice B. 0 = unbiased. |
| `slope` | Psychometric slope (σ) | Lower = steeper = better discrimination. |
| `lapse_low` | Lower lapse rate (γ) | P(choose B) floor. Guessing rate for category A stimuli. |
| `lapse_high` | Upper lapse rate (λ) | 1 - P(choose B) ceiling. Lapse rate for category B stimuli. |

Model: `P(B) = γ + (1 - γ - λ) × Φ((x - μ) / σ)`

**Reliability guard:** If slope > 5.0 or |PSE| > 0.99, PSE and slope are set to NaN (fit is unreliable — flat psychometric curve from chance performance or strong side bias). Lapse parameters are preserved.

### psychometric_gof
**Psychometric goodness-of-fit (R²).**

R² between binned observed choice proportions and fitted curve. Tracks learning: naive sessions have low R² (noisy), expert sessions have high R².

---

## Serial Dependence / History Effects

### recency
**Effect of previous trial's CATEGORY on current choice.**

Formula: `P(choose B | prev_category = B) − P(choose B | prev_category = A)`

High recency = recent trial categories strongly influence current choice (high learning rate / model updating). Low recency = stable behaviour (inference mode).

Note: Conditions on abstract category label, not stimulus magnitude.

### stimulus_recency
**Effect of previous trial's STIMULUS VALUE on current choice.**

Formula: `P(choose B | prev_stimulus > 0) − P(choose B | prev_stimulus ≤ 0)`

Same concept as recency but conditions on stimulus position rather than category. Maps more directly onto perceptual serial dependence mechanisms.

### recency_divergence
**Difference between stimulus-based and category-based recency.**

Formula: `stimulus_recency − recency`

For uniform stimulus distributions, these are highly correlated and divergence ≈ 0. After a distribution shift (stimuli concentrated on one side), they can diverge because category and stimulus magnitude become confounded.

- Positive: serial dependence is more sensory than categorical
- Negative: more categorical than sensory

### choice_autocorr
**Autocorrelation of choice sequence at lag 1.**

Pearson correlation between `choice_t` and `choice_{t-1}`. Measures raw choice repetition tendency beyond what stimulus would predict.

### perseveration
**Excess same-choice repetition beyond stimulus prediction.**

Formula: `observed_repeat_rate − expected_repeat_rate`

Expected rate computed from binned `P(B|stimulus)`. Positive = animal repeats choices more than stimulus alone predicts.

---

## Win-Stay / Lose-Shift

### win_stay
**Win-stay tendency: P(repeat | rewarded) − P(repeat | unrewarded).**

Positive = exploits correct responses. Near zero = ignores feedback. Measures the strength of outcome-dependent choice updating.

### win_stay_rate
**Raw win-stay rate: P(repeat choice | previous trial rewarded).**

Unlike `win_stay` (which is a contrast), this is the raw probability.

### lose_shift
**Lose-shift rate: P(switch | unrewarded).**

High = responsive to negative feedback. Low = perseverative despite errors.

---

## Stimulus Sensitivity

### stimulus_sensitivity
**Correlation between stimulus value and choice.**

Pearson correlation of `(stimulus, choice)` over valid trials. High = choices driven by stimulus (expert behaviour). Low = choices independent of stimulus (random or side-biased).

### side_bias
**Overall tendency to choose B: P(choose B) − 0.5.**

Positive = biased toward B. Negative = biased toward A. Should be near 0 for unbiased performance.

---

## Choice Entropy

### choice_entropy
**Mean entropy of choice distribution across stimulus bins.**

For each stimulus bin, computes binary entropy of `P(B|bin)`, then averages across bins. Normalised by log(2) so range is [0, 1].

- High entropy = random/uncertain choices (all bins near 50%)
- Low entropy = deterministic choices (each bin near 0% or 100%)

Tracks learning: decreases as animal improves discrimination.

---

## History Regression

### logistic_history
**L2-regularised logistic regression of current choice on stimulus + trial history.**

Returns a dict of regression weights:

| Key | Description |
|-----|-------------|
| `w_stimulus` | Weight of current stimulus (sensitivity) |
| `w_prev_choice_1..3` | Weight of previous 1–3 choices |
| `w_prev_outcome_1..3` | Weight of previous 1–3 outcomes |
| `history_decay` | Ratio of lag-3 to lag-1 history weights |

L2 penalty (default 0.1) prevents weight explosion from near-complete separation in expert sessions. Applied to all weights except intercept.

### history_interaction_r2
**How much trial history improves choice prediction beyond stimulus alone.**

McFadden's pseudo-R² difference between:
1. Stimulus-only model: `choice ~ stimulus`
2. Full model: `choice ~ stimulus + prev_choices + prev_outcomes`

High values = history-dependent behaviour (high learning rate). Low values = stimulus-driven behaviour (expert).

---

## Serial Dependence Profile

### sd_profile
**Scalar features from the serial dependence profile.** Returns a dict:

| Key | Description |
|-----|-------------|
| `sd_slope` | Linear slope of SD profile vs bin centre |
| `sd_curvature` | Quadratic coefficient (boundary-concentrated pattern) |
| `sd_range` | Max − min of profile values (total SD magnitude) |

Uses a fast raw computation (no psychometric fitting) for efficiency in pipelines.

---

## Conditional Psychometry

### conditional_psychometric
**Full psychometric fit per previous-stimulus bin.** Returns a dict:

| Key pattern | Description |
|-------------|-------------|
| `cond_pse_0..7` | PSE per previous-stimulus bin |
| `cond_slope_0..7` | Slope per previous-stimulus bin |
| `cond_lapse_low_0..7` | Lower lapse per bin |
| `cond_lapse_high_0..7` | Upper lapse per bin |

Total: 4 × n_bins values (32 for default n_bins=8).

Bins with too few trials (< `min_trials_per_bin`) fall back to the unconditional fit parameters.

### update_matrix
**Empirical update matrix via canonical psychometric-fit method.** Returns a dict:

| Key pattern | Description |
|-------------|-------------|
| `um_i_j` | Update matrix entry: shift in P(B) at current bin `i` given previous bin `j` |

Total: n_bins² values (64 for default n_bins=8).

Computed as: conditional P(B) minus overall P(B) for each current × previous bin combination. Post-correct trials only by default.

---

## Feature Matrix Stats

The default set used by `build_feature_matrix()`:

```python
FEATURE_MATRIX_STATS = [
    'accuracy', 'psychometric', 'psychometric_gof',
    'recency', 'stimulus_recency', 'recency_divergence',
    'win_stay', 'lose_shift',
    'stimulus_sensitivity', 'side_bias', 'choice_autocorr', 'choice_entropy',
    'perseveration', 'hard_easy_ratio', 'hard_accuracy', 'easy_accuracy',
    'history_interaction_r2', 'sd_profile',
    'logistic_history',
    'update_matrix',
    'conditional_psychometric',
]
```

This produces ~120 scalar features per session (most from the 64-value update matrix and 32-value conditional psychometric). For SLDS/HMM input, you'd typically select a subset of 6–8 non-redundant scalar stats.

---

## Per-Session vs Multi-Session

All stat functions accept either:
- **1D arrays** (n_trials,) — single session
- **2D arrays** (n_trials, n_sessions) — equal-length sessions (vectorised, faster for SBI)

For variable-length sessions (real data), use `compute_summary_stats_per_session()` which loops internally:

```python
from behav_utils.analysis import compute_summary_stats_per_session

# List of dicts (variable length)
session_data = [
    {'choices': ch1, 'stimuli': s1, 'categories': c1},
    {'choices': ch2, 'stimuli': s2, 'categories': c2},
]
results = compute_summary_stats_per_session(session_data, return_dict=True)

# 2D arrays (equal length, faster)
session_data = {
    'choices': np.stack([ch1, ch2], axis=1),
    'stimuli': np.stack([s1, s2], axis=1),
    'categories': np.stack([c1, c2], axis=1),
}
results = compute_summary_stats_per_session(session_data, return_dict=True)
```

The function auto-detects the format and uses the appropriate code path.
