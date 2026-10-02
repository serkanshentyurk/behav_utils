# Configuration Guide

## Overview

behav_utils uses a single YAML config file to map your experiment's data format to the library's internal structure. This means you never need to change library code when you move to a new experiment, lab, or task variant — you just write a new config.

The config defines four things:
1. **Where your files are** and how they're organised on disk
2. **What your CSV columns are called** and how to interpret them
3. **How your task works** (boundary, stimulus range, choice encoding)
4. **Default analysis and plotting settings**

---

## Minimal Config

The absolute minimum config that will load data:

```yaml
project:
  name: "My Experiment"

file_structure:
  data_dir: "/path/to/data"

columns:
  trial_number:
    csv_name: "Trial_Number"
    dtype: int
  stimulus:
    csv_name: "Stimulus"
    dtype: float
  choice:
    csv_name: "Choice"
    dtype: int
  outcome:
    csv_name: "Outcome"
    dtype: str
  correct:
    csv_name: "Correct"
    dtype: bool
```

Which columns are *required* is decided by your config, not by the library: any column you list **without** `optional: true` must be present in the CSV. Structurally, only `trial_number` is hard-required by the loader; the example above also makes `stimulus`, `choice`, `outcome`, and `correct` required simply by listing them as non-optional, which is the typical choice for a 2-AFC task. If your CSV uses different names, just change the `csv_name` values; to make any of these tolerated-when-missing, add `optional: true`.

---

## Full Config Reference

### project

```yaml
project:
  name: "Sound Categorisation"           # Project name (for display)
  description: "2-AFC amplitude task"    # Optional description
```

These are just labels — they don't affect loading or analysis.

---

### file_structure

Tells the loader where to find your data and how directories are named.

```yaml
file_structure:
  data_dir: "/path/to/data"
  animal_pattern: "{animal_id}"
  session_pattern: "TASK_{animal_id}_{date}"
  behaviour_file: "trial_summary*.csv"
  date_format: "{year}_{month}_{day}"
  date_regex: "(?P<year>\\d{4})_(?P<month>\\d{1,2})_(?P<day>\\d{1,2})"
```

**Expected directory structure:**

```
data_dir/
├── A01/                          ← animal_pattern
│   ├── TASK_A01_2026_1_15/  ← session_pattern
│   │   └── trial_summary.csv      ← behaviour_file
│   ├── TASK_A01_2026_1_16/
│   │   └── trial_summary.csv
│   └── ...
├── A02/
│   └── ...
└── A03/
    └── ...
```

| Field | Default | Description |
|-------|---------|-------------|
| `data_dir` | `"."` | Root directory containing animal subdirectories |
| `animal_pattern` | `"{animal_id}"` | Animal directory naming pattern (currently just used for documentation) |
| `session_pattern` | `"{protocol}_{animal_id}_{date}"` | Session directory naming pattern (documentation) |
| `behaviour_file` | `"trial_summary*.csv"` | Glob pattern to find the behaviour CSV inside each session directory |
| `date_format` | `"{year}_{month}_{day}"` | How dates appear in directory names (documentation) |
| `date_regex` | `"(\\d{4})_(\\d{1,2})_(\\d{1,2})"` | Regex to extract year, month, day from directory name |

**Adapting for your file structure:**

If your sessions are in directories named `2026-01-15_Session1`:
```yaml
  date_regex: "(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})"
```

If your CSVs are named `behaviour_log.csv`:
```yaml
  behaviour_file: "behaviour_log*.csv"
```

If you have one CSV per animal (not per session), this loader won't work directly — you'd need a preprocessing step to split the CSV into per-session files first, or a custom loader.

---

### task

Defines the experimental paradigm.

```yaml
task:
  boundary: 0.0                    # Category boundary in stimulus space
  stimulus_range: [-1.0, 1.0]      # Nominal stimulus range
  n_categories: 2                  # Number of categories (always 2 for now)
  category_rule: "above_boundary"  # How stimulus maps to category

  choice_mapping:
    type: "spatial_to_category"
    no_response_value: 0
    contingency_field: "sound_contingency"
    contingency_rules:
      Low_Left_High_Right:
        -1: 0
        1: 1
      Low_Right_High_Left:
        -1: 1
        1: 0
```

#### category_rule

How categories are derived from stimulus values:
- `"above_boundary"` — stimulus > boundary → category 1 (B). This is the standard.
- `"below_boundary"` — stimulus < boundary → category 1 (B). For inverted mappings.

#### choice_mapping

How raw choice values from the CSV get converted to category space (0=A, 1=B, NaN=no response).

**type options:**

| Type | When to use | What it does |
|------|------------|--------------|
| `"spatial_to_category"` | Choices are spatial (left/right) and need converting using a contingency rule | Reads contingency from session metadata, applies mapping |
| `"identity"` | Choices are already 0/1 in the CSV | Just marks no-response values as NaN |
| `"none"` | You don't want any conversion | Stores raw values as-is |

**spatial_to_category fields:**

| Field | Description |
|-------|-------------|
| `no_response_value` | Raw value that means "no response" (→ NaN). Usually 0. |
| `contingency_field` | Session metadata field containing the response mapping name |
| `contingency_rules` | Dict of `{contingency_name: {raw_value: category_value}}` |

**Example: Choices are already 0/1:**
```yaml
  choice_mapping:
    type: "identity"
    no_response_value: -1    # -1 in CSV means no response
```

**Example: No conversion needed:**
```yaml
  choice_mapping:
    type: "none"
```

**Example: Left/right with a single mapping (no contingency variation):**
```yaml
  choice_mapping:
    type: "spatial_to_category"
    no_response_value: 0
    contingency_field: "dummy"    # won't be used
    contingency_rules:
      default:                    # use any name
        -1: 0                     # left → A
        1: 1                      # right → B
```
Then make sure your session metadata has a matching field, or use `"identity"` type instead.

---

### columns

Maps your CSV column names to behav_utils internal names. This is the core of the config — get this right and everything works.

#### Typical (non-optional) columns

A standard 2-AFC config lists these as non-optional, so they must be present in the
CSV. Only `trial_number` is structurally required by the loader — the rest are
required *because* this config lists them without `optional: true`. Mark any of them
`optional: true` if your task lacks them:

```yaml
columns:
  trial_number:
    csv_name: "Trial_Number"     # Your CSV's column name
    dtype: int

  stimulus:
    csv_name: "Stim_Relative"
    dtype: float

  choice:
    csv_name: "Choice"
    dtype: int

  outcome:
    csv_name: "Trial_Outcome"
    dtype: str

  correct:
    csv_name: "Correct"
    dtype: bool
```

#### Column mapping format

Each column mapping can be either a simple string (just the CSV column name) or a full specification:

```yaml
  # Simple form (assumes dtype: float, required)
  stimulus: "Stim_Relative"

  # Full form
  stimulus:
    csv_name: "Stim_Relative"    # Column name in your CSV
    dtype: float                  # Expected type: float, int, str, bool
    optional: false               # If true, missing column is OK
    default: null                 # Default value when column missing or NaN
    mapping: null                 # Value mapping (see below)
```

#### dtype

| Type | Behaviour |
|------|-----------|
| `float` | Numeric conversion via `pd.to_numeric`. NaN for unparseable values. |
| `int` | Numeric conversion, then cast to int. NaN → default value (0 if not specified). |
| `str` | String conversion. NaN → empty string. |
| `bool` | Boolean conversion. Handles `True`/`False` strings and 0/1 integers. |

#### optional and default

```yaml
  reaction_time:
    csv_name: "Response_Latency"
    dtype: float
    optional: true          # Won't error if column is missing from CSV
    default: null           # Fills with NaN when missing (null = NaN for floats)

  opto_on:
    csv_name: "Opto_On"
    dtype: bool
    optional: true
    default: false          # Fills with False when missing
```

If `optional: false` (the default) and the column is missing from the CSV, loading will raise an error with a clear message telling you which column is missing.

#### mapping (value mapping)

Converts raw CSV values to different values during loading:

```yaml
  choice:
    csv_name: "Response"
    dtype: str
    mapping:
      left: -1
      right: 1
      none: 0
```

This reads the `Response` column, and wherever it says `"left"` it becomes `-1`, `"right"` becomes `1`, etc.

Mapping is applied **before** dtype conversion, so you can map strings to numbers.

#### Common optional columns

Here are columns you might want to add depending on your experiment:

```yaml
  # Reaction time
  reaction_time:
    csv_name: "Response_Latency"
    dtype: float
    optional: true

  # Abort / broken fixation
  abort:
    csv_name: "Abort_Trial"
    dtype: bool
    optional: true
    default: false

  # Optogenetic manipulation
  opto_on:
    csv_name: "Opto_On"
    dtype: bool
    optional: true
    default: false

  # Stimulus distribution
  distribution:
    csv_name: "Distribution"
    dtype: str
    optional: true
    default: "uniform"
```

Any column name you use here becomes accessible via `session.trials.get_field('column_name')` or appears in `session.trials.optional_fields`.

---

### session_metadata

Columns that are constant within a session (task parameters, not trial data). Extracted from the first row of each CSV.

```yaml
session_metadata:
  animal_id:
    csv_name: "Animal_ID"
    dtype: str

  stage:
    csv_name: "Stage"
    dtype: str

  sound_contingency:
    csv_name: "Sound_Contingency"
    dtype: str

  response_window:
    csv_name: "Response_Window"
    dtype: str
    optional: true
    parse_timespan: true        # Parse "00:00:10" → 10.0 (seconds)
```

#### parse_timespan

Set `parse_timespan: true` for columns that contain time values as strings in `HH:MM:SS` or `HH:MM:SS.ffffff` format (common with Bonsai). The loader will convert them to float seconds.

```yaml
  sound_duration:
    csv_name: "Sound_Duration"
    dtype: str
    optional: true
    parse_timespan: true     # "00:00:01.500000" → 1.5
```

Session metadata is accessible via:
```python
session.metadata.stage                    # attribute access
session.metadata.get('sound_contingency') # dict access
session.metadata.fields                   # raw dict
```

---

### extra_columns

CSV columns to load but not map to specific fields. They end up in `session.trials.extra['column_name']`.

```yaml
extra_columns:
  - "Trial_End_Time"
  - "Running_Score"
```

Any column in the CSV that isn't in `columns` or `session_metadata` is **also** automatically stored in `extra` — so `extra_columns` is just for being explicit about which unmapped columns you care about.

---

### cohorts

Named lists of animal ids. The library only stores them; a project uses them to pick which animals a
report or a notebook covers. Read with `behav_utils.config.schema.load_cohorts(path)` (import-safe,
touches nothing else).

```yaml
cohorts:
  pilot:    [A01, A02, A03]
  main:     [A04, A05, A06, A07]
```

Per-animal attributes (group, sex, genotype, …) do **not** go here: put them in
`<data root>/animal_metadata.json` (`{"A01": {"group": "control"}, …}`); they are merged into
`animal.metadata` at load time.

### session_types

Session types the data cannot derive on its own. The library derives `'regular'` and `'opto'` from the
trials; anything else (sham-light sessions, washout sessions, a control site) is your vocabulary and is
stamped from this block at load time, per animal and date:

```yaml
session_types:
  sham:                              # type name — yours
    A01: [20240315, 20240316]
    A02: 20240301                    # a single date is fine
  washout:
    A01: [20240320]
```

Dates match the session's date (`YYYYMMDD`). The stamped type is what `select_sessions(session_type=…)`
and `exclude_types=(…)` filter on, and what `session.session_type` / `animal.session_table` show.

### session_presets

Named session selections, registered at load time and used as `select_sessions(animal, preset='…')`.
Each preset is a set of `SessionFilter` fields:

| field | meaning |
|---|---|
| `stage` | one stage name or a list (OR) |
| `distribution` | one distribution label or a list (OR) |
| `session_type` | keep only this type (or list); overrides `exclude_types` |
| `exclude_types` | types to drop, e.g. `[sham, washout]` |
| `min_accuracy`, `max_accuracy` | accuracy over valid trials |
| `min_trials` | valid trials per session (default 10) |
| `first_n`, `last_n`, `last_fraction` | take the first/last N sessions, or the last fraction, of what remains |
| `after_session_idx`, `before_session_idx`, `session_indices` | by position |
| `exclude_opto` | drop sessions that contain opto trials |

```yaml
session_presets:
  expert:
    stage: task
    distribution: uniform
    min_accuracy: 0.70
    last_fraction: 0.50
    exclude_types: [sham, washout]
  naive:
    stage: task
    first_n: 5
    exclude_types: [sham, washout]
```

Order of application: stage → distribution → session type / exclusions → accuracy and trial-count
thresholds → positional cuts. `select_sessions(..., preset=…, **overrides)` lets a call override single
fields.

### sessions_to_ignore

Sessions dropped at load time — never built, never in a snapshot. Same shape as one `session_types`
entry: `{animal_id: [YYYYMMDD, …]}`.

```yaml
sessions_to_ignore:
  A03: [20240228]            # rig fault
```

---

## Complete examples

Three complete, loadable configs, each for a different kind of experiment:

- [`config_examples/orientation_discrimination.yaml`](config_examples/orientation_discrimination.yaml) — visual orientation discrimination (different lab)
- [`config_examples/go_nogo_minimal.yaml`](config_examples/go_nogo_minimal.yaml) — auditory go/no-go (minimal columns)
- [`config_examples/unusual_encoding.yaml`](config_examples/unusual_encoding.yaml) — existing dataset with unusual encoding

Check any of them with `python -c "from behav_utils.config.schema import load_config; load_config('docs/config_examples/<file>')"`.

## Validation

After creating your config, validate it against a real CSV:

```python
from behav_utils.config.schema import load_config, validate_csv_against_config
import pandas as pd

config = load_config('config.yaml')
df = pd.read_csv('path/to/one/session.csv')

result = validate_csv_against_config(list(df.columns), config)

print(f"Matched: {len(result['matched'])}")
print(f"Missing required: {result['missing_required']}")   # These will cause errors
print(f"Missing optional: {result['missing_optional']}")   # These are OK
print(f"Unmapped: {result['unmapped']}")                    # In CSV but not in config
```

If `missing_required` is non-empty, you have a column name mismatch — fix the `csv_name` values in your config.

---

## Troubleshooting

**"Required column 'X' not found"**
Your CSV uses a different column name. Check `csv_name` in your config matches exactly (case-sensitive).

**"Unknown contingency 'X'"**
Your CSV has a contingency value not listed in `contingency_rules`. Add it to the config.

**Dates not parsing correctly**
Check `date_regex` matches your directory naming convention. Use named groups `(?P<year>...)` for clarity. Test with:
```python
import re
match = re.search(config.file_structure.date_regex, "your_directory_name")
print(match.groups())
```

**TimeSpan columns showing as strings**
Add `parse_timespan: true` to the session_metadata entry.

**Values look wrong after loading**
Check `mapping` — value mappings are applied before dtype conversion. If your CSV has `"Left"` and you're mapping to `-1`, make sure the mapping key matches exactly: `Left: -1` not `left: -1` (YAML is case-sensitive for string keys).

**Too many NaN values**
Check whether your CSV has empty cells, `"NA"` strings, or unusual missing value indicators. The loader uses `pd.to_numeric(errors='coerce')` for floats, which converts unparseable values to NaN.
