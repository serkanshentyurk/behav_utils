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

Four columns are required: `trial_number`, `stimulus`, `choice`, and `outcome`. Everything else is optional. If your CSV has different column names, just change the `csv_name` values.

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
  session_pattern: "SOUND_CAT_{animal_id}_{date}"
  behaviour_file: "trial_summary*.csv"
  date_format: "{year}_{month}_{day}"
  date_regex: "(?P<year>\\d{4})_(?P<month>\\d{1,2})_(?P<day>\\d{1,2})"
```

**Expected directory structure:**

```
data_dir/
├── SS01/                          ← animal_pattern
│   ├── SOUND_CAT_SS01_2026_1_15/  ← session_pattern
│   │   └── trial_summary.csv      ← behaviour_file
│   ├── SOUND_CAT_SS01_2026_1_16/
│   │   └── trial_summary.csv
│   └── ...
├── SS02/
│   └── ...
└── SS03/
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

#### Required columns

These four must be present in every config:

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
    default: "Uniform"
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

### analysis

Default parameters for analysis functions. All can be overridden per-call.

```yaml
analysis:
  excluded_stats: []              # Stats to skip, e.g. ["update_matrix"]
  hard_threshold: 0.3             # |stimulus| threshold for easy/hard split
  default_n_bins: 8               # Bins for psychometric/update matrix
  min_valid_trials: 10            # Sessions below this are dropped
  default_stage: "Full_Task_Cont" # Default stage filter
```

| Field | Default | Description |
|-------|---------|-------------|
| `excluded_stats` | `[]` | Stats to skip when computing feature matrices. Use if a stat is slow or irrelevant for your task. |
| `hard_threshold` | `0.3` | Absolute stimulus value below which trials are "hard" (near boundary). Affects `hard_accuracy`, `easy_accuracy`, `hard_easy_ratio`, and RT features. |
| `default_n_bins` | `8` | Number of bins for stimulus discretisation in psychometric fitting, update matrices, and binned stats. |
| `min_valid_trials` | `10` | Sessions with fewer valid (non-abort, responded) trials are skipped during feature matrix computation. |
| `default_stage` | `null` | If set, functions like `build_feature_matrix` and `experiment.plot_trajectory` use this stage filter by default. |

---

### plotting

Default plotting parameters.

```yaml
plotting:
  dpi: 100
  font_size: 10
  figure_width: 10.0
  colourmap: "tab10"
  model_colours:
    BE: "steelblue"
    SC: "darkorange"
    default: "grey"
```

These are read by `apply_style()` and plotting functions but can always be overridden per-call.

---

## Complete Examples

### Example 1: Visual orientation discrimination (different lab)

```yaml
project:
  name: "Orientation Discrimination"
  description: "2-AFC orientation task in freely moving rats"

file_structure:
  data_dir: "/data/orientation_task"
  behaviour_file: "behavioural_data*.csv"
  date_regex: "(?P<year>\\d{4})-(?P<month>\\d{2})-(?P<day>\\d{2})"

task:
  boundary: 45.0                        # 45 degrees
  stimulus_range: [0.0, 90.0]           # orientation in degrees
  category_rule: "above_boundary"

  choice_mapping:
    type: "identity"                    # choices already 0/1 in CSV
    no_response_value: -1

columns:
  trial_number:
    csv_name: "trial"
    dtype: int
  stimulus:
    csv_name: "orientation_deg"
    dtype: float
  choice:
    csv_name: "response"
    dtype: int
  outcome:
    csv_name: "outcome_str"
    dtype: str
  correct:
    csv_name: "is_correct"
    dtype: bool
  reaction_time:
    csv_name: "rt_ms"
    dtype: float
    optional: true

session_metadata:
  animal_id:
    csv_name: "rat_id"
    dtype: str
  stage:
    csv_name: "training_phase"
    dtype: str

analysis:
  default_stage: "full_task"
  hard_threshold: 10.0                  # degrees from boundary
```

### Example 2: Auditory go/no-go (minimal columns)

```yaml
project:
  name: "Auditory Go/No-Go"

file_structure:
  data_dir: "/data/auditory_gng"
  behaviour_file: "*.csv"
  date_regex: "(\\d{4})(\\d{2})(\\d{2})"  # compact date: 20260115

task:
  boundary: 8000.0                      # 8 kHz
  stimulus_range: [2000.0, 16000.0]     # frequency in Hz

  choice_mapping:
    type: "identity"
    no_response_value: -99

columns:
  trial_number:
    csv_name: "TrialNum"
    dtype: int
  stimulus:
    csv_name: "Freq_Hz"
    dtype: float
  choice:
    csv_name: "Licked"                  # 0 = no lick, 1 = lick
    dtype: int
  outcome:
    csv_name: "Result"
    dtype: str
  correct:
    csv_name: "Hit"
    dtype: bool

session_metadata:
  animal_id:
    csv_name: "Mouse"
    dtype: str

analysis:
  hard_threshold: 2000.0               # Hz from boundary
```

### Example 3: Existing dataset with unusual encoding

```yaml
project:
  name: "Legacy Dataset"

file_structure:
  data_dir: "/data/legacy"
  behaviour_file: "trials_*.csv"
  date_regex: "(?P<year>\\d{4})(?P<month>\\d{2})(?P<day>\\d{2})"

task:
  boundary: 0.0
  stimulus_range: [-1.0, 1.0]

  choice_mapping:
    type: "spatial_to_category"
    no_response_value: "NA"             # string, not number
    contingency_field: "mapping"
    contingency_rules:
      standard:
        L: 0                            # string values in CSV
        R: 1
      reversed:
        L: 1
        R: 0

columns:
  trial_number:
    csv_name: "TRIAL"
    dtype: int
  stimulus:
    csv_name: "STIM_VAL"
    dtype: float
  choice:
    csv_name: "RESP"
    dtype: str                          # strings in CSV, mapped by choice_mapping
  outcome:
    csv_name: "FEEDBACK"
    dtype: str
  correct:
    csv_name: "CORRECT_YN"
    dtype: str
    mapping:                            # "Y"/"N" → True/False
      Y: true
      N: false

session_metadata:
  animal_id:
    csv_name: "SUBJECT"
    dtype: str
  stage:
    csv_name: "PHASE"
    dtype: str
  mapping:
    csv_name: "RESP_MAPPING"
    dtype: str
```

---

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
