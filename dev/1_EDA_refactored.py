# %% [markdown]
# # Notebook 1 — EDA (LendingClub Default Prediction)
#
# ## Pre-flight checklist
# - Lock the **prediction moment** (origination/investor view) and define a **binary target** using final outcomes only.
# - Perform a **leakage audit**: remove post-origination variables and LendingClub’s proprietary risk outputs (`grade`, `sub_grade`) to avoid circular reasoning.
# - Quantify **missingness + data quality issues** (invalid %s, impossible DTIs, extreme outliers) and decide simple fixes for Notebook 2.
# - Check whether key relationships are **monotonic** (supports Logistic Regression baseline) and whether feature effects are strong enough to matter.
# - Verify **time drift / censoring** so the modeling split is realistic and not accidentally biased.
#
# **Rule:** If a plot/table doesn’t change a decision, it doesn’t belong here.

# %%
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

RANDOM_STATE = 42

# %% [markdown]
# ## 0) Load data + (optional) data dictionary
# We expect the full dataset to be present locally (already downloaded).
# This notebook is written to work from either the repo root or a `dev/` subfolder.

# %%
# --- Path resolution: works from both root and dev/ directories ---
HERE = Path.cwd()
PROJECT_ROOT = HERE if (HERE / "data").exists() else HERE.parent
DATA_DIR = PROJECT_ROOT / "data"

print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)

DICT_PATH = Path("/mnt/data/Lending Club Data Dictionary Approved.csv")  # optional

def find_dataset_file(data_dir: Path) -> Path:
    candidates = [
        "accepted_2007_to_2018Q4.csv",
        "accepted_2007_to_2018Q4.csv.gz",
        "loan.csv",
        "loans.csv",
        "lending_club_loans.csv",
    ]
    for name in candidates:
        p = data_dir / name
        if p.exists():
            return p
    csvs = sorted(data_dir.glob("*.csv")) + sorted(data_dir.glob("*.csv.gz"))
    if not csvs:
        raise FileNotFoundError(
            f"No dataset file found in {data_dir.resolve()}. "
            f"Put the full LendingClub CSV there or update DATA_DIR."
        )
    return csvs[0]

DATA_PATH = find_dataset_file(DATA_DIR)
DATA_PATH

# %%
# Load dataset (gzip supported)
if str(DATA_PATH).endswith(".gz"):
    df_raw = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)
else:
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)

print("Loaded:", DATA_PATH.name)
print("Shape:", df_raw.shape)
df_raw.head(3)

# %%
# Optional: load data dictionary to map descriptions
col_desc = {}
if DICT_PATH.exists():
    dd = pd.read_csv(DICT_PATH, encoding="latin1")
    possible_field_cols = [c for c in dd.columns if "field" in c.lower() or "name" in c.lower()]
    possible_desc_cols = [c for c in dd.columns if "desc" in c.lower()]
    if possible_field_cols and possible_desc_cols:
        field_col = possible_field_cols[0]
        desc_col = possible_desc_cols[0]
        col_desc = dict(zip(dd[field_col].astype(str), dd[desc_col].astype(str)))
        print(f"Loaded dictionary: {DICT_PATH.name} (field='{field_col}', desc='{desc_col}')")
    else:
        print("Dictionary loaded but column names not recognized; skipping mapping.")
else:
    print("Dictionary file not found; continuing without it.")

# %%
# Quick schema snapshot (top missingness only; full table is too big)
schema = pd.DataFrame({
    "column": df_raw.columns,
    "dtype": [str(t) for t in df_raw.dtypes],
    "missing_pct": df_raw.isna().mean().values * 100,
    "n_unique": [df_raw[c].nunique(dropna=True) for c in df_raw.columns],
})
schema["description"] = schema["column"].map(col_desc).fillna("")
schema.sort_values("missing_pct", ascending=False).head(20)

# %% [markdown]
# ## 1) Define the target (final outcomes only)
# **Decision (baseline and defensible):** Predict whether a loan ends in default **eventually**:
# - `0 = Fully Paid`
# - `1 = Charged Off` (and `Default` if present)
# - Drop `Current` and other non-final statuses (they are not outcomes).
#
# **What this implies:** Recent vintages can suffer **right-censoring** (many loans haven’t had time to finish). We will explicitly check that later.

# %%
status_counts = df_raw["loan_status"].value_counts(dropna=False)
status_counts

# %%
FINAL_GOOD = {"Fully Paid"}
FINAL_BAD = {"Charged Off", "Default"}

df = df_raw.copy()
df["target_default"] = np.where(df["loan_status"].isin(FINAL_BAD), 1,
                                np.where(df["loan_status"].isin(FINAL_GOOD), 0, np.nan))

df_final = df[df["target_default"].notna()].copy()

print("Final-outcome subset shape:", df_final.shape)
print("Default rate:", df_final["target_default"].mean())

# %% [markdown]
# ### Observed (from this run)
# - Full dataset: **2,260,701 rows × 151 columns**
# - Final-outcome subset: **1,345,350 rows**
# - Default rate among final outcomes: **~19.96%**
#
# ✅ **Validation:** Target is binary and non-final statuses are excluded.  
# ➡️ **Next:** remove leakage before doing any “insightful” EDA.

# %% [markdown]
# ## 2) Leakage audit
# We separate leakage into two categories:
#
# ### A) Post-origination / outcome-driven variables (hard leakage)
# These include payment history, recoveries, collections, hardship and settlement fields.
#
# ### B) LendingClub proprietary risk outputs (circular reasoning)
# You explicitly requested this: **`grade` and `sub_grade` are treated as leakage** because they are LendingClub’s origination-time risk assessment.
# Keeping them would largely “learn LC’s model,” not default risk from raw signals.
#
# We *will still analyze* grade/sub-grade briefly to demonstrate why they must be excluded.

# %%
# --- A) True leakage / post-origination fields (explicit list, no broad regex) ---
POST_ORIGINATION_LEAKAGE = {
    # payment / recovery / collections
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "out_prncp", "out_prncp_inv", "last_credit_pull_d",
    # hardship / settlement / debt settlement programs
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status", "hardship_amount",
    "hardship_start_date", "hardship_end_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "hardship_last_payment_amount", "hardship_payoff_balance_amount",
    "settlement_status", "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
    "debt_settlement_flag", "debt_settlement_flag_date",
    # bookkeeping
    "loan_status", "pymnt_plan",
}

# --- B) LC proprietary risk outputs (circular reasoning) ---
LC_RISK_OUTPUTS = {"grade", "sub_grade"}

# Some identifiers / free-text fields (not leakage, but not useful for baseline)
ID_TEXT_DROPS = {"id", "member_id", "url", "emp_title", "title", "zip_code", "desc"}

# Keep issue_d for splitting and drift checks, but not necessarily as a model feature
SPLIT_COLS = {"issue_d"}

# %%
# Candidate columns BEFORE dropping LC grade/subgrade (so we can analyze them)
base_drop = POST_ORIGINATION_LEAKAGE | ID_TEXT_DROPS
candidate_all = [c for c in df_final.columns if c not in base_drop and c != "target_default"]

print("Candidate columns (including grade/sub_grade if present):", len(candidate_all))
[c for c in ["grade", "sub_grade", "int_rate", "issue_d"] if c in candidate_all]

# %%
# Build working frame for EDA: keep target and candidate columns
df_work = df_final[candidate_all + ["target_default"]].copy()

# %% [markdown]
# ## 3) Parse key fields + basic sanity checks
# We parse month-year dates robustly (`%b-%y` and `%b-%Y`) and convert `%` strings to numeric.
# We also derive `credit_history_years` from `issue_d` and `earliest_cr_line`.

# %%
def parse_month_year(series: pd.Series) -> pd.Series:
    """Parse month-year in either %b-%y (Dec-15) or %b-%Y (Dec-2015) format."""
    s = series.astype(str).str.strip()
    dt1 = pd.to_datetime(s, format="%b-%y", errors="coerce")
    dt2 = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    return dt1.fillna(dt2)

def parse_percent(series: pd.Series) -> pd.Series:
    """Convert percent strings like '13.56%' to float."""
    s = series.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def parse_term(series: pd.Series) -> pd.Series:
    """Extract numeric months from strings like '36 months'."""
    return series.astype(str).str.extract(r"(\d+)").astype(float)[0]

EMP_MAP = {
    "10+ years": 10, "9 years": 9, "8 years": 8, "7 years": 7, "6 years": 6, "5 years": 5,
    "4 years": 4, "3 years": 3, "2 years": 2, "1 year": 1, "< 1 year": 0,
}
def parse_emp_length(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.map(EMP_MAP).astype(float)

# Apply parsing where columns exist
if "issue_d" in df_work.columns:
    df_work["issue_d"] = parse_month_year(df_work["issue_d"])
if "earliest_cr_line" in df_work.columns:
    df_work["earliest_cr_line"] = parse_month_year(df_work["earliest_cr_line"])
for pct_col in ["int_rate", "revol_util"]:
    if pct_col in df_work.columns:
        df_work[pct_col] = parse_percent(df_work[pct_col])
if "term" in df_work.columns:
    df_work["term"] = parse_term(df_work["term"])
if "emp_length" in df_work.columns:
    df_work["emp_length"] = parse_emp_length(df_work["emp_length"])

if "issue_d" in df_work.columns and "earliest_cr_line" in df_work.columns:
    df_work["credit_history_years"] = (df_work["issue_d"] - df_work["earliest_cr_line"]).dt.days / 365.25

# %%
# Sanity check
if "issue_d" in df_work.columns:
    print("issue_d parsed NaT %:", df_work["issue_d"].isna().mean())
    print("issue_d min/max:", df_work["issue_d"].min(), df_work["issue_d"].max())
if "credit_history_years" in df_work.columns:
    print(df_work["credit_history_years"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

# %% [markdown]
# ## 4) Data quality issues (EDA → decisions, not permanent transforms)
# From the run outputs:
# - `annual_inc` is heavily right-skewed with extreme outliers.
# - `dti` contains impossible values >100.
# - `revol_util` contains invalid values >100.
# - Several count features are zero-inflated with rare extreme values.
#
# **Decision:** In Notebook 2 we will apply:
# - winsorization/capping (learned on train only) for extreme outliers
# - validity rules: set invalid percentages to missing, cap impossible DTIs
#
# For EDA visuals we create a **capped view** to avoid plots being dominated by nonsense.

# %%
df_vis = df_work.copy()

fix_log = []
def cap_at_p99(col: str):
    p99 = df_vis[col].quantile(0.99)
    n_over = (df_vis[col] > p99).sum()
    fix_log.append((col, "cap_p99", float(p99), int(n_over)))
    df_vis[col] = df_vis[col].clip(upper=p99)

# annual_inc: cap at p99 for visualization
if "annual_inc" in df_vis.columns:
    cap_at_p99("annual_inc")

# dti: cap at 100 (impossible beyond 100 for this context)
if "dti" in df_vis.columns:
    n_over = (df_vis["dti"] > 100).sum()
    fix_log.append(("dti", "cap_100", 100.0, int(n_over)))
    df_vis["dti"] = df_vis["dti"].clip(upper=100)

# revol_util: invalid >100 → set missing
if "revol_util" in df_vis.columns:
    n_over = (df_vis["revol_util"] > 100).sum()
    fix_log.append(("revol_util", "set_nan_if_gt_100", 100.0, int(n_over)))
    df_vis.loc[df_vis["revol_util"] > 100, "revol_util"] = np.nan

# revol_bal: cap at p99 for visualization
if "revol_bal" in df_vis.columns:
    cap_at_p99("revol_bal")

# count-like features: cap at p99 for visualization
for col in [c for c in ["inq_last_6mths", "delinq_2yrs", "pub_rec", "tax_liens"] if c in df_vis.columns]:
    cap_at_p99(col)

fix_log_df = pd.DataFrame(fix_log, columns=["feature", "action", "threshold", "n_affected"])
fix_log_df

# %% [markdown]
# ### Observed (from this run)
# - `annual_inc`: **13,448** values above p99 (**$250,000**)  
# - `dti`: **533** values > 100 (capped to 100)  
# - `revol_util`: **4,687** values > 100 (set to missing)  
# - `revol_bal`: **13,454** values above p99 (≈ **94,554**)  
# - `inq_last_6mths`: p99 = 4 (5,899 capped)  
# - `delinq_2yrs`: p99 = 4 (10,043 capped)  
# - `pub_rec`: p99 = 2 (12,662 capped)  
# - `tax_liens`: p99 = 2 (5,719 capped)
#
# ✅ **Validation:** We have specific, defensible data-quality rules to carry into Notebook 2.  
# ➡️ **Next:** missingness decisions.

# %% [markdown]
# ## 5) Missingness analysis (drives feature inclusion)
# **Rule:** Ultra-sparse features across 2007–2018 are not baseline-friendly.
# We use a simple threshold: drop columns with >70% missingness.

# %%
missing = df_work.isna().mean().sort_values(ascending=False) * 100
missing.head(25)

# %%
# Plot top 30 missingness columns
top_missing = missing.head(30).sort_values()
plt.figure(figsize=(12, 6))
top_missing.plot(kind="barh")
plt.title("Top 30 columns by missingness (%)")
plt.xlabel("Missing %")
plt.tight_layout()
plt.show()

# %%
DROP_MISSING_THRESHOLD = 70.0
cols_drop_missing = set(missing[missing > DROP_MISSING_THRESHOLD].index)
print(f"Columns > {DROP_MISSING_THRESHOLD:.0f}% missing:", len(cols_drop_missing))
sorted(list(cols_drop_missing))[:30]

# %%
# Missingness pattern summary (based on this run)
print("\nMissingness findings:")
print("1) Joint/secondary applicant fields (sec_app_*, *_joint): ~98–100% missing → drop for baseline")
print("2) Sparse operational fields (payment_plan_*, orig_projected_*): ~99% missing → drop")
print("3) Text field 'desc': ~90% missing → drop")
print("4) Recency variables (mths_since_*): often 60–80% missing → keep only if <70% and impute later")

# %% [markdown]
# ### Observed (from this run)
# - **21 columns** exceed the 70% missingness threshold.
# - These are dominated by:
#   - joint/secondary applicant fields (`sec_app_*`, `*_joint`)
#   - ultra-sparse operational fields (`payment_plan_start_date`, `orig_projected_additional_accrued_interest`)
#   - sparse free text (`desc`)
#
# ✅ **Validation:** Missingness threshold yields a clean, consistent baseline feature set.  
# ➡️ **Next:** relationship EDA to decide the modeling approach.

# %% [markdown]
# ## 6) Leakage from LendingClub grade (circular reasoning)
# You requested grade/sub-grade be treated as leakage. This run confirms why:
# grade has a near-monotonic relationship with default rate.
#
# **Decision:** We keep `grade`/`sub_grade` for demonstration only, but exclude them from any baseline model.

# %%
if "grade" in df_work.columns:
    grade_grp = df_work.groupby("grade")["target_default"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    display(grade_grp)

# %% [markdown]
# ### Observed (from this run)
# Default rate by grade is extreme:
# - **G ~49.9%**, **F ~45.2%**, **E ~38.5%**, ... down to **A ~6.0%**
#
# This is exactly what an internal risk score is supposed to do — which is why using it in our model would be circular.
#
# ✅ **Validation:** Excluding grade/sub-grade is justified empirically.  
# ➡️ **Next:** build the final EDA dataset **without** LC risk outputs.

# %%
# Final EDA dataset for modeling decisions (exclude LC grade/subgrade + ultra-sparse cols)
leakage_like = LC_RISK_OUTPUTS  # extend later if needed
drop_for_modeling = cols_drop_missing | leakage_like

df_eda = df_work.drop(columns=[c for c in drop_for_modeling if c in df_work.columns]).copy()
print("EDA dataset shape (post-missingness + no grade/subgrade):", df_eda.shape)

# %% [markdown]
# ## 7) Univariate distributions (focused)
# We focus on a compact core set of numeric features that are widely used, interpretable, and available at origination.

# %%
core_numeric = [
    c for c in [
        "loan_amnt", "funded_amnt", "funded_amnt_inv",
        "annual_inc", "dti", "fico_range_low", "fico_range_high",
        "revol_util", "revol_bal", "open_acc", "total_acc",
        "inq_last_6mths", "delinq_2yrs", "pub_rec", "tax_liens",
        "int_rate", "installment", "credit_history_years"
    ] if c in df_vis.columns  # use capped view for visuals
]

# 4x3 grid for compactness
n = len(core_numeric)
ncols = 4
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 4 * nrows))
axes = np.array(axes).reshape(-1)

for ax, col in zip(axes, core_numeric):
    s = df_vis[col].dropna()
    ax.hist(s, bins=50)
    ax.set_title(col)
    ax.tick_params(axis="x", labelrotation=45)

# hide unused axes
for ax in axes[len(core_numeric):]:
    ax.axis("off")

plt.suptitle("Univariate Distributions — Core Numeric Features (capped view)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observed (from this run)
# - Strong skew/outliers in `annual_inc` and `revol_bal` (handled via capping for visualization).
# - `dti` and `revol_util` include invalid/extreme values (fixed for visuals; will be handled properly in the model pipeline).
# - `loan_amnt` and `funded_amnt` are nearly redundant.
# - `installment` is structurally tied to `loan_amnt`, `term`, and `int_rate` (multicollinearity expected).
#
# ✅ **Validation:** The data behaves like a typical credit dataset and supports simple preprocessing.  
# ➡️ **Next:** bivariate EDA to see if effects are monotonic (supports Logistic Regression baseline).

# %% [markdown]
# ## 8) Bivariate EDA: default rate by quantile bins
# These plots test whether risk changes smoothly with features.
# Monotonic patterns → Logistic Regression is a strong baseline.

# %%
def default_rate_by_bin(df_in: pd.DataFrame, feature: str, n_bins: int = 10) -> pd.DataFrame:
    d = df_in[[feature, "target_default"]].dropna()
    d["bin"] = pd.qcut(d[feature], q=n_bins, duplicates="drop")
    grp = d.groupby("bin")["target_default"].agg(["mean", "count"]).reset_index()
    grp.rename(columns={"mean": "default_rate", "count": "n"}, inplace=True)
    return grp

features_for_bins = [c for c in ["fico_range_low", "int_rate", "dti", "revol_util", "annual_inc", "credit_history_years"] if c in df_vis.columns]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))
axes = axes.flatten()

for ax, feat in zip(axes, features_for_bins):
    grp = default_rate_by_bin(df_vis, feat, n_bins=10)
    ax.plot(range(len(grp)), grp["default_rate"].values, marker="o")
    ax.set_title(feat)
    ax.set_xlabel("Bin (low → high)")
    ax.set_ylabel("Default rate")

plt.suptitle("Default Rate by Feature (Quantile Bins)")
plt.tight_layout()
plt.show()

# Print the bin tables (useful for the markdown summary)
for feat in features_for_bins:
    print(f"\n{feat}:")
    display(default_rate_by_bin(df_vis, feat, n_bins=10))

# %% [markdown]
# ### Observed (from this run)
# Strong, mostly monotonic relationships:
# - `fico_range_low`: default rate drops **~26.3% → ~9.3%** (strong negative signal)
# - `int_rate`: default rate rises **~4.9% → ~40.2%** (very strong positive signal; investor-visible but reflects LC pricing)
# - `dti`: default rate rises **~14.7% → ~29.0%**
# - `revol_util`: default rate rises **~14.8% → ~22.8%**
# - `annual_inc`: default rate drops **~24.1% → ~15.1%**
# - `credit_history_years`: modest drop **~22.3% → ~17.8%**
#
# **Implication:** A simple additive model (Logistic Regression) should be competitive.
# Tree models may still help a bit, but the dominant patterns are monotonic and smooth.
#
# ✅ **Validation:** EDA supports starting with Logistic Regression + regularization.  
# ➡️ **Next:** categorical effects and time drift.

# %% [markdown]
# ## 9) Categorical EDA: default rate by category
# We inspect a few low-cardinality categoricals and group rare levels.

# %%
core_cat = [c for c in ["term", "home_ownership", "verification_status", "purpose", "addr_state", "emp_length"] if c in df_vis.columns]

def plot_default_rate_by_category(df_in: pd.DataFrame, col: str, top_n: int = 12):
    d = df_in[[col, "target_default"]].copy()
    vc = d[col].value_counts(dropna=False)
    keep = vc.head(top_n).index
    d[col] = d[col].where(d[col].isin(keep), other="Other")
    grp = d.groupby(col)["target_default"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    return grp

# Print summaries
for c in core_cat:
    print(f"\n{c}:")
    display(plot_default_rate_by_category(df_vis, c, top_n=12))

# Plot in a grid
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
axes = axes.flatten()

for ax, c in zip(axes, core_cat):
    grp = plot_default_rate_by_category(df_vis, c, top_n=12)
    grp["mean"].plot(kind="bar", ax=ax)
    ax.set_title(c)
    ax.set_ylabel("Default rate")
    ax.tick_params(axis="x", labelrotation=45)

for ax in axes[len(core_cat):]:
    ax.axis("off")

plt.suptitle("Default Rate by Category (Top levels + Other)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observed (from this run)
# - `term` is a strong driver: **60 months ~32.4%** vs **36 months ~16.0%** default rate.
# - Other categoricals show smaller but meaningful shifts (purpose, verification status, home ownership, etc.).
#
# ✅ **Validation:** Categorical effects exist and are easy to handle with OneHotEncoding later.  
# ➡️ **Next:** time drift + censoring check.

# %% [markdown]
# ## 10) Time drift + censoring (critical for split strategy)
# We plot default rate over issue date (quarterly) **using final outcomes only**.
#
# **Important:** Very recent vintages can look artificially “low default” because many loans have not had time to default and are removed as `Current`.
# This is right-censoring and can bias time-based evaluation if you naïvely use 2018 as a test set for “eventual default”.

# %%
if "issue_d" in df_vis.columns:
    tmp = df_vis[["issue_d", "target_default"]].dropna()
    tmp["issue_q"] = tmp["issue_d"].dt.to_period("Q").astype(str)

    rate_by_q = tmp.groupby("issue_q")["target_default"].agg(["mean", "count"]).reset_index()
    # show tail where censoring is worst
    tail = rate_by_q.tail(12)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    axes[0].plot(rate_by_q["issue_q"], rate_by_q["mean"], marker="o")
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].set_title("Default rate over time (quarterly)")
    axes[0].set_ylabel("Default rate")

    axes[1].plot(rate_by_q["issue_q"], rate_by_q["count"], marker="o")
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[1].set_title("Final-outcome sample size over time")
    axes[1].set_ylabel("Count (final outcomes only)")

    plt.tight_layout()
    plt.show()

    print("Default rate (tail 12 quarters):")
    display(tail)

# %%
# Feature drift (median by year) for a few key variables
if "issue_d" in df_vis.columns:
    df_drift = df_vis.copy()
    df_drift["issue_year"] = df_drift["issue_d"].dt.year

    drift_features = [c for c in ["fico_range_low", "int_rate", "dti", "annual_inc"] if c in df_drift.columns]
    drift_summary = df_drift.groupby("issue_year")[drift_features].median().dropna()

    display(drift_summary.tail(10))

    plt.figure(figsize=(10, 4))
    drift_summary.plot()
    plt.title("Median feature drift by year")
    plt.tight_layout()
    plt.show()


# %%
# OPTIONAL sanity check: how much of each vintage is still "Current"?
# This exposes right-censoring directly (especially for recent years and 60-month loans).
if "issue_d" in df_raw.columns and "loan_status" in df_raw.columns:
    tmp_all = df_raw[["issue_d", "loan_status"]].copy()
    tmp_all["issue_d"] = parse_month_year(tmp_all["issue_d"])
    tmp_all = tmp_all.dropna(subset=["issue_d"])
    tmp_all["issue_q"] = tmp_all["issue_d"].dt.to_period("Q").astype(str)

    status_q = (
        tmp_all.assign(is_current=(tmp_all["loan_status"] == "Current").astype(int))
        .groupby("issue_q")["is_current"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "pct_current", "count": "n_loans"})
    )

    display(status_q.tail(12))


# %% [markdown]
# ### Observed (from this run)
# - Default rate increases substantially around **2016–2017**, then appears to collapse in **2018**.
# - The **final-outcome sample size collapses** in 2018 (e.g., 2018Q4 has only ~5k final outcomes), consistent with **right-censoring**.
# - Feature medians drift meaningfully over years (e.g., FICO, DTI, income, rates shift).
#
# **Decision for Notebook 2:** Use a **time-based split** but avoid using 2018 as the “eventual default” test set.
# A clean, simple split is:
# - Train: 2007–2014
# - Validation: 2015
# - Test: 2016–2017
#
# ✅ **Validation:** Drift exists and censoring is visible, so random train_test_split is not acceptable.  
# ➡️ **Next:** multicollinearity + a concrete feature shortlist.

# %% [markdown]
# ## 11) Multicollinearity scan (practical)
# We use VIF on a compact numeric set to identify redundancy.
# Very high VIF → near-duplicate info.

# %%
from sklearn.linear_model import LinearRegression

def compute_vif(df_in: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df_in[features].dropna()
    X = X.loc[:, X.nunique() > 1]
    feats = list(X.columns)
    vifs = []
    for f in feats:
        y = X[f].values
        X_other = X.drop(columns=[f]).values
        if X_other.shape[1] == 0:
            vifs.append((f, np.nan))
            continue
        model = LinearRegression()
        model.fit(X_other, y)
        r2 = model.score(X_other, y)
        vif = 1.0 / (1.0 - r2 + 1e-12)
        vifs.append((f, vif))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

vif_features = [c for c in ["loan_amnt", "funded_amnt", "funded_amnt_inv", "installment", "int_rate",
                            "annual_inc", "dti", "fico_range_low", "revol_util",
                            "revol_bal", "open_acc", "total_acc", "inq_last_6mths", "delinq_2yrs",
                            "credit_history_years"] if c in df_vis.columns]

if len(vif_features) >= 3:
    vif_table = compute_vif(df_vis, vif_features)
    display(vif_table)

# %% [markdown]
# ### Observed (from this run)
# - `loan_amnt` and `funded_amnt` have **VIF > 1100** → near duplicates.
# - `installment` has **VIF ~11.5** (as expected; derived from amount/term/rate).
#
# **Decision:** For a clean baseline in Notebook 2:
# - keep `loan_amnt`
# - drop `funded_amnt` and `funded_amnt_inv` (redundant)
# - treat `installment` as optional and likely exclude in the first baseline; revisit only in ablation with `int_rate`.
#
# ✅ **Validation:** We have concrete redundancy evidence and a simplification plan.  
# ➡️ **Next:** lock feature sets and export artifacts.

# %% [markdown]
# ## 12) EDA decisions register → inputs for Notebook 2
# **Modeling feature sets (explicit):**
#
# ### A) Applicant/origination baseline (no LC pricing outputs)
# Excludes `grade`, `sub_grade`, and `int_rate` (plus derived `installment` by default).
#
# ### B) Investor-view model (includes LC’s published pricing)
# Includes `int_rate` (and optionally `installment`), but we will run an **ablation** to quantify how much signal comes from pricing.
#
# This gives you honesty + a clean story.

# %%
# Keep issue_d for splitting but do NOT include it as a model feature by default.
BASE_FEATURES = [
    # numeric
    "loan_amnt", "term", "annual_inc", "dti", "fico_range_low",
    "revol_util", "revol_bal", "open_acc", "total_acc",
    "inq_last_6mths", "delinq_2yrs", "pub_rec", "tax_liens",
    "credit_history_years",
    # categorical
    "emp_length", "home_ownership", "verification_status", "purpose", "addr_state",
]

INVESTOR_FEATURES = BASE_FEATURES + ["int_rate"]  # keep simple; installment will be tested in ablation if desired

BASE_FEATURES = [c for c in BASE_FEATURES if c in df_eda.columns]
INVESTOR_FEATURES = [c for c in INVESTOR_FEATURES if c in df_eda.columns]

print("Base features:", len(BASE_FEATURES), BASE_FEATURES)
print("Investor features:", len(INVESTOR_FEATURES), INVESTOR_FEATURES)

# %% [markdown]
# ## 13) Export EDA artifacts
# We export:
# - `eda_dataset.csv`: dataset containing `issue_d`, `target_default`, and all candidate features post-missingness and post-leakage rules
# - `eda_artifacts.json`: the decisions and lists (dropped columns, feature lists, thresholds)
#
# Notebook 2 will load these artifacts and build the modeling pipeline cleanly.

# %%
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

artifacts = {
    "data_path": str(DATA_PATH),
    "target_definition": {"good": list(FINAL_GOOD), "bad": list(FINAL_BAD)},
    "default_rate_final_outcomes": float(df_final["target_default"].mean()),
    "drop_missing_threshold": DROP_MISSING_THRESHOLD,
    "dropped_ultra_sparse_cols": sorted(list(cols_drop_missing)),
    "post_origination_leakage_cols": sorted(list(POST_ORIGINATION_LEAKAGE)),
    "lc_risk_outputs_excluded": sorted(list(LC_RISK_OUTPUTS)),
    "id_text_drops": sorted(list(ID_TEXT_DROPS)),
    "base_features": BASE_FEATURES,
    "investor_features": INVESTOR_FEATURES,
    "split_col": "issue_d",
    "suggested_time_split": {"train_end": "2014-12-01", "val_year": 2015, "test_years": [2016, 2017]},
}

with open(OUTPUT_DIR / "eda_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

# Save a compact EDA dataset. We keep issue_d for splitting later.
save_cols = sorted(set(df_eda.columns) | {"issue_d", "target_default"})
save_cols = [c for c in save_cols if c in df_work.columns] + ["target_default"]
save_cols = list(dict.fromkeys(save_cols))  # preserve order, remove dupes

df_export = df_work[save_cols].copy()
df_export.to_csv(OUTPUT_DIR / "eda_dataset.csv", index=False)

print("Saved:")
print("-", (OUTPUT_DIR / "eda_artifacts.json").resolve())
print("-", (OUTPUT_DIR / "eda_dataset.csv").resolve())

# %% [markdown]
# ## Validation + next steps
# ✅ **Validation:** EDA shows strong monotonic drivers of default (FICO↓, DTI↑, utilization↑, income↓) and exposes major pitfalls (LC grade leakage, right-censoring in 2018).
#
# **Next steps (Notebook 2):**
# 1) Use the suggested time split (train 2007–2014, val 2015, test 2016–2017).
# 2) Baseline model: Logistic Regression (regularized) with simple preprocessing (impute + one-hot).
# 3) Compare against one tree model (DecisionTree or RandomForest) only as a sanity check.
# 4) Run ablation: with vs without `int_rate` (and optionally `installment`) to quantify reliance on LC pricing signal.
