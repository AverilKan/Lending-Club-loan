# %% [markdown]
# # Notebook 1 — EDA (LendingClub Default Prediction)
#
# ## Pre-flight checklist
# - Lock the **prediction moment** (origination/investor view) and define a **binary target** using final outcomes only.
# - Perform a **leakage audit**: remove post-origination variables and LendingClub's proprietary risk outputs (`grade`, `sub_grade`) to avoid circular reasoning.
# - Quantify **missingness + data quality issues** (invalid %s, impossible DTIs, extreme outliers) and decide simple fixes for Notebook 2.
# - Check whether key relationships are **monotonic** (supports Logistic Regression baseline) and whether feature effects are strong enough to matter.
# - Verify **time drift / censoring** so the modeling split is realistic and not accidentally biased.
#
# **Rule:** If a plot/table doesn't change a decision, it doesn't belong here.

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
# The full dataset should be available locally (already downloaded).
# This notebook is designed to work from either the repo root or a `dev/` subfolder.

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
# **What this implies:** Recent vintages can suffer **right-censoring** (many loans haven't had time to finish). This will be verified in Section 10 (time drift analysis).

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
# ➡️ **Next:** remove leakage before doing any "insightful" EDA.

# %% [markdown]
# ## 2) Leakage audit
# Leakage is separated into two categories:
#
# ### A) Post-origination / outcome-driven variables (hard leakage)
# These include payment history, recoveries, collections, hardship and settlement fields.
#
# ### B) LendingClub proprietary risk outputs (circular reasoning)
# This analysis treats **`grade` and `sub_grade` as leakage** because they represent LendingClub's origination-time risk assessment.
# Keeping them would largely "learn LC's model," not default risk from raw signals.
#
# The analysis includes grade/sub_grade for demonstration purposes to show why they must be excluded.

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
    # servicing / operational (post-origination)
    "payment_plan_start_date", "deferral_term", "orig_projected_additional_accrued_interest",
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
# **Purpose:** Parse month-year dates robustly and convert percentage strings to numeric values.
#
# This section handles date parsing (supporting both `%b-%y` and `%b-%Y` formats),
# percentage string conversion, and derives `credit_history_years` from the time between `issue_d` and `earliest_cr_line`.

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
# **Critical distinction:** This section creates a **visualization-only capped view** (`df_vis`) to avoid plots being dominated by outliers.
# The **actual** data quality transformations will be applied in Notebook 2:
# - Winsorization/capping thresholds learned **on training data only**
# - Validity rules: set invalid percentages to missing, cap impossible DTIs
#
# This approach prevents test set leakage via preprocessing.

# %%
# Create visualization-only view with capping (NOT applied to modeling data)
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
# ✅ **Validation:** Specific, defensible data-quality rules have been identified for use in Notebook 2.
# ➡️ **Next:** missingness decisions.

# %% [markdown]
# ## 5) Missingness analysis (drives feature inclusion)
# **Purpose:** Identify columns that are too sparse across the 2007-2018 timespan and establish a simple handling strategy.
#
# A simple threshold of 70% missingness is used to identify ultra-sparse columns.
# This approach balances the need for feature completeness against the reality that many operational fields
# were not consistently recorded or are artifacts of data collection changes across the 12-year span.

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
print("2) Post-origination leakage (payment_plan_*, deferral_*, orig_projected_*): explicitly excluded as servicing variables")
print("3) Text field 'desc': ~90% missing → drop")
print("4) Recency variables (mths_since_*): often 60–80% missing → keep only if <70% and impute later")

# %% [markdown]
# ### Findings
# **Ultra-sparse columns identified (>70% missing):**
# - **21 columns** exceed the 70% missingness threshold
# - Joint/secondary applicant fields (`sec_app_*`, `*_joint`): Not consistently available across 2007-2018
# - Ultra-sparse operational fields (`payment_plan_start_date`, `orig_projected_additional_accrued_interest`): Only recorded when events occurred (biased representation)
# - Sparse free text (`desc`): ~90% missing and difficult to encode for sklearn-based modeling
#
# **Reasoning:** These ultra-sparse fields are not baseline-friendly because:
# - Imputation becomes unreliable with <30% observed data
# - Missing values often indicate "no event occurred" rather than "data not recorded", introducing bias
# - Joint applicant fields are artifacts of data collection changes over time (not available for pre-2009 loans)
#
# ✅ **Validation:** The 70% threshold yields a clean, consistent baseline feature set with 87 candidate features.
# ➡️ **Next:** Feature relationship analysis to assess modeling approach.

# %% [markdown]
# ## 5a) Missingness-as-signal check (justifies imputation strategy)
# **Purpose:** Check if missingness itself is predictive for moderate-missing features.
#
# If missingness is strongly predictive (e.g., "no recent delinquency" vs "delinquency data missing"),
# we should add missing indicators instead of simple median imputation.

# %%
# Identify moderate-missing features (30-70% missing)
moderate_missing = missing[(missing > 30) & (missing <= 70)].index.tolist()
moderate_missing = [c for c in moderate_missing if c in df_work.columns and c != "target_default"]

missingness_signal = []
for col in moderate_missing[:10]:  # Check top 10 for brevity
    df_tmp = df_work[[col, "target_default"]].copy()
    df_tmp["is_missing"] = df_tmp[col].isna().astype(int)

    grp = df_tmp.groupby("is_missing")["target_default"].agg(["mean", "count"])
    if len(grp) == 2:
        dr_present = grp.loc[0, "mean"]
        dr_missing = grp.loc[1, "mean"]
        diff = abs(dr_missing - dr_present)
        missingness_signal.append({
            "feature": col,
            "dr_present": dr_present,
            "dr_missing": dr_missing,
            "abs_diff": diff,
            "n_missing": grp.loc[1, "count"]
        })

signal_df = pd.DataFrame(missingness_signal).sort_values("abs_diff", ascending=False)
print(signal_df.to_string())

# %% [markdown]
# ### Findings
# **Missingness signal check results:**
# - Most moderate-missing features show **<5% absolute difference** in default rate between missing vs. non-missing
# - This suggests missingness is **not strongly predictive** for most features
# - **Decision:** Use simple median/mode imputation for baseline model; missing indicators not necessary
# - **Exception:** If any feature shows >10% difference, consider adding missing indicator for that feature in Notebook 2

# %% [markdown]
# ## 6) Leakage from LendingClub grade (circular reasoning)
# **Purpose:** Verify that LendingClub's grade field encodes default risk and confirm it should be excluded.
#
# The grade/sub_grade fields are treated as leakage (outcome encoding) because they are LC's proprietary risk scores,
# derived from their internal assessment process. Using LC's own risk output to predict defaults would be circular reasoning.
# This analysis includes grade for demonstration only, but excludes it from any baseline model.

# %%
if "grade" in df_work.columns:
    grade_grp = df_work.groupby("grade")["target_default"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(grade_grp)

# %% [markdown]
# ### Findings
# **Grade perfectly encodes default risk (circular reasoning confirmed):**
# - Default rate varies dramatically by grade: **A ~6.0%** → **G ~49.9%** (8x difference)
# - Each grade step shows a clear monotonic increase in default rate (A<B<C<D<E<F<G)
#
# **Reasoning:** This is precisely what an internal risk scoring model is designed to do.
# LendingClub's grade is their proprietary assessment of risk, so using it to predict defaults is circular reasoning.
# It would not represent an independent investor's assessment and inflates performance artificially.
# The grade field is included in this analysis for demonstration only but explicitly excluded from baseline modeling.
#
# ✅ **Validation:** Excluding grade/sub-grade is justified empirically and methodologically.
# ➡️ **Next:** Build the final EDA dataset using only applicant fundamentals and published investor features (int_rate).

# %%
# Final EDA dataset for modeling decisions (exclude LC grade/subgrade + ultra-sparse cols)
leakage_like = LC_RISK_OUTPUTS  # extend later if needed
drop_for_modeling = cols_drop_missing | leakage_like

df_eda = df_work.drop(columns=[c for c in drop_for_modeling if c in df_work.columns]).copy()
print("EDA dataset shape (post-missingness + no grade/subgrade):", df_eda.shape)

# %% [markdown]
# ## 7) Univariate distributions (focused)
# **Purpose:** Examine the individual distributions of core numeric features to identify outliers and understand data quality.
#
# The univariate analysis focuses on a compact set of numeric features that are widely used, interpretable, and available at origination.
# This includes loan amount, interest rate, borrower income, debt ratio, credit score, and account metrics.

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
# ### Findings
# **Data quality issues confirmed (visualization-only capping applied):**
# - `annual_inc`: Heavily right-skewed with extreme outliers exceeding $250k (capped for visualization)
# - `revol_bal`: Similar right-skew with values >$94k (capped for visualization)
# - `dti`: Contains impossible values >100 (debt-to-income can't exceed 100%), indicating data entry errors
# - `revol_util`: Invalid values >100% (credit utilization capped at 100%), indicating reporting errors
# - `loan_amnt` and `funded_amnt`: Nearly perfectly redundant (99.96% correlated), requires dropping one in Notebook 2
# - `installment`: Structurally derived from loan amount, term, and rate, showing expected multicollinearity
#
# **Note on capping:** The visualization-only capped view (`df_vis`) prevents extreme outliers from dominating plots.
# Actual data quality transforms (winsorization/missing handling) will be applied in Notebook 2 using train-only thresholds.
#
# ✅ **Validation:** Data structure matches typical credit datasets; clear preprocessing strategy identified.
# ➡️ **Next:** Bivariate analysis to assess monotonic relationships (informs model selection between linear/nonlinear).

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
    print(default_rate_by_bin(df_vis, feat, n_bins=10).to_string())

# %% [markdown]
# ### Findings
# **Monotonic risk relationships (strong linear patterns observed):**
# - `fico_range_low`: Default rate decreases from **26.3% → 9.3%** across deciles (strong negative signal)
#   - **Reasoning:** Lower FICO scores indicate higher historical credit risk and payment stress
# - `int_rate`: Default rate increases from **4.9% → 40.2%** across deciles (very strong positive signal)
#   - **Reasoning:** LC's interest rate pricing encodes their internal risk assessment; higher rates indicate riskier borrowers
# - `dti`: Default rate increases from **14.7% → 29.0%** across deciles
#   - **Reasoning:** Higher debt-to-income ratios indicate borrower financial stress and reduced repayment capacity
# - `revol_util`: Default rate increases from **14.8% → 22.8%** across deciles
#   - **Reasoning:** Higher revolving credit utilization signals liquidity constraints and financial strain
# - `annual_inc`: Default rate decreases from **24.1% → 15.1%** across deciles (weaker but clear effect)
#   - **Reasoning:** Higher income provides greater repayment capacity, though effect is partially captured by DTI
# - `credit_history_years`: Modest decrease from **22.3% → 17.8%** across deciles
#   - **Reasoning:** Longer credit history indicates experience managing credit obligations
#
# **Modeling implication:** The predominantly monotonic, smooth relationships support starting with **Logistic Regression** as the baseline model.
# Non-linear models (decision trees, boosting) may provide marginal improvements for interaction capture, but the dominant signal is additive.
#
# ✅ **Validation:** EDA evidence supports regularized Logistic Regression baseline with strong statistical foundation.
# ➡️ **Next:** Categorical feature analysis and time-based drift validation.

# %% [markdown]
# ## 9) Categorical EDA: default rate by category
# **Purpose:** Assess whether categorical features (employment tenure, home ownership, loan purpose, state) show meaningful default rate variation.
#
# This section inspects low-cardinality categorical features and groups rare levels to avoid sparse categories with unreliable estimates.

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
    print(plot_default_rate_by_category(df_vis, c, top_n=12).to_string())

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
# ### Findings
# **Categorical effects on default risk (moderate predictive signal):**
# - `term`: Strongest categorical driver: **36 months ~16.0%** vs **60 months ~32.4%** default rate (2x difference)
#   - **Reasoning:** Longer terms extend repayment duration, increasing vulnerability to life events and economic shocks
# - `purpose`: Meaningful variation by loan purpose (debt consolidation, home improvement, personal loans show different default rates)
# - `verification_status`: Verification level shows modest differentiation in default risk
# - `home_ownership`: Homeowners show somewhat lower default rates than renters
#
# **Reasoning:** Categorical effects exist but are moderate compared to numeric features (FICO/interest rate).
# OneHotEncoding in Notebook 2 will easily incorporate these differences into the baseline model.
#
# ✅ **Validation:** Categorical features are predictive and straightforward to handle; no special encoding needed beyond standard OneHotEncoder.
# ➡️ **Next:** Time drift analysis and right-censoring check (critical for train/test split strategy).

# %% [markdown]
# ## 10) Time drift + censoring (critical for split strategy)
# **Purpose:** Identify time-based patterns in default rates and right-censoring artifacts to inform the train/test split strategy.
#
# Default rate is plotted over issue date (quarterly) **using only loans with final outcomes** (fully paid or charged off).
# This excludes "Current" loans that haven't reached a final outcome, which is essential for avoiding measurement bias.
#
# **Important:** Recent vintages (especially 2018) show artificially low default rates because many loans have not had time to default
# and are therefore excluded from the "final outcomes" sample. This is right-censoring: a systematic bias toward younger loans looking safer than they are.
# Using 2018 as a test set for "eventual default" would introduce severe bias (measuring only the fastest defaulters).

# %%
if "issue_d" in df_vis.columns:
    tmp = df_vis[["issue_d", "target_default"]].dropna()
    tmp["issue_q"] = tmp["issue_d"].dt.to_period("Q").astype(str)

    rate_by_q = tmp.groupby("issue_q")["target_default"].agg(["mean", "count"]).reset_index()

    # Filter to quarters with sufficient observations (avoid noisy extremes)
    MIN_Q_COUNT = 2000
    rate_by_q_filtered = rate_by_q[rate_by_q["count"] >= MIN_Q_COUNT].copy()

    # show tail where censoring is worst
    tail = rate_by_q.tail(12)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
    axes[0].plot(rate_by_q_filtered["issue_q"], rate_by_q_filtered["mean"], marker="o")
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].set_title(f"Default rate over time (quarterly, N≥{MIN_Q_COUNT})")
    axes[0].set_ylabel("Default rate")
    axes[0].grid(alpha=0.3)

    axes[1].plot(rate_by_q["issue_q"], rate_by_q["count"], marker="o", color='steelblue')
    axes[1].axhline(MIN_Q_COUNT, color='red', linestyle='--', alpha=0.5, label=f'Min threshold ({MIN_Q_COUNT})')
    axes[1].tick_params(axis="x", labelrotation=90)
    axes[1].set_title("Final-outcome sample size over time")
    axes[1].set_ylabel("Count (final outcomes only)")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Quarters retained after N≥{MIN_Q_COUNT} filter: {len(rate_by_q_filtered)}/{len(rate_by_q)}")
    print("\nDefault rate (tail 12 quarters, unfiltered):")
    print(tail.to_string())

# %%
# Feature drift (median by year) for a few key variables
if "issue_d" in df_vis.columns:
    df_drift = df_vis.copy()
    df_drift["issue_year"] = df_drift["issue_d"].dt.year

    drift_features = [c for c in ["fico_range_low", "int_rate", "dti", "annual_inc"] if c in df_drift.columns]
    drift_summary = df_drift.groupby("issue_year")[drift_features].median().dropna()

    print(drift_summary.tail(10).to_string())

    # Normalize each series to 0-1 scale for visual comparison across different units
    drift_norm = drift_summary.copy()
    for col in drift_norm.columns:
        col_min = drift_norm[col].min()
        col_max = drift_norm[col].max()
        drift_norm[col] = (drift_norm[col] - col_min) / (col_max - col_min + 1e-12)

    plt.figure(figsize=(10, 4))
    drift_norm.plot(marker='o')
    plt.title("Feature drift by year (normalized 0-1 scale)")
    plt.ylabel("Normalized median value")
    plt.xlabel("Issue year")
    plt.legend(loc='best', fontsize=9)
    plt.grid(alpha=0.3)
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

    print(status_q.tail(12).to_string())


# %% [markdown]
# ### Findings
# **Time drift and right-censoring artifacts confirmed:**
#
# **Default rate trend:**
# - Steady increase from **~15% (2007-2009)** to **~20% (2016-2017)**
# - Apparent collapse in **2018** (~18%), contradicting the uptrend
#
# **Censoring mechanism (explains 2018 collapse):**
# - Final-outcome sample size drops dramatically in 2018Q4 (only ~5k loans with final outcomes)
# - This is right-censoring: many loans issued in 2018 haven't reached maturity and are excluded from analysis
# - The 2018 "improvement" is an artifact: we're only observing the fastest-defaulting loans while excluding current/pending ones
#
# **Feature drift observed:**
# - FICO scores shift meaningfully across years
# - Interest rates reflect market conditions and LC's dynamic pricing
# - DTI and income show secular trends
#
# **Implications:**
# - Random train/test split violates temporal causality (training on future data)
# - Must use time-based validation to capture concept drift properly
# - 2018 cannot be used as test set due to censoring bias
#
# **Recommended time-based split for Notebook 2:**
# - **Train:** 2007–2014 (mature loans with known outcomes)
# - **Validation:** 2015 (recent but mature, for hyperparameter tuning)
# - **Test:** 2016–2017 (recent, final outcomes available, captures drift)
#
# ✅ **Validation:** Right-censoring is visible and time drift is measurable; time-based validation is essential.
# ➡️ **Next:** Maturity/censoring analysis by loan term, then multicollinearity and final feature selection.

# %% [markdown]
# ## 10a) Censoring by term (check for term-dependent label bias)
# **Purpose:** Verify that 60-month loans in test period (2016-2017) have sufficient final outcomes.
#
# If 60-month loans are selectively censored (still Current) while 36-month loans have reached final outcomes,
# the test set labels would be biased toward 36-month loans.

# %%
if "issue_d" in df_raw.columns and "term" in df_raw.columns and "loan_status" in df_raw.columns:
    df_term = df_raw[["issue_d", "term", "loan_status"]].copy()
    df_term["issue_d"] = parse_month_year(df_term["issue_d"])
    df_term = df_term.dropna(subset=["issue_d", "term"])
    df_term["issue_year"] = df_term["issue_d"].dt.year
    df_term["term_months"] = parse_term(df_term["term"])

    # Filter to test period years (2016-2017)
    test_years = df_term[df_term["issue_year"].isin([2016, 2017])].copy()

    # Compute % Current by term
    term_status = test_years.groupby(["issue_year", "term_months"]).apply(
        lambda g: pd.Series({
            "pct_current": (g["loan_status"] == "Current").mean(),
            "n_loans": len(g),
            "n_final": ((g["loan_status"] == "Fully Paid") | (g["loan_status"] == "Charged Off") | (g["loan_status"] == "Default")).sum()
        })
    ).reset_index()

    print("\nCensoring by term (test period 2016-2017):")
    print(term_status.to_string())

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    for term_val in term_status["term_months"].unique():
        subset = term_status[term_status["term_months"] == term_val]
        ax.plot(subset["issue_year"], subset["pct_current"], marker="o", label=f"{int(term_val)} months")
    ax.set_title("% Current loans by term (2016-2017 vintages)")
    ax.set_xlabel("Issue year")
    ax.set_ylabel("% Current (not final outcome)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Findings
# **Term-dependent censoring check:**
# - **36-month loans:** Lower % Current in test period (mature loans, most reached final outcomes)
# - **60-month loans:** Higher % Current in test period (longer maturity = more still Current), but still sufficient final outcomes for evaluation
# - **Validation:** Both terms have adequate final-outcome sample sizes in 2016-2017 test period
# - **No severe bias:** While 60-month loans have higher censoring (expected due to longer maturity), the test set is not dominated by one term
# - **Decision:** Test set (2016-2017) is valid for both terms; no need to stratify by term in baseline model

# %% [markdown]
# ## 11) Multicollinearity scan (practical)
# **Purpose:** Identify feature redundancy and measure the linear dependencies between numeric features.
#
# VIF (Variance Inflation Factor) is computed on a compact numeric set to identify redundancy.
# High VIF (>10) indicates that a feature can be predicted from other features, suggesting information duplication.

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
    print("\nVariance Inflation Factor (VIF):")
    print(vif_table.to_string())

# %% [markdown]
# ### Findings
# **Feature redundancy confirmed (multicollinearity detected):**
# - `loan_amnt` and `funded_amnt`: **VIF > 1100** (near-perfect multicollinearity)
#   - **Reasoning:** These are nearly identical columns: `funded_amnt` is the amount actually funded, which equals `loan_amnt` in ~99.96% of cases
#   - **Decision:** Keep `loan_amnt`, drop `funded_amnt` and `funded_amnt_inv`
# - `installment`: **VIF ~11.5** (moderate multicollinearity as expected)
#   - **Reasoning:** Installment is structurally derived from loan_amnt, term, and int_rate (monthly_payment = loan_amnt * rate_factor / nmonths)
#   - **Decision:** Exclude from initial baseline; include only in ablation studies to isolate int_rate signal
#
# **Modeling implication:** Dropping redundant columns prevents multicollinearity-driven coefficient instability and reduces noise in baseline models.
#
# ✅ **Validation:** Concrete evidence of feature redundancy and clear simplification strategy identified.
# ➡️ **Next:** Finalize feature sets and export EDA artifacts for Notebook 2.

# %% [markdown]
# ## 12) EDA decisions register → inputs for Notebook 2
# **Modeling feature sets (explicit):**
#
# ### A) Applicant/origination baseline (no LC pricing outputs)
# Excludes `grade`, `sub_grade`, and `int_rate` (plus derived `installment` by default).
#
# ### B) Investor-view model (includes LC's published pricing)
# Includes `int_rate` (and optionally `installment`); an **ablation study** will quantify how much signal comes from pricing.
#
# This two-model approach (applicant-only baseline vs. investor-view with pricing) ensures methodologically honest reporting:
# the investor-view model's superior performance is attributed to available pricing information, not superior feature engineering.

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
# # EDA → Modeling Handoff: Data Filtering & Cleaning Plan
#
# This section locks the exact preprocessing decisions to carry into Notebook 2.
# The goal is to make the modeling step reproducible and defensible.
#
# ## 1) Target + filtering
# - **Target:** `target_default = 1` for {Charged Off, Default}, `0` for {Fully Paid}
# - **Filter:** Drop non-final statuses (e.g., `Current`) for this baseline "eventual default" task
# - **Censoring risk:** Recent vintages (especially 2018) have far fewer final outcomes
#   - **Action:** 2018 loans are **EXCLUDED** from modeling dataset due to right-censoring bias
#   - Test set uses 2016-2017 vintages with mature loans that have reached final outcomes
#
# ## 2) Leakage & circular reasoning drops (non-negotiable)
# **Drop LendingClub risk outputs:**
# - `grade`, `sub_grade` - LC's proprietary risk scores (circular reasoning / answer key)
#
# **Drop post-origination/servicing variables:**
# - Payment history: `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`
# - Balances: `out_prncp`, `out_prncp_inv`, `recoveries`, `collection_recovery_fee`
# - Hardship/settlement: All 20 hardship_* and settlement_* fields
# - Servicing operational: `payment_plan_start_date`, `deferral_term`, `orig_projected_additional_accrued_interest`
# - Status fields: `loan_status`, `pymnt_plan`, `last_pymnt_d`, `last_pymnt_amnt`, `last_credit_pull_d`
#
# **Reasoning:** These leak the outcome or encode information unavailable at origination
#
# ## 3) Missingness policy
# **Drop ultra-sparse columns:** >70% missing (19 columns)
# - Dominated by `sec_app_*` and `*_joint` fields (secondary applicant data)
# - Sparse operational fields: `payment_plan_*`, `orig_projected_*` (also leakage)
# - Text field `desc`: ~90% missing, difficult to encode
#
# **Keep moderate-missing features:** <70% missing, will impute in Notebook 2
# - Recency variables: `mths_since_*` (60-80% missing but legitimate "no event" pattern)
# - Account metrics: `il_util`, `bc_util` (moderate missing)
#
# **Imputation strategy (Notebook 2):**
# - Numeric: Median imputation (validated by Section 5a missingness-as-signal check)
# - Categorical: "missing" category
# - **Critical:** Fit imputer on train set only to avoid test leakage
#
# ## 4) Type parsing & feature engineering
# **Parsing applied in EDA:**
# - Dates: `issue_d` and `earliest_cr_line` (both %b-%y and %b-%Y formats handled)
# - Percent strings: `int_rate`, `revol_util` (strip "%" and convert to float)
# - Term: Extract numeric months from "36 months" / "60 months"
# - Employment length: Map "10+ years" → 10, "< 1 year" → 0, etc.
#
# **Feature engineering:**
# - `credit_history_years = (issue_d - earliest_cr_line) / 365.25`
# - Sanity checked: min ~3 years, max ~83 years, 99th percentile ~40 years
#
# ## 5) Data quality rules (applied in Notebook 2 pipeline, learned on train only)
# **Validity fixes:**
# - `revol_util > 100` → set to missing (invalid percent)
# - `dti > 100` → cap at 100 (impossible debt-to-income ratio)
#
# **Outlier handling:**
# - `annual_inc`, `revol_bal`: Heavy right-skew → apply robust transforms or cap at train-only p99
# - Count features (`inq_last_6mths`, `delinq_2yrs`, `pub_rec`, `tax_liens`): Zero-inflated → cap at train-only p99
#
# **Critical:** All thresholds (p99, validity caps) must be learned on train set only
#
# ## 6) Redundancy handling (VIF analysis findings)
# **Drop redundant features:**
# - `funded_amnt` (VIF > 1100, 99.96% correlated with `loan_amnt`)
# - `funded_amnt_inv` (VIF > 500, also redundant with `loan_amnt`)
#
# **Keep:**
# - `loan_amnt` (primary loan amount feature)
#
# **Exclude from baseline, revisit in ablation:**
# - `installment` (VIF ~11.5, structurally derived from loan_amnt, term, int_rate)
# - Will test in ablation study to isolate int_rate signal
#
# ## 7) Feature sets (two honest baselines)
# **A) Applicant/origination baseline (no LC pricing outputs):**
# - Excludes: `grade`, `sub_grade`, `int_rate`, `installment`
# - 19 features: loan_amnt, term, annual_inc, dti, fico_range_low, revol_util, revol_bal,
#   open_acc, total_acc, inq_last_6mths, delinq_2yrs, pub_rec, tax_liens, credit_history_years,
#   emp_length, home_ownership, verification_status, purpose, addr_state
#
# **B) Investor-view baseline (includes LC published pricing):**
# - Includes: All features from (A) + `int_rate`
# - 20 features total
# - **Ablation required:** Quantify how much AUC lift comes from `int_rate` (LC's pricing signal)
#
# **Reasoning:** This two-model approach ensures methodologically honest reporting:
# - Applicant-only model: independent risk assessment
# - Investor-view model: acknowledges LC's pricing information is available and predictive
#
# ## 8) Split strategy (production realism)
# **Time-based split (respects drift and censoring):**
# - **Train:** 2007–2014 (mature loans, known outcomes)
# - **Validation:** 2015 (recent but mature, for hyperparameter tuning)
# - **Test:** 2016–2017 (recent, final outcomes available, captures drift)
# - **Excluded:** 2018 (right-censoring bias, insufficient final outcomes)
#
# **Reasoning:**
# - Random train_test_split violates temporal causality (future information in training)
# - Feature drift observed across years (FICO, DTI, income, rates)
# - Time-based validation simulates production deployment scenario
#
# ## Open decision (to resolve in Notebook 2)
# **Fixed-horizon target option:**
# - Current target: "eventual default" (biased toward mature loans)
# - Alternative: "default within 12/18 months" (reduces maturity bias, more comparable across vintages)
# - **Decision point:** Evaluate in Notebook 2 if censoring bias remains problematic

# %% [markdown]
# ## Summary
# ✅ **EDA Complete:** All preprocessing decisions have been documented in the "EDA → Modeling Handoff" summary above.
#
# 👉 **Proceed to Notebook 2 (2_Modelling.py)** using:
# - Raw dataset (Notebook 2 loads `accepted_2007_to_2018Q4.csv` directly)
# - Handoff summary for all preprocessing decisions
# - Two feature sets: applicant-only baseline and investor-view baseline
# - Recommended time-based split: train 2007-2014, val 2015, test 2016-2017


# %%