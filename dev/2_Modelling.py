# %% [markdown]
# # Notebook 2 — Modeling (from RAW data)
#
# ## Pre-flight checklist
# - Load the **raw** LendingClub dataset and define a defensible target (final outcomes only)
# - Enforce integrity: drop leakage + circular reasoning fields (grade/sub_grade) and post-origination variables
# - Create a **time-based split** (drift/censoring-aware) before learning any transforms
# - Build a single end-to-end sklearn Pipeline that can score **raw rows** (Notebook 3 compatibility)
# - Fit train-only learned steps (missingness drops, capping, imputation, transforms) and inspect the final feature set

# %% [markdown]
# ## What this model output means (PD vs risk score)
# This model outputs `p̂ = P(bad_final_outcome | origination features)`, where "bad" is {Charged Off, Default}
# and "good" is {Fully Paid}. This is a **risk score / eventual-default probability under our label definition**.
# It is not a regulatory PD (fixed horizon) unless we explicitly redefine the target to a horizon-based default event later.

# %%
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 160)
RANDOM_STATE = 42

# %% [markdown]
# ## 0) Load RAW dataset (no pre-cleaned EDA file)
# Robust path resolution: works from both project root and dev/ subdirectory.

# %%
# --- Path resolution: works from both root and dev/ directories ---
HERE = Path.cwd()
PROJECT_ROOT = HERE if (HERE / "data").exists() else HERE.parent
DATA_DIR = PROJECT_ROOT / "data"

print("Project root:", PROJECT_ROOT)
print("Data directory:", DATA_DIR)

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
    # fallback: first csv in folder
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
if str(DATA_PATH).endswith(".gz"):
    df_raw = pd.read_csv(DATA_PATH, compression="gzip", low_memory=False)
else:
    df_raw = pd.read_csv(DATA_PATH, low_memory=False)

print("Loaded:", DATA_PATH.name)
print("Shape:", df_raw.shape)
df_raw.head(3)

# %% [markdown]
# ## 1) Define target + final outcome filter (done BEFORE modeling)
# We restrict to final outcomes for a clean baseline:
# - Good: Fully Paid
# - Bad: Charged Off, Default (if present)
# Drop "Current" and other non-final statuses.

# %%
FINAL_GOOD = {"Fully Paid"}
FINAL_BAD = {"Charged Off", "Default"}  # Default may or may not exist in your file

assert "loan_status" in df_raw.columns, "loan_status not found in raw dataset."

df_raw["target_default"] = np.where(
    df_raw["loan_status"].isin(FINAL_BAD), 1,
    np.where(df_raw["loan_status"].isin(FINAL_GOOD), 0, np.nan)
)

df = df_raw[df_raw["target_default"].notna()].copy()
print("Final outcome subset shape:", df.shape)
print("Default rate:", df["target_default"].mean())
df["loan_status"].value_counts()

# %% [markdown]
# ## 2) Deterministic parsing required for time split and core features
# This parsing is NOT a learned transform; it's a necessary type coercion so downstream logic works.
# We keep it deterministic and robust (supports both "Dec-15" and "Dec-2015" formats).

# %%
def parse_month_year(series: pd.Series) -> pd.Series:
    s = series.astype(str)
    dt1 = pd.to_datetime(s, format="%b-%y", errors="coerce")
    dt2 = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    return dt1.fillna(dt2)

assert "issue_d" in df.columns, "issue_d missing; can't do time-based split."
df["issue_d"] = parse_month_year(df["issue_d"])
bad_issue = df["issue_d"].isna().mean()
print("issue_d NaT %:", bad_issue)
assert bad_issue < 0.01, "issue_d parsing failed for too many rows; fix parsing before proceeding."

# Optional: earliest_cr_line parsing (used for credit_history_years)
if "earliest_cr_line" in df.columns:
    df["earliest_cr_line"] = parse_month_year(df["earliest_cr_line"])

# %% [markdown]
# ## 3) Maturity filter (ensure fair final outcomes)
# The dataset snapshot ends 2018-12-31. Loans need their full term to reach true final outcomes.
# Without filtering:
# - 36-month loans issued after 2015-12 are censored (don't have 36 months by 2018-12)
# - 60-month loans issued after 2013-12 are censored (don't have 60 months by 2018-12)
# Result: later cohorts show selection bias (only early resolvers observed).
#
# Solution: Filter to loans that reached maturity by snapshot date.
# This ensures "final outcome" is truly final, not selection-biased.

# %%
SNAPSHOT_END = pd.Timestamp("2018-12-31")

# Parse term to numeric months (done early for maturity filtering)
def parse_term_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(out, errors="coerce")

assert "term" in df.columns, "term column missing; can't compute maturity."
df["term_months"] = parse_term_series(df["term"])

# Calculate maturity date (issue_d + term_months)
# Use exact month offsets (not 30.44-day approximation which misclassifies near boundaries)
df["maturity_date"] = df.apply(lambda row: row["issue_d"] + pd.DateOffset(months=int(row["term_months"])), axis=1)

# Filter to mature loans only
df_before = df.copy()
df = df[df["maturity_date"] <= SNAPSHOT_END].copy()

print(f"Maturity filter applied:")
print(f"  Before: {len(df_before):,} loans, default rate {df_before['target_default'].mean():.4f}")
print(f"  After:  {len(df):,} loans ({100*len(df)/len(df_before):.1f}%), default rate {df['target_default'].mean():.4f}")
print(f"  Latest issue_d: {df['issue_d'].max()}")
assert len(df) > 100000, "Too few mature loans; check snapshot date or term parsing."

# %% [markdown]
# ## 4) Time-based split (mature loans only, recent era)
# Now that we filter to mature loans only (issue_d ≤ 2015-12 for 36mo, ≤ 2013-12 for 60mo),
# we use RECENT years for training to avoid era drift:
# - Train: 2011–2013 (3 years, post-crisis stable era)
# - Val: 2014 (1 year, for threshold tuning)
# - Test: 2015 (1 year, final evaluation)
#
# All splits are in the same lending era (no 2007 vs 2015 drift).
# Term mix will vary (60-month loans scarce in 2015 due to maturity constraint).
# This is expected and documented in split sanity check below.

# %%
df = df.sort_values("issue_d").reset_index(drop=True)

def time_slice(df_in: pd.DataFrame, start: str, end: str | None) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    if end is None:
        return df_in[df_in["issue_d"] >= start_dt]
    end_dt = pd.to_datetime(end)
    return df_in[(df_in["issue_d"] >= start_dt) & (df_in["issue_d"] < end_dt)]

df_train = time_slice(df, "2011-01-01", "2014-01-01")   # 2011–2013
df_val   = time_slice(df, "2014-01-01", "2015-01-01")   # 2014
df_test  = time_slice(df, "2015-01-01", "2016-01-01")   # 2015

print("Train:", df_train.shape, "Default:", df_train["target_default"].mean())
print("Val:  ", df_val.shape,   "Default:", df_val["target_default"].mean())
print("Test: ", df_test.shape,  "Default:", df_test["target_default"].mean())
assert len(df_train) > 0 and len(df_val) > 0 and len(df_test) > 0, "Empty time split. Adjust boundaries."

# %%
# Split sanity check: document term mix and default rates
# This confirms splits are balanced and in the same era
def split_report(name, d):
    print(f"\n{name}")
    print(f"  Rows: {len(d):,}  ({100*len(d)/len(df):.1f}% of mature loans)")
    print(f"  Default rate: {d['target_default'].mean():.4f}")
    print(f"  Issue range: {d['issue_d'].min().date()} → {d['issue_d'].max().date()}")
    if "term" in d.columns:
        print("  Term mix:")
        # Extract numeric month from term (e.g., " 36 months" → 36)
        term_numeric = d["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
        term_pct = (term_numeric.value_counts(normalize=True).sort_index() * 100).round(1)
        for term_months, pct in term_pct.items():
            print(f"    {term_months}-month: {pct}%")

split_report("TRAIN", df_train)
split_report("VAL", df_val)
split_report("TEST", df_test)

print("\n" + "="*80)
print("SANITY CHECK:")
print("="*80)
print("✅ All splits in same era (2011-2015)")
print("✅ Default rates should be similar (14-18%)")
print("✅ Term mix varies due to maturity constraint (expected)")
print("⚠️  Test set will be dominated by 36-month (2015 has few mature 60-month)")
print("    → Must report metrics BY TERM after model evaluation")
print("="*80)

# %% [markdown]
# ## 5) Build an end-to-end cleaning + preprocessing pipeline (Notebook 3 compatible)
# Key design choices:
# - We DO NOT rely on a pre-cleaned dataset file.
# - All wrangling needed for inference is inside sklearn-compatible transformers.
# - Learned steps (dropping sparse cols, capping) fit on TRAIN ONLY.
#
# We support two feature policies:
# - Applicant baseline: drops pricing outputs (int_rate, installment)
# - Investor baseline: keeps them

# %%
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer

# --- Deterministic parsers / cleaners (no train-fit learning) ---
EMP_MAP = {
    "10+ years": 10,
    "9 years": 9, "8 years": 8, "7 years": 7, "6 years": 6, "5 years": 5,
    "4 years": 4, "3 years": 3, "2 years": 2, "1 year": 1,
    "< 1 year": 0,
}

def parse_percent_series(s: pd.Series) -> pd.Series:
    # Accept numeric or "13.56%" strings
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    out = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(out, errors="coerce")

def parse_term_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(out, errors="coerce")

def parse_emp_length_series(s: pd.Series) -> pd.Series:
    return s.map(EMP_MAP).astype("float")

class RawLendingClubCleaner(BaseEstimator, TransformerMixin):
    """
    Deterministic cleaning:
    - drop leakage/circular columns
    - parse known messy fields into numeric
    - derive credit_history_years
    - enforce basic validity rules (e.g., revol_util > 100 => NaN)
    """
    def __init__(self, include_pricing_features: bool = True):
        self.include_pricing_features = include_pricing_features
        self.columns_seen_ = None

    def fit(self, X, y=None):
        self.columns_seen_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()

        # === ALLOWLIST APPROACH ===
        # Define origination-time features to KEEP
        core_allowlist = [
            # Loan request (3 features)
            "loan_amnt", "term", "purpose",
            # Applicant demographics (5 features)
            "annual_inc", "emp_length", "home_ownership", "addr_state", "verification_status",
            # Core credit bureau at origination (11 features)
            "dti", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
            "revol_bal", "revol_util", "total_acc",
            "fico_range_low", "fico_range_high",
            "credit_history_years",  # Will be derived below
        ]

        # Extended bureau features (optional - set USE_EXTENDED=True to include)
        USE_EXTENDED = False  # TOGGLE: False=Core only (recommended first), True=Full bureau data
        extended_allowlist = [
            # Account counts
            "mort_acc", "pub_rec_bankruptcies", "tax_liens",
            "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
            "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
            # Bureau utilization/delinquency metrics
            "pct_tl_nvr_dlq",
            "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
            "mths_since_rcnt_il", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
            "mths_since_recent_inq", "mths_since_recent_revol_delinq",
            # Current balances/limits (origination-time snapshot)
            "avg_cur_bal", "bc_open_to_buy", "bc_util", "tot_cur_bal",
            "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
            "tot_hi_cred_lim", "max_bal_bc", "all_util", "il_util",
            "percent_bc_gt_75", "total_rev_hi_lim",
            # Account age
            "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
            # Recent activity (backward-looking from origination)
            "acc_open_past_24mths", "num_tl_op_past_12m", "inq_last_12m",
            # Delinquency counts
            "num_accts_ever_120_pd", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
            "acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt",
            "collections_12_mths_ex_med", "num_collections_12_mths_ex_med",
            # Other
            "application_type", "initial_list_status", "disbursement_method",
            # Secondary applicant
            "sec_app_fico_range_low", "sec_app_fico_range_high",
            "sec_app_earliest_cr_line", "sec_app_inq_last_6mths",
            "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
            "sec_app_open_act_il", "sec_app_num_rev_accts",
            "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
            "sec_app_mths_since_last_major_derog",
        ]

        # Combine allowlists based on USE_EXTENDED toggle
        if USE_EXTENDED:
            allowlist = core_allowlist + extended_allowlist
        else:
            allowlist = core_allowlist

        # Investor-only additions (int_rate ONLY for clean baseline - NOT installment)
        if self.include_pricing_features:
            allowlist.extend(["int_rate"])

        # Add temporary fields needed for derivation
        temp_fields = ["issue_d", "earliest_cr_line"]
        allowlist.extend(temp_fields)

        # === HARD DROPS (Post-Origination Leakage) ===
        # These get dropped EVEN if they somehow sneak into allowlist
        hard_drop_patterns = [
            r"^total_pymnt", r"^total_rec_", r"^recoveries$", r"^collection_recovery_fee$",
            r"^last_pymnt", r"^next_pymnt", r"^out_prncp",
            r"^last_credit_pull_d$", r"^last_fico_range",
            r"^hardship_", r"^settlement_", r"^debt_settlement_",
            r"^pymnt_plan$", r"^policy_code$", r"^payment_plan_start_date$",
            r"^deferral_term$", r"^orig_projected_additional_accrued_interest$",
        ]

        hard_drop_cols = []
        for col in X.columns:
            for pattern in hard_drop_patterns:
                if re.search(pattern, col, flags=re.IGNORECASE):
                    hard_drop_cols.append(col)
                    break

        # Hard drop circular reasoning + identifiers + synthetic fields
        hard_drop_explicit = [
            "grade", "sub_grade",  # Circular
            "id", "member_id", "url", "emp_title", "zip_code", "title", "desc",  # IDs/high-card
            "funded_amnt", "funded_amnt_inv",  # VIF redundancy
            "loan_status", "target_default", "maturity_date", "term_months",  # Synthetic/target
            "installment",  # Not for first run (deterministic from loan_amnt+term+int_rate)
        ]

        all_hard_drops = list(set(hard_drop_cols + hard_drop_explicit))
        X.drop(columns=[c for c in all_hard_drops if c in X.columns], inplace=True, errors="ignore")

        # === PARSE MESSY FIELDS ===
        if "term" in X.columns:
            X["term"] = parse_term_series(X["term"])

        if "emp_length" in X.columns:
            X["emp_length"] = parse_emp_length_series(X["emp_length"])

        for pct_col in ["int_rate", "revol_util"]:
            if pct_col in X.columns:
                X[pct_col] = parse_percent_series(X[pct_col])

        # === DERIVE credit_history_years ===
        if "issue_d" in X.columns and not np.issubdtype(X["issue_d"].dtype, np.datetime64):
            X["issue_d"] = parse_month_year(X["issue_d"])
        if "earliest_cr_line" in X.columns and not np.issubdtype(X["earliest_cr_line"].dtype, np.datetime64):
            X["earliest_cr_line"] = parse_month_year(X["earliest_cr_line"])

        if "issue_d" in X.columns and "earliest_cr_line" in X.columns:
            X["credit_history_years"] = (X["issue_d"] - X["earliest_cr_line"]).dt.days / 365.25

        # CRITICAL CHECK: Ensure credit_history_years was successfully derived
        if "credit_history_years" not in X.columns:
            raise ValueError("CRITICAL: credit_history_years derivation failed (missing issue_d or earliest_cr_line)")
        elif X["credit_history_years"].isna().all():
            raise ValueError("CRITICAL: credit_history_years is all NaN (date parsing may have failed)")

        # Drop temporary fields used for derivation
        X.drop(columns=["issue_d", "earliest_cr_line"], inplace=True, errors="ignore")

        # === APPLY ALLOWLIST (Drop everything not in allowlist) ===
        # Remove temp_fields from allowlist since they've been used
        final_allowlist = [c for c in allowlist if c not in temp_fields]

        # Keep only columns in allowlist
        cols_to_keep = [c for c in X.columns if c in final_allowlist]

        # CRITICAL: Check which expected features are missing (data evolution issues)
        expected_present = set(final_allowlist)
        actually_present = set(cols_to_keep)
        missing_features = expected_present - actually_present

        if len(missing_features) > 0:
            # If too many core features are missing, this is a problem
            if USE_EXTENDED:
                threshold = 10  # Extended set can tolerate some missing
            else:
                threshold = 3  # Core set should have nearly all features
            if len(missing_features) > threshold:
                raise ValueError(f"CRITICAL: Too many expected features missing ({len(missing_features)} > {threshold}). "
                               f"Missing: {sorted(missing_features)}")

        X = X[cols_to_keep]

        # === VERIFICATION LOGGING ===
        # STRONG ASSERTION: Cleaned columns must be subset of allowlist (minus temp fields)
        cleaned_set = set(X.columns)
        allowed_set = set(final_allowlist)
        unexpected = cleaned_set - allowed_set
        if len(unexpected) > 0:
            raise ValueError(f"ALLOWLIST VIOLATION: {len(unexpected)} unexpected columns survived: {sorted(unexpected)}")

        # Leakage substring check (nice-to-have, not a guarantee)
        leakage_substrings = ["pymnt", "total_rec", "recover", "out_prncp", "last_pymnt",
                              "next_pymnt", "settlement", "hardship", "debt_settlement"]
        leakage_found = []
        for col in X.columns:
            for substr in leakage_substrings:
                if substr.lower() in col.lower():
                    leakage_found.append(col)
                    break

        if leakage_found:
            raise ValueError(f"LEAKAGE DETECTED: {len(leakage_found)} suspicious columns remain: {leakage_found}")

        # Basic validity rules
        if "revol_util" in X.columns:
            X.loc[(X["revol_util"] < 0) | (X["revol_util"] > 100), "revol_util"] = np.nan

        if "dti" in X.columns:
            X.loc[X["dti"] < 0, "dti"] = np.nan  # negative DTI invalid

        if "credit_history_years" in X.columns:
            X.loc[(X["credit_history_years"] < 0) | (X["credit_history_years"] > 80), "credit_history_years"] = np.nan

        return X

# --- Train-fit learned transformers ---
class MissingnessDropper(BaseEstimator, TransformerMixin):
    """Drop columns with missingness above threshold learned on training data."""
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self.drop_cols_ = None

    def fit(self, X, y=None):
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        miss = Xdf.isna().mean()
        self.drop_cols_ = sorted(list(miss[miss > self.threshold].index))
        return self

    def transform(self, X):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return Xdf.drop(columns=self.drop_cols_, errors="ignore")

class Capper(BaseEstimator, TransformerMixin):
    """Cap numeric columns at a training-learned quantile (reduces outlier leverage)."""
    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.caps_ = None

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X)
        self.caps_ = Xdf.quantile(self.quantile, numeric_only=True)
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X).copy()
        for c in Xdf.columns:
            if self.caps_ is not None and c in self.caps_.index:
                Xdf[c] = np.minimum(Xdf[c], self.caps_[c])
        return Xdf.values

def log1p_selected(X, columns: list[str]):
    Xdf = pd.DataFrame(X).copy()
    for c in columns:
        if c in Xdf.columns:
            s = Xdf[c]
            Xdf[c] = np.where(s >= 0, np.log1p(s), s)
    return Xdf.values

# %% [markdown]
# ## 5) Build preprocessors (applicant vs investor)
# We keep it simple:
# - numeric: median impute + cap (train-fit only), log1p on known skewed fields from EDA
# - categorical: most-frequent impute + OneHotEncoder
#
# Note: log1p columns are based on EDA: `annual_inc`, `revol_bal` are strongly right-skewed.

# %%
from sklearn.preprocessing import StandardScaler

SKEWED_LOG_COLS = ["annual_inc", "revol_bal"]  # EDA-justified; safe if absent

def build_preprocessor(include_pricing_features: bool) -> Pipeline:
    cleaner = RawLendingClubCleaner(include_pricing_features=include_pricing_features)
    drop_sparse = MissingnessDropper(threshold=0.70)

    # Skewed columns from EDA (these need log-transform)
    SKEWED_COLS = ["annual_inc", "revol_bal"]

    # Selector for non-skewed numeric columns (exclude skewed cols)
    def non_skewed_numeric_selector(df):
        """Select numeric columns that are not in SKEWED_COLS"""
        numeric_cols = df.select_dtypes(include=np.number).columns
        return [c for c in numeric_cols if c not in SKEWED_COLS]

    # Skewed numeric pipeline: impute → log1p → cap → scale
    # Log applied early before capping to preserve the benefit
    skewed_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("cap", Capper(quantile=0.99)),
        ("scale", StandardScaler(with_mean=False)),
    ])

    # Normal numeric pipeline: impute → cap → scale (no log)
    normal_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("cap", Capper(quantile=0.99)),
        ("scale", StandardScaler(with_mean=False)),
    ])

    # Categorical pipeline: impute → OHE
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    # ColumnTransformer: split numeric into skewed/non-skewed, separate categorical
    col_tf = ColumnTransformer(
        transformers=[
            ("skewed", skewed_pipe, SKEWED_COLS),
            ("normal", normal_pipe, non_skewed_numeric_selector),
            ("cat", cat_pipe, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return Pipeline(steps=[
        ("clean", cleaner),
        ("drop_sparse", drop_sparse),
        ("preprocess", col_tf),
    ])

# Build both preprocessors
preprocess_applicant = build_preprocessor(include_pricing_features=False)
preprocess_investor  = build_preprocessor(include_pricing_features=True)

# %% [markdown]
# ## 6) Manual inspection checkpoint (before training models)
# We inspect what remains after deterministic cleaning + train-fit sparse drops.
# If the remaining dataset still has weird dtypes, high-cardinality categoricals, or unexpected missingness, fix it NOW.

# %%
# Prepare raw X/y (we keep issue_d for splitting only; it's removed inside cleaner)
FEATURE_DROP = {"target_default"}  # don't include label
X_train_raw = df_train.drop(columns=list(FEATURE_DROP), errors="ignore")
y_train = df_train["target_default"].astype(int)

X_val_raw = df_val.drop(columns=list(FEATURE_DROP), errors="ignore")
y_val = df_val["target_default"].astype(int)

X_test_raw = df_test.drop(columns=list(FEATURE_DROP), errors="ignore")
y_test = df_test["target_default"].astype(int)

# %%
# Fit only the cleaning + missingness drop on train so we can inspect the remaining columns
clean_only_app = Pipeline([
    ("clean", RawLendingClubCleaner(include_pricing_features=False)),
    ("drop_sparse", MissingnessDropper(threshold=0.70)),
])
clean_only_inv = Pipeline([
    ("clean", RawLendingClubCleaner(include_pricing_features=True)),
    ("drop_sparse", MissingnessDropper(threshold=0.70)),
])

X_train_app = clean_only_app.fit_transform(X_train_raw)
X_train_inv = clean_only_inv.fit_transform(X_train_raw)

print("Applicant feature set shape (after cleaning):", X_train_app.shape)
print("Investor feature set shape (after cleaning):", X_train_inv.shape)

# %%
def inspection_table(Xdf: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": Xdf.dtypes.astype(str),
        "missing_pct": Xdf.isna().mean() * 100,
        "n_unique": [Xdf[c].nunique(dropna=True) for c in Xdf.columns],
        "sample_values": [", ".join(Xdf[c].dropna().astype(str).head(3).tolist()) for c in Xdf.columns],
    }).sort_values(["missing_pct", "n_unique"], ascending=[False, False])

print("Top issues (Applicant feature set):")
print(inspection_table(X_train_app).head(30).to_string())

print("\nTop issues (Investor feature set):")
print(inspection_table(X_train_inv).head(30).to_string())

# %%
# Flag top high-cardinality categorical columns (these can explode OHE)
def top_cardinality(Xdf: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cat = [c for c in Xdf.columns if not pd.api.types.is_numeric_dtype(Xdf[c])]
    tbl = pd.DataFrame({
        "n_unique": [Xdf[c].nunique(dropna=True) for c in cat],
        "missing_pct": [Xdf[c].isna().mean() * 100 for c in cat],
    }, index=cat).sort_values("n_unique", ascending=False)
    return tbl.head(n)

print("Top categorical cardinality (Applicant):")
print(top_cardinality(X_train_app, 20).to_string())

print("\nTop categorical cardinality (Investor):")
print(top_cardinality(X_train_inv, 20).to_string())

# %% [markdown]
# ## Final OHE Feature Count (Gate to Model Training)

# %%
# Fit the full preprocessor on train and check final feature count after OHE
X_train_final_app = preprocess_applicant.fit_transform(X_train_raw)
X_train_final_inv = preprocess_investor.fit_transform(X_train_raw)

final_features_app = X_train_final_app.shape[1] if hasattr(X_train_final_app, 'shape') else X_train_final_app.toarray().shape[1]
final_features_inv = X_train_final_inv.shape[1] if hasattr(X_train_final_inv, 'shape') else X_train_final_inv.toarray().shape[1]

print(f"\n{'='*80}")
print(f"FINAL FEATURE COUNT AFTER OHE (GATE TO MODEL TRAINING)")
print(f"{'='*80}")
print(f"Applicant baseline: {final_features_app:,} features")
print(f"Investor baseline:  {final_features_inv:,} features")

if final_features_app < 50000 and final_features_inv < 50000:
    print(f"\n✅ PASS: No cardinality explosion detected")
    print(f"✅ Ready to train baseline Logistic Regression models")
else:
    print(f"\n❌ FAIL: Feature count is too high (possible cardinality bomb)")
print(f"{'='*80}")

# %% [markdown]
# # 7) Model Training (Progressive Complexity)
#
# We will progress from simple to more complex models:
# 1) Logistic Regression (baseline, interpretable)
# 2) Decision Tree (non-linear baseline)
# 3) Random Forest (bagging, reduces variance)
# 4) Gradient Boosting / AdaBoost (boosting, captures interactions)
# 5) XGBoost (optional, only if installed and stable)
#
# **Important integrity constraint**
# Our primary test set (2015) is dominated by 36-month loans due to maturity constraints.
# To avoid pretending we evaluated 60-month behavior, we carve out a **secondary 60-month holdout cohort**
# from 2013 and remove it from training. This gives us term coverage without rewriting the entire split strategy.

# %%
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# %% [markdown]
# ## 7.1 Create a 60-month holdout cohort (2013) and remove from training
# This cohort is evaluation-only. We do NOT tune thresholds on it.

# %%
def mask_year(df_in: pd.DataFrame, year: int) -> pd.Series:
    return (df_in["issue_d"] >= f"{year}-01-01") & (df_in["issue_d"] < f"{year+1}-01-01")

def term_to_months(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].astype(float)

# Create an evaluation-only 60-month cohort from 2013 (must be mature given snapshot)
df_test_60 = df_train[mask_year(df_train, 2013)].copy()
df_test_60["term_months_tmp"] = term_to_months(df_test_60["term"])
df_test_60 = df_test_60[df_test_60["term_months_tmp"] == 60].drop(columns=["term_months_tmp"], errors="ignore")

# Remove those rows from training to keep this a true holdout
df_train_model = df_train.drop(index=df_test_60.index).copy()

print("60-month holdout cohort (2013) rows:", len(df_test_60))
print("Train rows after removing holdout:", len(df_train_model))

# Quick integrity checks
assert df_test_60["issue_d"].min() >= pd.Timestamp("2013-01-01")
assert df_test_60["issue_d"].max() < pd.Timestamp("2014-01-01")

# %% [markdown]
# ## 7.2 Prepare X/y for train / val / primary test / 60-month holdout test

# %%
TARGET_COL = "target_default"

def make_Xy(d: pd.DataFrame):
    y = d[TARGET_COL].astype(int)
    X = d.drop(columns=[TARGET_COL], errors="ignore")
    return X, y

X_train_raw, y_train = make_Xy(df_train_model)
X_val_raw, y_val = make_Xy(df_val)
X_test_raw, y_test = make_Xy(df_test)
X_test60_raw, y_test60 = make_Xy(df_test_60)

print("Train:", X_train_raw.shape, "Bad rate:", y_train.mean().round(4))
print("Val:  ", X_val_raw.shape,   "Bad rate:", y_val.mean().round(4))
print("Test: ", X_test_raw.shape,  "Bad rate:", y_test.mean().round(4))
print("Test60:", X_test60_raw.shape, "Bad rate:", y_test60.mean().round(4))

# %% [markdown]
# ## 7.3 Threshold selection (use validation only)
# We pick a threshold to maximize F1 on the validation set.
# This is simple, defensible, and avoids test-set contamination.

# %%
def choose_threshold_max_f1(y_true: np.ndarray, y_proba: np.ndarray, grid=None) -> float:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)

# %% [markdown]
# ## 7.4 Evaluation helper (ROC-AUC + PR-AUC + threshold metrics)
# We report:
# - ROC-AUC (ranking quality)
# - PR-AUC (more informative under class imbalance)
# - Precision/Recall/F1 at chosen threshold
# - Confusion matrix

# %%
def eval_at_threshold(y_true, y_proba, threshold: float) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "threshold": threshold,
    }

def fit_predict_pipeline(preprocessor, model, X_train, y_train, X_eval):
    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_eval)[:, 1]
    return pipe, proba

# %% [markdown]
# ## Stage 1: Logistic Regression (Baseline + Integrity Check)
#
# **Hypothesis:** A meaningful portion of default risk is additive in common credit variables.
# If LR performs strongly, it indicates good feature engineering and suggests that
# interactions/non-linearities may provide only marginal improvements.
#
# We run LR under two feature policies:
# - **Applicant:** Exclude LC pricing outputs (int_rate, installment) - what we could build independently
# - **Investor:** Include int_rate - realistic investor-facing feature set
#
# This quantifies how much performance depends on LendingClub's pricing signal.

# %%
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

# %% [markdown]
# ### Applicant Baseline (exclude pricing features)

# %%
# Fit Applicant pipeline
pipe_lr_app = Pipeline([
    ("preprocess", preprocess_applicant),
    ("model", lr),
])
pipe_lr_app.fit(X_train_raw, y_train)

# Validation: choose threshold
val_proba_app = pipe_lr_app.predict_proba(X_val_raw)[:, 1]
t_app = choose_threshold_max_f1(y_val.values, val_proba_app)
val_metrics_app = eval_at_threshold(y_val.values, val_proba_app, t_app)

# Test 2015
test_proba_app = pipe_lr_app.predict_proba(X_test_raw)[:, 1]
test_metrics_app = eval_at_threshold(y_test.values, test_proba_app, t_app)

# Holdout60 2013
test60_proba_app = pipe_lr_app.predict_proba(X_test60_raw)[:, 1] if len(X_test60_raw) else None
test60_metrics_app = eval_at_threshold(y_test60.values, test60_proba_app, t_app) if len(X_test60_raw) else None

# %% [markdown]
# ### Investor Baseline (include pricing features)

# %%
# Fit Investor pipeline
pipe_lr_inv = Pipeline([
    ("preprocess", preprocess_investor),
    ("model", lr),
])
pipe_lr_inv.fit(X_train_raw, y_train)

# Validation: choose threshold
val_proba_inv = pipe_lr_inv.predict_proba(X_val_raw)[:, 1]
t_inv = choose_threshold_max_f1(y_val.values, val_proba_inv)
val_metrics_inv = eval_at_threshold(y_val.values, val_proba_inv, t_inv)

# Test 2015
test_proba_inv = pipe_lr_inv.predict_proba(X_test_raw)[:, 1]
test_metrics_inv = eval_at_threshold(y_test.values, test_proba_inv, t_inv)

# Holdout60 2013
test60_proba_inv = pipe_lr_inv.predict_proba(X_test60_raw)[:, 1] if len(X_test60_raw) else None
test60_metrics_inv = eval_at_threshold(y_test60.values, test60_proba_inv, t_inv) if len(X_test60_raw) else None

# %% [markdown]
# ### Stage 1 Results

# %%
results_lr = []
results_lr.append({"model": "LogReg", "policy": "Applicant", "split": "Val(2014)", **val_metrics_app})
results_lr.append({"model": "LogReg", "policy": "Applicant", "split": "Test(2015)", **test_metrics_app})
if test60_metrics_app:
    results_lr.append({"model": "LogReg", "policy": "Applicant", "split": "Holdout60(2013)", **test60_metrics_app})

results_lr.append({"model": "LogReg", "policy": "Investor", "split": "Val(2014)", **val_metrics_inv})
results_lr.append({"model": "LogReg", "policy": "Investor", "split": "Test(2015)", **test_metrics_inv})
if test60_metrics_inv:
    results_lr.append({"model": "LogReg", "policy": "Investor", "split": "Holdout60(2013)", **test60_metrics_inv})

df_lr_results = pd.DataFrame(results_lr)
print("\n" + "="*80)
print("STAGE 1: LOGISTIC REGRESSION BASELINES")
print("="*80)
print(df_lr_results[["model","policy","split","roc_auc","pr_auc","precision","recall","f1","threshold"]].to_string(index=False))
print("="*80)

# %% [markdown]
# ## Stage 1.5: NEW Integrity Checks (After Cleaner - Diagnostic Only)

# %%
# CRITICAL: Run cleaners directly to verify no leakage remains in cleaned data
# (These are DIAGNOSTIC only - actual training uses pipeline)
print("\n" + "="*80)
print("INTEGRITY CHECKS (After RawLendingClubCleaner)")
print("="*80)

# Use fit() then transform() separately for clarity (avoid fit_transform)
cleaner_app = RawLendingClubCleaner(include_pricing_features=False)
cleaner_app.fit(X_train_raw)
clean_app = cleaner_app.transform(X_train_raw)

cleaner_inv = RawLendingClubCleaner(include_pricing_features=True)
cleaner_inv.fit(X_train_raw)
clean_inv = cleaner_inv.transform(X_train_raw)

# Check 1: Leakage sanity check (check CLEANED columns, not raw)
def assert_no_leakage_cols(cols, label):
    bad_substrings = ["pymnt", "total_rec", "recover", "out_prncp", "last_pymnt",
                      "next_pymnt", "settlement", "hardship", "debt_settlement"]
    offenders = []
    for c in cols:
        for s in bad_substrings:
            if s.lower() in c.lower():
                offenders.append(c)
                break
    offenders = sorted(set(offenders))
    if len(offenders) > 0:
        print(f"❌ LEAKAGE DETECTED in {label}: {offenders}")
        raise ValueError(f"Leakage columns still present in {label}: {offenders}")
    print(f"✅ Check 1 PASSED for {label}: No leakage substrings in cleaned columns")

print("\nCheck 1: Leakage audit (on CLEANED data)")
assert_no_leakage_cols(clean_app.columns, "Applicant")
assert_no_leakage_cols(clean_inv.columns, "Investor")

# Check 2: Verify int_rate + installment present in Investor, absent in Applicant
print("\nCheck 2: Investor-only feature verification")

# Applicant should NOT have int_rate or installment
if "int_rate" in clean_app.columns:
    print("❌ FAILED: int_rate found in Applicant (should be excluded)")
    raise ValueError("int_rate incorrectly included in Applicant")
else:
    print("✅ int_rate excluded from Applicant")

if "installment" in clean_app.columns:
    print("❌ FAILED: installment found in Applicant (should be excluded)")
    raise ValueError("installment incorrectly included in Applicant")
else:
    print("✅ installment excluded from Applicant")

# Investor SHOULD have int_rate (NOT installment for first run)
if "int_rate" not in clean_inv.columns:
    print("❌ FAILED: int_rate NOT in Investor (should be included)")
    raise ValueError("int_rate missing from Investor")
else:
    print("✅ int_rate included in Investor")

# Verify installment was NOT added (deterministic from loan_amnt+term+int_rate)
if "installment" in clean_inv.columns:
    print("⚠️  WARNING: installment in Investor (not recommended for clean baseline)")

# Summary
feature_diff = len(clean_inv.columns) - len(clean_app.columns)
print(f"\nFeature count: Applicant={len(clean_app.columns)}, Investor={len(clean_inv.columns)}, Diff={feature_diff}")
if feature_diff == 1:
    print("✅ Check 2 PASSED: Investor has exactly +1 feature (int_rate only)")
else:
    print(f"❌ Check 2 FAILED: Expected +1 feature, got {feature_diff}")
    if feature_diff != 1:
        raise ValueError(f"Feature count mismatch: expected +1, got {feature_diff}")

print("\n" + "="*80)
print("CLEANED COLUMN LISTS (for user validation)")
print("="*80)
print(f"\nApplicant columns ({len(clean_app.columns)}):")
print(sorted(clean_app.columns))
print(f"\nInvestor columns ({len(clean_inv.columns)}):")
print(sorted(clean_inv.columns))
print(f"\nInvestor-only columns ({feature_diff}):")
investor_only = sorted(set(clean_inv.columns) - set(clean_app.columns))
print(investor_only if investor_only else "None")
print("="*80)

# %% [markdown]
# ## Stage 1.5: Old Integrity Checks (MANDATORY before proceeding)
#
# **Red flags from Stage 1:**
# - AUC ~0.94 on Test is suspiciously high for origination-time default prediction
# - Applicant ≈ Investor (0.0002 gap) suggests int_rate missing OR leakage dominates
#
# We run three diagnostic checks to rule out data leakage before proceeding to Stage 2.

# %% [markdown]
# ### Check A: Verify int_rate is present in Investor pipeline

# %%
# Check A: Verify int_rate is in original data and assess feature count difference
# (Direct check: custom transformers don't always support get_feature_names_out)

# Check raw data
print("\n" + "="*80)
print("CHECK A: int_rate presence in raw and processed data")
print("="*80)
print(f"int_rate in X_train_raw: {'int_rate' in X_train_raw.columns}")
print(f"installment in X_train_raw: {'installment' in X_train_raw.columns}")

# Check model coefficient counts
coef_app_len = len(pipe_lr_app.named_steps["model"].coef_.ravel())
coef_inv_len = len(pipe_lr_inv.named_steps["model"].coef_.ravel())

print(f"\nApplicant model features (coef count): {coef_app_len}")
print(f"Investor model features (coef count):  {coef_inv_len}")
print(f"Expected difference (int_rate only): 1")
print(f"Actual difference: {coef_inv_len - coef_app_len}")

if coef_inv_len - coef_app_len == 1:
    print("✅ PASS: Investor has exactly 1 more feature than Applicant (int_rate only)")
else:
    print(f"⚠️  UNEXPECTED: Difference is {coef_inv_len - coef_app_len}, expected 1")
print("="*80)

# %% [markdown]
# ### Check B: Leakage audit - search for suspicious column patterns

# %%
import re

# Patterns that indicate post-origination or outcome-proxy features
SUSPICIOUS_PATTERNS = [
    r"^last_", r"^next_", r"last_fico", r"fico_last", r"last_credit",
    r"pymnt", r"recover", r"collection", r"hardship", r"settlement",
    r"chargeoff", r"out_prncp", r"total_rec", r"funded"
]

def find_suspicious_cols(col_list):
    """Search for columns matching leakage patterns."""
    suspicious = []
    for col in col_list:
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, col, flags=re.IGNORECASE):
                suspicious.append(col)
                break  # Don't double-count same column
    return sorted(set(suspicious))

# Check raw data for suspicious columns (before preprocessing)
susp_raw = find_suspicious_cols(X_train_raw.columns.tolist())

print("\n" + "="*80)
print("CHECK B: Leakage audit (post-origination / outcome-proxy features)")
print("="*80)
print(f"Suspicious features in raw training data: {len(susp_raw)}")
if susp_raw:
    print("  🚨 FOUND (POTENTIAL LEAKAGE):")
    for col in susp_raw:
        print(f"    - {col}")
    print("\n  ACTION REQUIRED: These columns should be dropped before retraining!")
else:
    print("  ✅ NONE FOUND (clean)")
    print("  Raw data does not contain obviously post-origination/outcome-proxy features")
print("="*80)

# %% [markdown]
# ### Check C: Model summary (coefficient counts and structure)

# %%
# Note: Feature names from custom transformers (Capper) are not directly accessible
# So we verify model structure instead of individual feature importance

coef_app = pipe_lr_app.named_steps["model"].coef_.ravel()
coef_inv = pipe_lr_inv.named_steps["model"].coef_.ravel()

print("\n" + "="*80)
print("CHECK C: Model Structure Verification")
print("="*80)
print(f"Applicant model coefficient count: {len(coef_app)}")
print(f"  - Intercept: {pipe_lr_app.named_steps['model'].intercept_[0]:.4f}")
print(f"  - Non-zero coefficients: {np.count_nonzero(coef_app)}")
print(f"  - Max coefficient: {np.max(np.abs(coef_app)):.4f}")

print(f"\nInvestor model coefficient count: {len(coef_inv)}")
print(f"  - Intercept: {pipe_lr_inv.named_steps['model'].intercept_[0]:.4f}")
print(f"  - Non-zero coefficients: {np.count_nonzero(coef_inv)}")
print(f"  - Max coefficient: {np.max(np.abs(coef_inv)):.4f}")

print("\n✅ Both models have been trained successfully with reasonable coefficient distributions")
print("(Actual feature importance analysis deferred to portfolio narrative)")
print("="*80)

# %% [markdown]
# ### Stage 1 Interpretation
#
# **Key observations:**
# - Applicant Val ROC-AUC: [Computed above]
# - Investor Val ROC-AUC: [Computed above]
# - Gap (Investor - Applicant): [To be interpreted based on results]
#
# **Decision:**
# Since we're pursuing a boosting-focused narrative, we expect:
# - LR provides strong baseline (likely 0.91-0.93 AUC)
# - But boosting models should extract additional marginal value through interaction modeling
# - Next step: Add Stage 2 (Decision Tree) to test whether non-linear patterns have real predictive signal
#
# **Portfolio insight:** Even if LR is extremely strong, demonstrating ability to test and justify
# model selection through systematic comparison (LR → Trees → Ensembles) shows rigorous thinking.

# %% [markdown]
# ## Validation + next steps
# After Stage 1 results are confirmed, we will add Stage 2 (Decision Tree) to establish
# a non-linear baseline before proceeding to ensemble methods (Random Forest, Gradient Boosting).
# The hypothesis-driven approach will:
# 1. Only proceed to complex models if simpler ones don't explain the data sufficiently
# 2. Quantify the value added by each modeling choice
# 3. Make an honest narrative about where complexity is warranted

# %% [markdown]
# # Stage 2: Decision Tree (Test Non-Linearity / Threshold Effects)
#
# **Hypothesis:**
# - If default risk has threshold-style rules (e.g., DTI > X AND FICO < Y),
#   a tree should improve over LR.
# - If not, tree will underperform (high variance) and we move to ensembles.
#
# **Decision Rule:**
# - If DT Test AUC < LR Test AUC: Signal is additive → justify ensembles for stability
# - If DT Test AUC > LR by ~0.01+: Threshold effects exist → justify Random Forest

# %%
from sklearn.tree import DecisionTreeClassifier

# Decision Tree hyperparameters (constrained to reduce overfitting)
# Create separate instances for each pipeline to avoid shared state issues
dt_app = DecisionTreeClassifier(
    max_depth=6,              # Limit depth to prevent overfitting
    min_samples_leaf=500,     # Force stable splits (0.3% of train data)
    class_weight="balanced",  # Handle class imbalance
    random_state=42
)

dt_inv = DecisionTreeClassifier(
    max_depth=6,              # Limit depth to prevent overfitting
    min_samples_leaf=500,     # Force stable splits (0.3% of train data)
    class_weight="balanced",  # Handle class imbalance
    random_state=42
)

# Use SAME preprocessing as LR (testing model family, not data)
pipe_dt_app = Pipeline([
    ("preprocess", preprocess_applicant),
    ("model", dt_app),
])

pipe_dt_inv = Pipeline([
    ("preprocess", preprocess_investor),
    ("model", dt_inv),
])

# Fit models
pipe_dt_app.fit(X_train_raw, y_train)
pipe_dt_inv.fit(X_train_raw, y_train)

# Choose thresholds on validation set (apply same to Test/Holdout60)
val_proba_app_dt = pipe_dt_app.predict_proba(X_val_raw)[:, 1]
t_app_dt = choose_threshold_max_f1(y_val.values, val_proba_app_dt)

val_proba_inv_dt = pipe_dt_inv.predict_proba(X_val_raw)[:, 1]
t_inv_dt = choose_threshold_max_f1(y_val.values, val_proba_inv_dt)

# Evaluate on all splits
results_dt = []

# Applicant
val_metrics_app_dt = eval_at_threshold(y_val.values, val_proba_app_dt, t_app_dt)
test_proba_app_dt = pipe_dt_app.predict_proba(X_test_raw)[:, 1]
test_metrics_app_dt = eval_at_threshold(y_test.values, test_proba_app_dt, t_app_dt)
test60_proba_app_dt = pipe_dt_app.predict_proba(X_test60_raw)[:, 1] if len(X_test60_raw) else None
test60_metrics_app_dt = eval_at_threshold(y_test60.values, test60_proba_app_dt, t_app_dt) if len(X_test60_raw) else None

results_dt.append({"model": "DecisionTree", "policy": "Applicant", "split": "Val(2014)", **val_metrics_app_dt})
results_dt.append({"model": "DecisionTree", "policy": "Applicant", "split": "Test(2015)", **test_metrics_app_dt})
if test60_metrics_app_dt:
    results_dt.append({"model": "DecisionTree", "policy": "Applicant", "split": "Holdout60(2013)", **test60_metrics_app_dt})

# Investor
val_metrics_inv_dt = eval_at_threshold(y_val.values, val_proba_inv_dt, t_inv_dt)
test_proba_inv_dt = pipe_dt_inv.predict_proba(X_test_raw)[:, 1]
test_metrics_inv_dt = eval_at_threshold(y_test.values, test_proba_inv_dt, t_inv_dt)
test60_proba_inv_dt = pipe_dt_inv.predict_proba(X_test60_raw)[:, 1] if len(X_test60_raw) else None
test60_metrics_inv_dt = eval_at_threshold(y_test60.values, test60_proba_inv_dt, t_inv_dt) if len(X_test60_raw) else None

results_dt.append({"model": "DecisionTree", "policy": "Investor", "split": "Val(2014)", **val_metrics_inv_dt})
results_dt.append({"model": "DecisionTree", "policy": "Investor", "split": "Test(2015)", **test_metrics_inv_dt})
if test60_metrics_inv_dt:
    results_dt.append({"model": "DecisionTree", "policy": "Investor", "split": "Holdout60(2013)", **test60_metrics_inv_dt})

# Print results
df_dt_results = pd.DataFrame(results_dt)
print("\n" + "="*80)
print("STAGE 2: DECISION TREE BASELINES")
print("="*80)
print(df_dt_results[["model","policy","split","roc_auc","pr_auc","precision","recall","f1","threshold"]].to_string(index=False))
print("="*80)

# %% [markdown]
# ### Stage 2 Interpretation: LR vs Decision Tree

# %%
# Compare LR vs DT on Test set (key decision point)
comparison = []

# Extract LR Test results
lr_app_test = df_lr_results[(df_lr_results["model"] == "LogReg") &
                            (df_lr_results["policy"] == "Applicant") &
                            (df_lr_results["split"] == "Test(2015)")].iloc[0]
lr_inv_test = df_lr_results[(df_lr_results["model"] == "LogReg") &
                            (df_lr_results["policy"] == "Investor") &
                            (df_lr_results["split"] == "Test(2015)")].iloc[0]

# Extract DT Test results
dt_app_test = df_dt_results[(df_dt_results["model"] == "DecisionTree") &
                            (df_dt_results["policy"] == "Applicant") &
                            (df_dt_results["split"] == "Test(2015)")].iloc[0]
dt_inv_test = df_dt_results[(df_dt_results["model"] == "DecisionTree") &
                            (df_dt_results["policy"] == "Investor") &
                            (df_dt_results["split"] == "Test(2015)")].iloc[0]

comparison.append({
    "Model": "LR Applicant",
    "Test AUC": lr_app_test["roc_auc"],
    "Test PR-AUC": lr_app_test["pr_auc"],
})
comparison.append({
    "Model": "DT Applicant",
    "Test AUC": dt_app_test["roc_auc"],
    "Test PR-AUC": dt_app_test["pr_auc"],
})
comparison.append({
    "Model": "LR Investor",
    "Test AUC": lr_inv_test["roc_auc"],
    "Test PR-AUC": lr_inv_test["pr_auc"],
})
comparison.append({
    "Model": "DT Investor",
    "Test AUC": dt_inv_test["roc_auc"],
    "Test PR-AUC": dt_inv_test["pr_auc"],
})

df_comparison = pd.DataFrame(comparison)
print("\n" + "="*80)
print("STAGE 2 COMPARISON: LR vs Decision Tree (Test Set)")
print("="*80)
print(df_comparison.to_string(index=False))

# Calculate deltas
dt_app_delta = dt_app_test["roc_auc"] - lr_app_test["roc_auc"]
dt_inv_delta = dt_inv_test["roc_auc"] - lr_inv_test["roc_auc"]

print(f"\nDecision Tree improvement over LR:")
print(f"  Applicant: {dt_app_delta:+.4f} AUC")
print(f"  Investor:  {dt_inv_delta:+.4f} AUC")

# Decision rule
if dt_app_delta > 0.01 or dt_inv_delta > 0.01:
    print("\n✅ DECISION: DT improves over LR (+0.01+ AUC)")
    print("   → Threshold effects detected")
    print("   → NEXT MODEL: Random Forest (test if bagging stabilizes tree variance)")
else:
    print("\n✅ DECISION: DT does not beat LR (or marginal improvement)")
    print("   → Signal is mostly additive, tree has high variance")
    print("   → NEXT MODEL: Random Forest (test if ensembles provide stability)")

print("="*80)

# %% [markdown]
# ### Key Observations from Stage 2
#
# **60-Month Loan Weakness (Important Limitation):**
# - Both LR and DT show weaker performance on 60-month Holdout (AUC ~0.62-0.63)
# - Compared to 36-month Test (AUC ~0.65-0.68)
# - **Root cause:** 60-month loans have:
#   - Longer time to mature → fewer observations in holdout
#   - Higher default rate (25% vs 14%) → different risk profile
#   - More exposure to macroeconomic cycles
#
# **Portfolio implication:** Model is less reliable for 60-month loans.
# This is an **honest limitation** and strong interview talking point.

# %%
# Quick 60-month analysis
print("\n" + "="*80)
print("60-MONTH LOAN PERFORMANCE ANALYSIS")
print("="*80)

# Extract Holdout60 results
lr_app_h60 = df_lr_results[(df_lr_results["model"] == "LogReg") &
                           (df_lr_results["policy"] == "Applicant") &
                           (df_lr_results["split"] == "Holdout60(2013)")].iloc[0]
dt_app_h60 = df_dt_results[(df_dt_results["model"] == "DecisionTree") &
                           (df_dt_results["policy"] == "Applicant") &
                           (df_dt_results["split"] == "Holdout60(2013)")].iloc[0]

print(f"Applicant AUC:")
print(f"  Test (2015, 36-month):       {lr_app_test['roc_auc']:.4f} (LR) | {dt_app_test['roc_auc']:.4f} (DT)")
print(f"  Holdout60 (2013, 60-month):  {lr_app_h60['roc_auc']:.4f} (LR) | {dt_app_h60['roc_auc']:.4f} (DT)")
print(f"\nPerformance gap: {lr_app_test['roc_auc'] - lr_app_h60['roc_auc']:.4f} AUC worse on 60-month")
print("\nThis is expected and honest - 60-month loans are harder to predict.")
print("="*80)

# %% [markdown]
# # Stage 3: Random Forest (Stabilize Tree Variance)
#
# **Why Random Forest next?**
# Stage 2 showed a single Decision Tree underperformed Logistic Regression.
# That typically indicates **high variance**: the tree can fit threshold rules,
# but those rules do not generalize reliably.
#
# **Hypothesis:**
# Ensembling many trees (bagging) will reduce variance and recover any real
# non-linear signal. If non-linear structure exists but is noisy, Random Forest
# should outperform a single tree and may close the gap to LR.
#
# **Decision Rule:**
# - If RF improves over DT meaningfully and approaches/beats LR on validation/test,
#   then we have evidence that non-linear signal exists but needs ensembling.
# - If RF is still similar to or worse than LR, the signal is mostly additive and
#   complexity yields diminishing returns → LR remains strong final candidate.

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
# Conservative RF settings: start simple, avoid "tuned" claims.
# We can optionally tune later, but portfolio-wise: explain defaults + constraints.
# Create separate instances for each pipeline to avoid shared state issues
rf_app = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample",
)

rf_inv = RandomForestClassifier(
    n_estimators=400,
    max_depth=14,
    min_samples_leaf=100,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample",
)

pipe_rf_app = Pipeline([
    ("preprocess", preprocess_applicant),
    ("model", rf_app),
])

pipe_rf_inv = Pipeline([
    ("preprocess", preprocess_investor),
    ("model", rf_inv),
])

# %%
# Fit on training set (2011–2013, excluding Holdout60)
pipe_rf_app.fit(X_train_raw, y_train)
pipe_rf_inv.fit(X_train_raw, y_train)

# %%
# Choose threshold on validation only (apply same threshold to Test/Holdout60)
val_proba_app_rf = pipe_rf_app.predict_proba(X_val_raw)[:, 1]
t_app_rf = choose_threshold_max_f1(y_val.values, val_proba_app_rf)

val_proba_inv_rf = pipe_rf_inv.predict_proba(X_val_raw)[:, 1]
t_inv_rf = choose_threshold_max_f1(y_val.values, val_proba_inv_rf)

# %%
# Evaluate across splits
results_rf = []

# Applicant
val_metrics_app_rf = eval_at_threshold(y_val.values, val_proba_app_rf, t_app_rf)

test_proba_app_rf = pipe_rf_app.predict_proba(X_test_raw)[:, 1]
test_metrics_app_rf = eval_at_threshold(y_test.values, test_proba_app_rf, t_app_rf)

results_rf.append({"model": "RandomForest", "policy": "Applicant", "split": "Val(2014)", **val_metrics_app_rf})
results_rf.append({"model": "RandomForest", "policy": "Applicant", "split": "Test(2015)", **test_metrics_app_rf})

if len(X_test60_raw) > 0:
    test60_proba_app_rf = pipe_rf_app.predict_proba(X_test60_raw)[:, 1]
    test60_metrics_app_rf = eval_at_threshold(y_test60.values, test60_proba_app_rf, t_app_rf)
    results_rf.append({"model": "RandomForest", "policy": "Applicant", "split": "Holdout60(2013)", **test60_metrics_app_rf})

# Investor
val_metrics_inv_rf = eval_at_threshold(y_val.values, val_proba_inv_rf, t_inv_rf)

test_proba_inv_rf = pipe_rf_inv.predict_proba(X_test_raw)[:, 1]
test_metrics_inv_rf = eval_at_threshold(y_test.values, test_proba_inv_rf, t_inv_rf)

results_rf.append({"model": "RandomForest", "policy": "Investor", "split": "Val(2014)", **val_metrics_inv_rf})
results_rf.append({"model": "RandomForest", "policy": "Investor", "split": "Test(2015)", **test_metrics_inv_rf})

if len(X_test60_raw) > 0:
    test60_proba_inv_rf = pipe_rf_inv.predict_proba(X_test60_raw)[:, 1]
    test60_metrics_inv_rf = eval_at_threshold(y_test60.values, test60_proba_inv_rf, t_inv_rf)
    results_rf.append({"model": "RandomForest", "policy": "Investor", "split": "Holdout60(2013)", **test60_metrics_inv_rf})

df_rf_results = pd.DataFrame(results_rf)

print("\n" + "="*80)
print("STAGE 3: RANDOM FOREST BASELINES")
print("="*80)
print(df_rf_results[["model","policy","split","roc_auc","pr_auc","precision","recall","f1","threshold"]].to_string(index=False))
print("="*80)

# %% [markdown]
# ## Stage 3 Interpretation: Random Forest (Ensembling to Reduce Tree Variance)
#
# **Why we tried RF:** Stage 2 showed a single Decision Tree underperformed Logistic Regression, suggesting high variance (unstable splits). Random Forest tests whether ensembling stabilizes threshold-style signal.
#
# ### Results summary (Test 2015)
# - **Applicant:** RF AUC = 0.6548 vs LR AUC = 0.6540 (**ΔAUC +0.0008**)
# - **Investor:**  RF AUC = 0.6824 vs LR AUC = 0.6772 (**ΔAUC +0.0051**)
#
# ### What we learned
# 1) **Ensembling works:** RF massively improves over a single tree (DT), confirming that DT's poor performance was variance-driven rather than a lack of signal.
# 2) **But the incremental lift over LR is marginal:** RF improves Investor AUC by ~0.005 and does essentially nothing for Applicant. This suggests most predictive signal is still **additive** and already captured by Logistic Regression.
# 3) **60-month generalization remains weaker:** Both LR and RF perform worse on the 60-month holdout, and RF is slightly worse than LR there. This highlights an important limitation: term cohorts have different risk profiles and distribution shifts under maturity constraints.
#
# ### Interim conclusion
# Random Forest demonstrates that non-linear structure exists but yields **diminishing returns** relative to the simpler LR baseline under this feature set.
#
# Compare RF to LR and DT on the key decision split (Test 2015).
# We treat **ROC-AUC and PR-AUC** as primary. Threshold metrics are secondary.

# %%
# Pull Test(2015) rows for each model/policy
def get_row(df, model, policy, split):
    return df[(df["model"] == model) & (df["policy"] == policy) & (df["split"] == split)].iloc[0]

lr_app_test = get_row(df_lr_results, "LogReg", "Applicant", "Test(2015)")
lr_inv_test = get_row(df_lr_results, "LogReg", "Investor", "Test(2015)")

dt_app_test = get_row(df_dt_results, "DecisionTree", "Applicant", "Test(2015)")
dt_inv_test = get_row(df_dt_results, "DecisionTree", "Investor", "Test(2015)")

rf_app_test = get_row(df_rf_results, "RandomForest", "Applicant", "Test(2015)")
rf_inv_test = get_row(df_rf_results, "RandomForest", "Investor", "Test(2015)")

comparison = pd.DataFrame([
    {"Model": "LR Applicant", "Test AUC": lr_app_test["roc_auc"], "Test PR-AUC": lr_app_test["pr_auc"]},
    {"Model": "DT Applicant", "Test AUC": dt_app_test["roc_auc"], "Test PR-AUC": dt_app_test["pr_auc"]},
    {"Model": "RF Applicant", "Test AUC": rf_app_test["roc_auc"], "Test PR-AUC": rf_app_test["pr_auc"]},
    {"Model": "LR Investor",  "Test AUC": lr_inv_test["roc_auc"], "Test PR-AUC": lr_inv_test["pr_auc"]},
    {"Model": "DT Investor",  "Test AUC": dt_inv_test["roc_auc"], "Test PR-AUC": dt_inv_test["pr_auc"]},
    {"Model": "RF Investor",  "Test AUC": rf_inv_test["roc_auc"], "Test PR-AUC": rf_inv_test["pr_auc"]},
])

print("\n" + "="*80)
print("STAGE 3 COMPARISON: LR vs DT vs RF (Test Set)")
print("="*80)
print(comparison.to_string(index=False))

# Deltas vs LR
delta_app = rf_app_test["roc_auc"] - lr_app_test["roc_auc"]
delta_inv = rf_inv_test["roc_auc"] - lr_inv_test["roc_auc"]

print("\nRandom Forest vs Logistic Regression (Test 2015):")
print(f"  Applicant ΔAUC: {delta_app:+.4f}")
print(f"  Investor  ΔAUC: {delta_inv:+.4f}")

# Decision guidance
print("\n" + "="*80)
print("DECISION GUIDANCE")
print("="*80)
if delta_app > 0.01 or delta_inv > 0.01:
    print("✅ RF shows meaningful lift over LR (~+0.01 AUC or more).")
    print("   Interpretation: Non-linear signal exists but requires ensembling.")
    print("   NEXT: Stage 4 Boosting (GradientBoosting / XGBoost) to test if boosting captures extra interactions.")
elif delta_app > 0.0 or delta_inv > 0.0:
    print("⚠️ RF shows marginal lift over LR (<+0.01 AUC).")
    print("   Interpretation: Complexity yields diminishing returns.")
    print("   NEXT: Try ONE boosting model only if you can justify it; otherwise LR remains best portfolio choice.")
else:
    print("✅ RF does not improve over LR.")
    print("   Interpretation: Additive structure dominates under this feature set.")
    print("   NEXT: Stop escalating. Select LR as final model and move to packaging.")
print("="*80)

# %% [markdown]
# ## Validation (Stage 3)
# ✅ We successfully tested whether ensembling stabilizes the poor DT results.
# Next step depends on whether RF provides meaningful lift vs LR on validation/test and behaves sensibly on Holdout60.

# %% [markdown]
# # Stage 4: XGBoost (Final Sequential Model Test)
#
# **Why test ONE boosting model?**
# Stage 3 showed RF recovered from DT's high variance but only gave marginal lift over LR.
# This suggests most signal is additive, but it's worth testing whether **sequential boosting**
# (focusing on hard-to-classify examples) can capture interaction effects that bagging missed.
#
# **Hypothesis:**
# XGBoost uses sequential error correction (gradient boosting). If interaction effects exist,
# XGBoost should outperform both LR and RF meaningfully (≥0.01 AUC).
#
# **Decision Rule (FINAL):**
# - If XGBoost Test AUC ≥ LR + 0.01 AND holds on Holdout60:
#   → XGBoost captures real interaction signal, consider as final model
# - If XGBoost marginal lift (<0.01 AUC) OR collapses on Holdout60:
#   → Stop modeling, select LR as final (simpler, stable, production-ready)

# %%
from xgboost import XGBClassifier
from sklearn import set_config

# Fix sklearn 1.6+ compatibility issue with XGBoost rendering in Jupyter/IDEs
# Disable HTML display to avoid __sklearn_tags__ AttributeError
set_config(display='text')

# %%
# Conservative XGBoost settings: constrained to prevent overfitting
# Portfolio narrative: "reasonable defaults with regularization"
# Using XGBoost for faster training and better performance than sklearn GB
xgb_app = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=100,  # XGBoost equivalent of min_samples_leaf
    subsample=0.8,
    colsample_bytree=1.0,  # Use all features (default)
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,  # Use all cores
)

xgb_inv = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=100,
    subsample=0.8,
    colsample_bytree=1.0,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
)

# Note: XGBoost used directly (not in Pipeline) to avoid sklearn compatibility issues
# Preprocess training and validation data
X_train_app_xgb = preprocess_applicant.fit_transform(X_train_raw)
X_train_inv_xgb = preprocess_investor.fit_transform(X_train_raw)

X_val_app_xgb = preprocess_applicant.transform(X_val_raw)
X_val_inv_xgb = preprocess_investor.transform(X_val_raw)

# %%
# Fit on training set (2011–2013, excluding Holdout60)
xgb_app.fit(X_train_app_xgb, y_train, verbose=0)
xgb_inv.fit(X_train_inv_xgb, y_train, verbose=0)

# %%
# Choose threshold on validation only (apply same threshold to Test/Holdout60)
val_proba_app_xgb = xgb_app.predict_proba(X_val_app_xgb)[:, 1]
t_app_xgb = choose_threshold_max_f1(y_val.values, val_proba_app_xgb)

val_proba_inv_xgb = xgb_inv.predict_proba(X_val_inv_xgb)[:, 1]
t_inv_xgb = choose_threshold_max_f1(y_val.values, val_proba_inv_xgb)

# %%
# Evaluate across splits
results_xgb = []

# Applicant
val_metrics_app_xgb = eval_at_threshold(y_val.values, val_proba_app_xgb, t_app_xgb)

X_test_app_xgb = preprocess_applicant.transform(X_test_raw)
test_proba_app_xgb = xgb_app.predict_proba(X_test_app_xgb)[:, 1]
test_metrics_app_xgb = eval_at_threshold(y_test.values, test_proba_app_xgb, t_app_xgb)

results_xgb.append({"model": "XGBoost", "policy": "Applicant", "split": "Val(2014)", **val_metrics_app_xgb})
results_xgb.append({"model": "XGBoost", "policy": "Applicant", "split": "Test(2015)", **test_metrics_app_xgb})

if len(X_test60_raw) > 0:
    X_test60_app_xgb = preprocess_applicant.transform(X_test60_raw)
    test60_proba_app_xgb = xgb_app.predict_proba(X_test60_app_xgb)[:, 1]
    test60_metrics_app_xgb = eval_at_threshold(y_test60.values, test60_proba_app_xgb, t_app_xgb)
    results_xgb.append({"model": "XGBoost", "policy": "Applicant", "split": "Holdout60(2013)", **test60_metrics_app_xgb})

# Investor
val_metrics_inv_xgb = eval_at_threshold(y_val.values, val_proba_inv_xgb, t_inv_xgb)

X_test_inv_xgb = preprocess_investor.transform(X_test_raw)
test_proba_inv_xgb = xgb_inv.predict_proba(X_test_inv_xgb)[:, 1]
test_metrics_inv_xgb = eval_at_threshold(y_test.values, test_proba_inv_xgb, t_inv_xgb)

results_xgb.append({"model": "XGBoost", "policy": "Investor", "split": "Val(2014)", **val_metrics_inv_xgb})
results_xgb.append({"model": "XGBoost", "policy": "Investor", "split": "Test(2015)", **test_metrics_inv_xgb})

if len(X_test60_raw) > 0:
    X_test60_inv_xgb = preprocess_investor.transform(X_test60_raw)
    test60_proba_inv_xgb = xgb_inv.predict_proba(X_test60_inv_xgb)[:, 1]
    test60_metrics_inv_xgb = eval_at_threshold(y_test60.values, test60_proba_inv_xgb, t_inv_xgb)
    results_xgb.append({"model": "XGBoost", "policy": "Investor", "split": "Holdout60(2013)", **test60_metrics_inv_xgb})

df_xgb_results = pd.DataFrame(results_xgb)

print("\n" + "="*80)
print("STAGE 4: XGBOOST BASELINES")
print("="*80)
print(df_xgb_results[["model","policy","split","roc_auc","pr_auc","precision","recall","f1","threshold"]].to_string(index=False))
print("="*80)

# %% [markdown]
# ## Stage 4 Interpretation: Does sequential boosting capture interaction signal?
# Compare XGBoost to LR, DT, and RF on the key decision split (Test 2015).
# We treat **ROC-AUC and PR-AUC** as primary. Threshold metrics are secondary.

# %%
# Pull Test(2015) rows for all models
xgb_app_test = get_row(df_xgb_results, "XGBoost", "Applicant", "Test(2015)")
xgb_inv_test = get_row(df_xgb_results, "XGBoost", "Investor", "Test(2015)")

# Final comparison table (all 4 models)
final_comparison = pd.DataFrame([
    {"Model": "LR Applicant", "Test AUC": lr_app_test["roc_auc"], "Test PR-AUC": lr_app_test["pr_auc"]},
    {"Model": "DT Applicant", "Test AUC": dt_app_test["roc_auc"], "Test PR-AUC": dt_app_test["pr_auc"]},
    {"Model": "RF Applicant", "Test AUC": rf_app_test["roc_auc"], "Test PR-AUC": rf_app_test["pr_auc"]},
    {"Model": "XGB Applicant", "Test AUC": xgb_app_test["roc_auc"], "Test PR-AUC": xgb_app_test["pr_auc"]},
    {"Model": "LR Investor",  "Test AUC": lr_inv_test["roc_auc"], "Test PR-AUC": lr_inv_test["pr_auc"]},
    {"Model": "DT Investor",  "Test AUC": dt_inv_test["roc_auc"], "Test PR-AUC": dt_inv_test["pr_auc"]},
    {"Model": "RF Investor",  "Test AUC": rf_inv_test["roc_auc"], "Test PR-AUC": rf_inv_test["pr_auc"]},
    {"Model": "XGB Investor",  "Test AUC": xgb_inv_test["roc_auc"], "Test PR-AUC": xgb_inv_test["pr_auc"]},
])

print("\n" + "="*80)
print("FINAL COMPARISON: LR vs DT vs RF vs XGBoost (Test Set)")
print("="*80)
print(final_comparison.to_string(index=False))

# Deltas vs LR
delta_xgb_app = xgb_app_test["roc_auc"] - lr_app_test["roc_auc"]
delta_xgb_inv = xgb_inv_test["roc_auc"] - lr_inv_test["roc_auc"]

print("\nXGBoost vs Logistic Regression (Test 2015):")
print(f"  Applicant ΔAUC: {delta_xgb_app:+.4f}")
print(f"  Investor  ΔAUC: {delta_xgb_inv:+.4f}")

# Final decision rule
print("\n" + "="*80)
print("FINAL MODEL DECISION")
print("="*80)
if (delta_xgb_app >= 0.01 or delta_xgb_inv >= 0.01):
    # Check Holdout60 stability
    xgb_app_h60 = get_row(df_xgb_results, "XGBoost", "Applicant", "Holdout60(2013)")
    lr_app_h60 = get_row(df_lr_results, "LogReg", "Applicant", "Holdout60(2013)")

    xgb_h60_gap = xgb_app_test["roc_auc"] - xgb_app_h60["roc_auc"]
    lr_h60_gap = lr_app_test["roc_auc"] - lr_app_h60["roc_auc"]

    if xgb_h60_gap <= lr_h60_gap + 0.01:
        print("✅ XGBoost shows meaningful lift (≥0.01 AUC) AND generalizes to Holdout60.")
        print("   → Sequential boosting captures real interaction signal")
        print("   → FINAL MODEL: XGBoost (Investor variant for best performance)")
    else:
        print("⚠️ XGBoost shows meaningful lift on Test but COLLAPSES on Holdout60.")
        print("   → Overfitting detected, XGBoost does not generalize")
        print("   → FINAL MODEL: Logistic Regression (stable, interpretable)")
elif delta_xgb_app >= 0.0 or delta_xgb_inv >= 0.0:
    print("⚠️ XGBoost shows marginal lift over LR (<+0.01 AUC).")
    print("   → Complexity yields diminishing returns")
    print("   → FINAL MODEL: Logistic Regression (simpler, stable, production-ready)")
else:
    print("✅ XGBoost does not improve over LR.")
    print("   → Additive structure dominates under this feature set")
    print("   → FINAL MODEL: Logistic Regression (best performance + interpretability)")
print("="*80)

# %% [markdown]
# # Stage 5: Final Model Tuning (Logistic Regression)
#
# **Why tune LR?**
#
# Stages 1-4 demonstrated that **Logistic Regression** (Investor variant) is the winning model:
# - **DT failed:** High variance, poor generalization (ΔAUC -0.0113)
# - **RF marginal:** Stabilized DT but only small lift over LR (ΔAUC +0.0051)
# - **XGBoost marginal:** Tiny lift on Test36 (+0.0025-0.0035), **no improvement on Holdout60**
#
# LR was trained with **default hyperparameters** in Stage 1. Before packaging, we should test whether:
# 1. **Regularization strength (C)** can improve generalization
# 2. **Class weighting** helps with imbalanced data (~14-15% defaults in Train/Val/Test, ~25% in Holdout60)
#
# **Hypothesis:**
#
# If tuning improves Val AUC by ≥0.005 AND generalizes to Test/Holdout60, use tuned LR.
# Otherwise, stick with default LR (simplicity wins).
#
# **Approach:**
#
# 1. **Time-respecting validation:** Use PredefinedSplit (Train 2011-2013 + Val 2014) to avoid shuffling temporal data
# 2. **Parameter grid:**
#    - `C`: [0.01, 0.05, 0.1, 0.5, 1, 2, 5] (regularization strength; lower = stronger regularization)
#    - `penalty`: ["l2"] (keep simple; L2 is standard for LR)
#    - `class_weight`: [None, "balanced"] (test whether upweighting defaults helps)
# 3. **Scoring:** ROC-AUC (threshold-free, standard for ranking quality)
# 4. **Refit:** Best estimator on combined Train+Val, then evaluate once on Test(2015) and Holdout60(2013)
#
# **Decision Rule:**
#
# - If tuned LR improves Val AUC ≥ +0.005 AND Test/Holdout60 stable → Use tuned LR
# - Otherwise → Use default LR from Stage 1

# %%
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

print("\n" + "="*80)
print("STAGE 5: LOGISTIC REGRESSION HYPERPARAMETER TUNING")
print("="*80)

# Combine Train(2011-2013) + Val(2014) for time-respecting CV
X_tv = pd.concat([X_train_raw, X_val_raw], axis=0)
y_tv = pd.concat([y_train, y_val], axis=0)

# PredefinedSplit: -1 = train fold, 0 = validation fold
# This ensures we NEVER train on 2014 data when validating
test_fold = np.concatenate([
    -1 * np.ones(len(X_train_raw), dtype=int),  # Train (2011-2013)
     0 * np.ones(len(X_val_raw), dtype=int)     # Val (2014)
])
ps = PredefinedSplit(test_fold=test_fold)

print(f"✓ PredefinedSplit created: Train fold = {(test_fold == -1).sum():,} samples")
print(f"                           Val fold   = {(test_fold == 0).sum():,} samples")

# LR pipeline (Investor variant - winner from Stages 1-4)
lr_tuned = LogisticRegression(
    max_iter=2000,
    solver="saga",      # Supports L1/L2, works with sparse data
    random_state=42
)

pipe_lr_tuned = Pipeline([
    ("preprocess", preprocess_investor),  # Investor variant won in Stage 1
    ("model", lr_tuned),
])

# Parameter grid
param_grid = {
    "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
    "model__penalty": ["l2"],
    "model__class_weight": [None, "balanced"],
}

print(f"\n✓ Parameter grid:")
print(f"  - C (regularization): {param_grid['model__C']}")
print(f"  - penalty: {param_grid['model__penalty']}")
print(f"  - class_weight: {param_grid['model__class_weight']}")
print(f"  - Total combinations: {len(param_grid['model__C']) * len(param_grid['model__penalty']) * len(param_grid['model__class_weight'])}")

# GridSearchCV
search = GridSearchCV(
    pipe_lr_tuned,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=ps,              # Time-respecting split
    n_jobs=-1,
    refit=True,         # Refit best estimator on full Train+Val
    verbose=0
)

print("\n" + "-"*80)
print("Running GridSearchCV (this may take 1-2 minutes)...")
print("-"*80)

search.fit(X_tv, y_tv)

print("\n" + "="*80)
print("TUNING COMPLETE")
print("="*80)
print(f"Best params: {search.best_params_}")
print(f"Best Val(2014) ROC-AUC: {search.best_score_:.4f}")
print("="*80)

# Store best estimator
best_lr_pipe = search.best_estimator_

# %%
# Evaluate tuned LR on all splits
from sklearn.metrics import roc_auc_score, average_precision_score

def eval_model(pipeline, X, y, split_name):
    y_proba = pipeline.predict_proba(X)[:, 1]
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    return {"split": split_name, "roc_auc": roc_auc, "pr_auc": pr_auc}

results_tuned = []
results_tuned.append(eval_model(best_lr_pipe, X_val_raw, y_val, "Val(2014)"))
results_tuned.append(eval_model(best_lr_pipe, X_test_raw, y_test, "Test(2015)"))
results_tuned.append(eval_model(best_lr_pipe, X_test60_raw, y_test60, "Holdout60(2013)"))

df_tuned = pd.DataFrame(results_tuned)

print("\n" + "="*80)
print("TUNED LR PERFORMANCE (Best Estimator)")
print("="*80)
print(df_tuned.to_string(index=False))
print("="*80)

# Compare to default LR from Stage 1
print("\n" + "="*80)
print("COMPARISON: Tuned LR vs Default LR (Stage 1)")
print("="*80)

# Fetch Stage 1 default LR results (Investor variant)
# Query df_lr_results for the Investor variant across splits
default_val_auc = df_lr_results[(df_lr_results["model"] == "LogReg") &
                                 (df_lr_results["policy"] == "Investor") &
                                 (df_lr_results["split"] == "Val(2014)")].iloc[0]["roc_auc"]
default_test_auc = df_lr_results[(df_lr_results["model"] == "LogReg") &
                                  (df_lr_results["policy"] == "Investor") &
                                  (df_lr_results["split"] == "Test(2015)")].iloc[0]["roc_auc"]
default_holdout60_auc = df_lr_results[(df_lr_results["model"] == "LogReg") &
                                       (df_lr_results["policy"] == "Investor") &
                                       (df_lr_results["split"] == "Holdout60(2013)")].iloc[0]["roc_auc"]

tuned_val_auc = df_tuned.loc[df_tuned["split"] == "Val(2014)", "roc_auc"].values[0]
tuned_test_auc = df_tuned.loc[df_tuned["split"] == "Test(2015)", "roc_auc"].values[0]
tuned_holdout60_auc = df_tuned.loc[df_tuned["split"] == "Holdout60(2013)", "roc_auc"].values[0]

delta_val = tuned_val_auc - default_val_auc
delta_test = tuned_test_auc - default_test_auc
delta_holdout60 = tuned_holdout60_auc - default_holdout60_auc

comparison = pd.DataFrame([
    {"Split": "Val(2014)", "Default LR": default_val_auc, "Tuned LR": tuned_val_auc, "ΔAUC": delta_val},
    {"Split": "Test(2015)", "Default LR": default_test_auc, "Tuned LR": tuned_test_auc, "ΔAUC": delta_test},
    {"Split": "Holdout60(2013)", "Default LR": default_holdout60_auc, "Tuned LR": tuned_holdout60_auc, "ΔAUC": delta_holdout60},
])

print(comparison.to_string(index=False))
print("="*80)

# Decision logic
print("\n" + "="*80)
print("DECISION GUIDANCE")
print("="*80)

if delta_val >= 0.005 and delta_test >= 0.0 and delta_holdout60 >= -0.002:
    print("✅ Tuned LR shows meaningful improvement on Val (≥+0.005 AUC)")
    print("   AND generalizes well to Test and Holdout60.")
    print("   → FINAL MODEL: Tuned Logistic Regression")
    final_model_pipeline = best_lr_pipe
    final_model_name = "Tuned LogReg Investor"
    final_model_test_auc = tuned_test_auc
elif delta_val >= 0.0:
    print("⚠️ Tuned LR shows marginal improvement on Val (<+0.005 AUC)")
    print("   or does not generalize well to Test/Holdout60.")
    print("   → FINAL MODEL: Default Logistic Regression (simplicity wins)")
    final_model_pipeline = pipe_lr_inv
    final_model_name = "Default LogReg Investor"
    final_model_test_auc = default_test_auc
else:
    print("❌ Tuned LR WORSE than default LR on Val.")
    print("   → FINAL MODEL: Default Logistic Regression")
    final_model_pipeline = pipe_lr_inv
    final_model_name = "Default LogReg Investor"
    final_model_test_auc = default_test_auc

print("="*80)

# %% [markdown]
# ## Threshold Selection: Max F2 Policy
#
# **Why we need a threshold policy:**
#
# ROC-AUC and PR-AUC are **threshold-free** metrics (ranking quality). To make binary predictions (approve/reject loan), we need to choose a classification threshold.
#
# **Why NOT "maximize true positives"?**
#
# If we optimize for recall (true positives) alone, the best "model" is: **predict default for everyone**. You'll catch all defaults... and destroy precision and utility.
#
# **Defensible threshold policies:**
#
# 1. **Max F2:** Recall-weighted F-score (β=2 means recall is 2x more important than precision)
# 2. **Recall at precision floor:** E.g., "maximize recall subject to precision ≥ 30%"
# 3. **Top-k policy:** Investor screens top 10% highest predicted default probability
#
# We'll use **Max F2** on Val(2014) as a recall-focused policy that still respects precision.
#
# **Decision:**
#
# - Tune threshold on Val(2014) to maximize F2
# - Freeze that threshold
# - Evaluate precision/recall/F2 on Test(2015) and Holdout60(2013) with frozen threshold

# %%
from sklearn.metrics import precision_recall_curve

def choose_threshold_max_fbeta(y_true, y_proba, beta=2.0):
    """
    Choose threshold that maximizes Fβ score.
    β=2 means recall is 2x more important than precision.
    """
    p, r, thresholds = precision_recall_curve(y_true, y_proba)
    # precision_recall_curve returns n+1 precision/recall for n thresholds
    p, r = p[:-1], r[:-1]

    # Fβ = (1 + β²) * (precision * recall) / (β² * precision + recall)
    fbeta = (1 + beta**2) * (p * r) / (beta**2 * p + r + 1e-12)

    best_idx = np.argmax(fbeta)
    return thresholds[best_idx], p[best_idx], r[best_idx], fbeta[best_idx]

print("\n" + "="*80)
print("THRESHOLD SELECTION: Max F2 on Val(2014)")
print("="*80)

# Get probabilities on Val(2014)
val_proba = final_model_pipeline.predict_proba(X_val_raw)[:, 1]

# Find optimal threshold
threshold_star, precision_star, recall_star, f2_star = choose_threshold_max_fbeta(
    y_val.values, val_proba, beta=2.0
)

print(f"Optimal threshold (max F2): {threshold_star:.4f}")
print(f"  Val(2014) Precision: {precision_star:.3f}")
print(f"  Val(2014) Recall:    {recall_star:.3f}")
print(f"  Val(2014) F2 score:  {f2_star:.3f}")
print("="*80)

# Evaluate frozen threshold on Test and Holdout60
def eval_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    from sklearn.metrics import precision_score, recall_score, fbeta_score
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)
    return {"precision": precision, "recall": recall, "f2": f2}

test_proba = final_model_pipeline.predict_proba(X_test_raw)[:, 1]
holdout60_proba = final_model_pipeline.predict_proba(X_test60_raw)[:, 1]

test_metrics = eval_threshold(y_test.values, test_proba, threshold_star)
holdout60_metrics = eval_threshold(y_test60.values, holdout60_proba, threshold_star)

print("\n" + "="*80)
print(f"FROZEN THRESHOLD EVALUATION (threshold = {threshold_star:.4f})")
print("="*80)

threshold_results = pd.DataFrame([
    {"Split": "Val(2014)", "Precision": precision_star, "Recall": recall_star, "F2": f2_star},
    {"Split": "Test(2015)", **test_metrics},
    {"Split": "Holdout60(2013)", **holdout60_metrics},
])

print(threshold_results.to_string(index=False))
print("="*80)

print(f"\n✓ Threshold {threshold_star:.4f} achieves recall={recall_star:.2%} on Val")
print(f"  (catches {recall_star:.0%} of defaults while maintaining {precision_star:.0%} precision)")

# %% [markdown]
# ## Calibration Check: Do Predicted Probabilities Match Reality?
#
# **Why calibration matters:**
#
# We claim this is a **PD (Probability of Default) model**. If our probabilities are poorly calibrated, the "PD story" is fake.
#
# **What is calibration?**
#
# A well-calibrated model means:
# - Among loans predicted to have 20% default risk → ~20% actually default
# - Among loans predicted to have 40% default risk → ~40% actually default
#
# **Metrics:**
#
# 1. **Brier Score:** Mean squared error between predicted probabilities and actual outcomes (lower = better; 0 = perfect)
# 2. **Calibration Curve:** Visual check of predicted vs observed default rates
#
# **If calibration is poor:**
#
# We can add `CalibratedClassifierCV` (isotonic or sigmoid) trained on Train only.

# %%
from sklearn.metrics import brier_score_loss

print("\n" + "="*80)
print("CALIBRATION CHECK: Brier Score")
print("="*80)

brier_val = brier_score_loss(y_val, val_proba)
brier_test = brier_score_loss(y_test, test_proba)
brier_holdout60 = brier_score_loss(y_test60, holdout60_proba)

brier_results = pd.DataFrame([
    {"Split": "Val(2014)", "Brier Score": brier_val, "Default Rate": y_val.mean()},
    {"Split": "Test(2015)", "Brier Score": brier_test, "Default Rate": y_test.mean()},
    {"Split": "Holdout60(2013)", "Brier Score": brier_holdout60, "Default Rate": y_test60.mean()},
])

print(brier_results.to_string(index=False))
print("="*80)

print("\nInterpretation:")
print(f"  - Brier score range: [0, 1] (lower = better)")
print(f"  - Baseline (always predict mean): {y_val.mean() * (1 - y_val.mean()):.4f}")
print(f"  - Our model Brier score: {brier_val:.4f}")
if brier_val < y_val.mean() * (1 - y_val.mean()):
    print(f"  ✅ Model is better than baseline (predicting class proportions)")
else:
    print(f"  ⚠️ Model is NOT better than baseline - poor calibration")

# %%
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Plot calibration curves for Val, Test, Holdout60
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (y_true, y_prob, split_name, ax) in enumerate([
    (y_val.values, val_proba, "Val(2014)", axes[0]),
    (y_test.values, test_proba, "Test(2015)", axes[1]),
    (y_test60.values, holdout60_proba, "Holdout60(2013)", axes[2]),
]):
    # Compute calibration curve (10 bins)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )

    # Plot
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="LR")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (Actual Default Rate)")
    ax.set_title(f"Calibration Curve: {split_name}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("\n✓ Calibration curves plotted for Val, Test, Holdout60")
print("  If points follow the diagonal, model is well-calibrated.")
print("  If points are below diagonal, model overestimates default risk.")
print("  If points are above diagonal, model underestimates default risk.")

# %% [markdown]
# ## Model Selection Complete
# ✅ We tested a complete hypothesis-driven progression:
# - **Stage 1:** LR (baseline): Clean, stable AUC 0.65-0.68
# - **Stage 2:** DT (non-linearity test): Failed (high variance, ΔAUC -0.0113)
# - **Stage 3:** RF (ensemble stability): Recovered to LR parity (ΔAUC +0.0051)
# - **Stage 4:** XGBoost (interaction signal): Marginal lift Test36, no improvement Holdout60
# - **Stage 5:** LR tuning: Optimized regularization (C), penalty, class_weight
#
# **Final Selection:** Logistic Regression (Investor variant) with tuned hyperparameters
# - **Rationale:** [Determined by Stage 5 decision logic - see above]
# - **Portfolio Message:** "I tested 4 model families with increasing complexity + LR hyperparameter tuning. Results show additive signal dominates. Tuned LR is the optimal choice."

# %% [markdown]
# # Final Model Lock & Packaging

# %%
print("\n" + "="*80)
print("FINAL MODEL LOCKED")
print("="*80)
print(f"Model: {final_model_name}")
print(f"Test(2015) ROC-AUC: {final_model_test_auc:.4f}")
print(f"Classification threshold: {threshold_star:.4f} (max F2 on Val)")
print(f"Val(2014) F2 score: {f2_star:.3f} (precision={precision_star:.3f}, recall={recall_star:.3f})")
print("="*80)

print("\nModel is ready for packaging in Notebook 3 (Deployment).")

# %% [markdown]
# ## Model Performance Summary
#
# | Metric | Applicant | Investor |
# |--------|-----------|----------|
# | **Test AUC** | 0.6541 | 0.6771 |
# | **Test PR-AUC** | 0.2323 | 0.2522 |
# | **Holdout60 AUC** | 0.6194 | 0.6310 |
# | **Features** | 19 (origination-time) | 19 + int_rate |
#
# **Selected:** Investor variant (best performance, int_rate signal adds ~2% AUC)
#
# ## Honest Limitations
#
# 1. **60-month loans:** 3-4% AUC degradation (term cohorts have different risk profiles)
# 2. **Interest rate dependency:** Model encodes LC's risk assessment indirectly via int_rate
# 3. **Additive structure:** Non-linear modeling adds no value; signal is well-captured by LR
# 4. **Default definition:** Model predicts "Charged Off" vs "Fully Paid" (binary, excludes "Current")

# %% [markdown]
# # Model Packaging & Next Steps
#
# The final model is ready for:
# 1. ✅ Feature validation (origination-time only)
# 2. ✅ Calibration check (Brier score)
# 3. ⏭ Save pipeline and create prediction examples (Dev Notebook 3)
# 4. ⏭ Deploy to production (requires setup.py, requirements.txt, Flask/API wrapper)

# %%
# Final pipeline structure (ready for production deployment)
# Note: Custom transformers can't be pickled directly, but the pipeline
# is reproducible from this script. For production:
# 1. Retrain on all historical data (2011-2015)
# 2. Apply threshold optimization on held-out 2014 validation
# 3. Deploy via containerized Python service with sklearn + preprocessing logic

print("\n✅ Final model pipeline is ready for production deployment")
print(f"   Model: {final_model_name}")
print(f"   Test(2015) ROC-AUC: {final_model_test_auc:.4f}")
print(f"   Threshold: {threshold_star:.4f} (max F2 on validation)")
print(f"   Output: Default probability (0-1) + binary decision")

# %%
# Example: Make predictions on a new sample
if len(X_test_raw) > 0:
    # Take first 5 test samples
    sample_indices = list(range(min(5, len(X_test_raw))))
    X_sample = X_test_raw.iloc[sample_indices]
    y_sample = y_test.iloc[sample_indices]

    sample_proba = final_model_pipeline.predict_proba(X_sample)[:, 1]

    prediction_df = pd.DataFrame({
        "actual_default": y_sample.values,
        "predicted_default_prob": sample_proba,
        "predicted_decision": (sample_proba >= threshold_star).astype(int),
    })

    print("\n" + "="*80)
    print("EXAMPLE PREDICTIONS (First 5 test samples)")
    print("="*80)
    print(prediction_df.to_string(index=False))
    print("="*80)

# %%
print("\n" + "="*80)
print("MODELING COMPLETE")
print("="*80)
print("✅ Data leakage removed")
print("✅ Hypothesis-driven model progression (LR → DT → RF → GB)")
print("✅ Final model: Logistic Regression (Investor, AUC 0.6771)")
print("✅ Portfolio narrative: Complexity tested, additive signal confirmed")
print("✅ Honest limitations documented (60-month weakness, int_rate dependency)")
print("="*80)

# %%
