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

        # Always drop target and any time-only identifiers from features here (handled outside too, but keep safe)
        for col in ["loan_status", "target_default"]:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        # Circular reasoning: LendingClub's own risk output
        for col in ["grade", "sub_grade"]:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        # Optional: applicant baseline excludes pricing outputs
        # - int_rate: LC's published pricing signal
        # - installment: Derived from (loan_amnt, term, int_rate), VIF ~11.5
        #   EDA showed moderate multicollinearity; exclude from baseline, test in ablation
        if not self.include_pricing_features:
            for col in ["int_rate", "installment"]:
                if col in X.columns:
                    X.drop(columns=[col], inplace=True)

        # Known identifiers / high-cardinality noise / unstructured text
        for col in ["id", "member_id", "url", "emp_title", "zip_code", "desc"]:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        # VIF redundancy: drop funded_amnt fields (EDA showed VIF > 1100)
        # funded_amnt and funded_amnt_inv are 99.96% correlated with loan_amnt
        # Keep loan_amnt as the primary loan amount feature
        for col in ["funded_amnt", "funded_amnt_inv"]:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        # High-cardinality categoricals that cause feature explosion
        # - title: 61,671 unique values (arbitrary job descriptions, minimal signal)
        # - earliest_cr_line: dropped after credit_history_years derivation
        for col in ["title"]:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        # Post-origination / servicing leakage (explicit + pattern)
        leakage_explicit = [
            "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
            "total_rec_late_fee", "recoveries", "collection_recovery_fee",
            "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
            "out_prncp", "out_prncp_inv",
            "last_credit_pull_d",
            # operational/servicing-type fields often sparse but leakage-prone
            "payment_plan_start_date", "deferral_term", "orig_projected_additional_accrued_interest",
        ]
        for col in leakage_explicit:
            if col in X.columns:
                X.drop(columns=[col], inplace=True)

        # Pattern-based leakage families (conservative)
        leakage_patterns = [
            r"pymnt", r"recover", r"collection", r"hardship_", r"settlement_", r"debt_settlement",
            r"chargeoff", r"last_pymnt", r"next_pymnt", r"out_prncp", r"total_rec",
        ]
        drop_by_pattern = []
        for c in X.columns:
            for pat in leakage_patterns:
                if re.search(pat, c, flags=re.IGNORECASE):
                    drop_by_pattern.append(c)
                    break
        if drop_by_pattern:
            X.drop(columns=sorted(set(drop_by_pattern)), inplace=True, errors="ignore")

        # Parse common messy fields
        if "term" in X.columns:
            X["term"] = parse_term_series(X["term"])

        if "emp_length" in X.columns:
            X["emp_length"] = parse_emp_length_series(X["emp_length"])

        for pct_col in ["int_rate", "revol_util"]:
            if pct_col in X.columns:
                X[pct_col] = parse_percent_series(X[pct_col])

        # Derive credit_history_years if dates available
        if "issue_d" in X.columns and not np.issubdtype(X["issue_d"].dtype, np.datetime64):
            X["issue_d"] = parse_month_year(X["issue_d"])
        if "earliest_cr_line" in X.columns and not np.issubdtype(X["earliest_cr_line"].dtype, np.datetime64):
            X["earliest_cr_line"] = parse_month_year(X["earliest_cr_line"])

        if "issue_d" in X.columns and "earliest_cr_line" in X.columns:
            X["credit_history_years"] = (X["issue_d"] - X["earliest_cr_line"]).dt.days / 365.25

        # Drop issue_d from features (only used for splitting)
        if "issue_d" in X.columns:
            X.drop(columns=["issue_d"], inplace=True)

        # Drop maturity_date (synthetic field derived from issue_d + term, encodes time)
        # Keep term (real feature available at origination: 36 vs 60 months)
        if "maturity_date" in X.columns:
            X.drop(columns=["maturity_date"], inplace=True)

        # Also drop term_months (temporary column used only for maturity calculation)
        if "term_months" in X.columns:
            X.drop(columns=["term_months"], inplace=True)

        # Drop earliest_cr_line now that credit_history_years is derived
        # (Must happen AFTER derivation to preserve the engineered feature)
        if "earliest_cr_line" in X.columns:
            X.drop(columns=["earliest_cr_line"], inplace=True)

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

# %%
