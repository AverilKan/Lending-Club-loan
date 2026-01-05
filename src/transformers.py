"""
Custom transformers for Lending Club loan default prediction.

These transformers are used in the final production pipeline.
All transformers must be in an importable module to enable joblib pickling.
"""

import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# --- Helper functions and constants (module scope) ---

EMP_MAP = {
    "10+ years": 10,
    "9 years": 9, "8 years": 8, "7 years": 7, "6 years": 6, "5 years": 5,
    "4 years": 4, "3 years": 3, "2 years": 2, "1 year": 1,
    "< 1 year": 0,
}


def parse_month_year(series: pd.Series) -> pd.Series:
    """Parse 'Dec-15' or 'Dec-2015' format to datetime."""
    s = series.astype(str)
    dt1 = pd.to_datetime(s, format="%b-%y", errors="coerce")
    dt2 = pd.to_datetime(s, format="%b-%Y", errors="coerce")
    return dt1.fillna(dt2)


def parse_percent_series(s: pd.Series) -> pd.Series:
    """Parse '13.56%' or numeric values to float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    out = s.astype(str).str.replace("%", "", regex=False).str.strip()
    return pd.to_numeric(out, errors="coerce")


def parse_term_series(s: pd.Series) -> pd.Series:
    """Parse ' 36 months' to 36."""
    out = s.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(out, errors="coerce")


def parse_emp_length_series(s: pd.Series) -> pd.Series:
    """Parse '10+ years' to 10."""
    return s.map(EMP_MAP).astype("float")


# --- Custom Transformers ---

class RawLendingClubCleaner(BaseEstimator, TransformerMixin):
    """
    Deterministic cleaning step for raw LendingClub data.

    Removes:
    - Leakage features (post-origination outcomes)
    - Circular reasoning features (grade/sub_grade)
    - Extremely sparse/useless columns

    Parses:
    - term: "36 months" → 36
    - emp_length: "10+ years" → 10
    - Derives credit_history_years from earliest_cr_line

    Parameters:
    -----------
    include_pricing_features : bool, default=True
        If True, keep int_rate (investor features)
        If False, remove them (applicant-only policy)
    """

    def __init__(self, include_pricing_features=True):
        self.include_pricing_features = include_pricing_features
        self.columns_seen_ = None

    def fit(self, X, y=None):
        self.columns_seen_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.copy()

        # === ALLOWLIST APPROACH ===
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

        # Extended bureau features (optional)
        USE_EXTENDED = False
        extended_allowlist = [
            "mort_acc", "pub_rec_bankruptcies", "tax_liens",
            "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl",
            "num_il_tl", "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
            "pct_tl_nvr_dlq",
            "mths_since_last_delinq", "mths_since_last_record", "mths_since_last_major_derog",
            "mths_since_rcnt_il", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
            "mths_since_recent_inq", "mths_since_recent_revol_delinq",
            "avg_cur_bal", "bc_open_to_buy", "bc_util", "tot_cur_bal",
            "total_bal_ex_mort", "total_bc_limit", "total_il_high_credit_limit",
            "tot_hi_cred_lim", "max_bal_bc", "all_util", "il_util",
            "percent_bc_gt_75", "total_rev_hi_lim",
            "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl",
            "acc_open_past_24mths", "num_tl_op_past_12m", "inq_last_12m",
            "num_accts_ever_120_pd", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
            "acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt",
            "collections_12_mths_ex_med", "num_collections_12_mths_ex_med",
            "application_type", "initial_list_status", "disbursement_method",
            "sec_app_fico_range_low", "sec_app_fico_range_high",
            "sec_app_earliest_cr_line", "sec_app_inq_last_6mths",
            "sec_app_mort_acc", "sec_app_open_acc", "sec_app_revol_util",
            "sec_app_open_act_il", "sec_app_num_rev_accts",
            "sec_app_chargeoff_within_12_mths", "sec_app_collections_12_mths_ex_med",
            "sec_app_mths_since_last_major_derog",
        ]

        allowlist = core_allowlist + extended_allowlist if USE_EXTENDED else core_allowlist

        if self.include_pricing_features:
            allowlist.extend(["int_rate"])

        temp_fields = ["issue_d", "earliest_cr_line"]
        allowlist.extend(temp_fields)

        # === HARD DROPS (Post-Origination Leakage) ===
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

        hard_drop_explicit = [
            "grade", "sub_grade",  # Circular
            "id", "member_id", "url", "emp_title", "zip_code", "title", "desc",
            "funded_amnt", "funded_amnt_inv",  # VIF redundancy
            "loan_status", "target_default", "maturity_date", "term_months",
            "installment",  # Deterministic from loan_amnt+term+int_rate
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
        # Three-tier approach for inference tolerance:
        # 1. If credit_history_years already exists (pre-computed) → use it
        # 2. Else if issue_d + earliest_cr_line exist → compute it
        # 3. Else → raise clear schema error

        if "credit_history_years" in X.columns and X["credit_history_years"].notna().any():
            # CASE 1: Pre-computed credit_history_years (inference mode)
            # Validate it's reasonable (0-80 years)
            valid_range = (X["credit_history_years"] >= 0) & (X["credit_history_years"] <= 80)
            if not valid_range.any():
                raise ValueError(
                    "CRITICAL: Pre-computed credit_history_years exists but all values are out of range (expected 0-80 years). "
                    "Either provide valid credit_history_years, OR provide issue_d + earliest_cr_line for automatic computation."
                )
            # Keep pre-computed values (drop source columns if present)
            X.drop(columns=["issue_d", "earliest_cr_line"], inplace=True, errors="ignore")

        elif "issue_d" in X.columns and "earliest_cr_line" in X.columns:
            # CASE 2: Compute from source columns (training mode)
            if not np.issubdtype(X["issue_d"].dtype, np.datetime64):
                X["issue_d"] = parse_month_year(X["issue_d"])
            if not np.issubdtype(X["earliest_cr_line"].dtype, np.datetime64):
                X["earliest_cr_line"] = parse_month_year(X["earliest_cr_line"])

            X["credit_history_years"] = (X["issue_d"] - X["earliest_cr_line"]).dt.days / 365.25

            # Sanity check: did computation succeed?
            if X["credit_history_years"].isna().all():
                raise ValueError(
                    "CRITICAL: credit_history_years computation failed (all NaN). "
                    "Check that issue_d and earliest_cr_line are valid date strings (e.g., 'Dec-15')."
                )

            # Drop source columns (no longer needed)
            X.drop(columns=["issue_d", "earliest_cr_line"], inplace=True, errors="ignore")

        else:
            # CASE 3: Missing required inputs
            raise ValueError(
                "CRITICAL: Cannot derive credit_history_years. "
                "Please provide ONE of:\n"
                "  (a) Pre-computed 'credit_history_years' column, OR\n"
                "  (b) Both 'issue_d' and 'earliest_cr_line' columns for automatic computation.\n"
                "\n"
                "Available columns: " + ", ".join(sorted(X.columns))
            )

        # Final validation: ensure credit_history_years exists and has valid values
        if "credit_history_years" not in X.columns:
            raise ValueError("CRITICAL: credit_history_years missing after derivation logic")

        # === APPLY ALLOWLIST ===
        final_allowlist = [c for c in allowlist if c not in temp_fields]
        cols_to_keep = [c for c in X.columns if c in final_allowlist]

        expected_present = set(final_allowlist)
        actually_present = set(cols_to_keep)
        missing_features = expected_present - actually_present

        if len(missing_features) > 0:
            threshold = 10 if USE_EXTENDED else 3
            if len(missing_features) > threshold:
                raise ValueError(f"CRITICAL: Too many expected features missing ({len(missing_features)} > {threshold}). "
                               f"Missing: {sorted(missing_features)}")

        X = X[cols_to_keep]

        # === VERIFICATION ===
        cleaned_set = set(X.columns)
        allowed_set = set(final_allowlist)
        unexpected = cleaned_set - allowed_set
        if len(unexpected) > 0:
            raise ValueError(f"ALLOWLIST VIOLATION: {len(unexpected)} unexpected columns: {sorted(unexpected)}")

        leakage_substrings = ["pymnt", "total_rec", "recover", "out_prncp", "last_pymnt",
                              "next_pymnt", "settlement", "hardship", "debt_settlement"]
        leakage_found = []
        for col in X.columns:
            for substr in leakage_substrings:
                if substr.lower() in col.lower():
                    leakage_found.append(col)
                    break

        if leakage_found:
            raise ValueError(f"LEAKAGE DETECTED: {len(leakage_found)} suspicious columns: {leakage_found}")

        # Basic validity rules
        if "revol_util" in X.columns:
            X.loc[(X["revol_util"] < 0) | (X["revol_util"] > 100), "revol_util"] = np.nan

        if "dti" in X.columns:
            X.loc[X["dti"] < 0, "dti"] = np.nan

        if "credit_history_years" in X.columns:
            X.loc[(X["credit_history_years"] < 0) | (X["credit_history_years"] > 80), "credit_history_years"] = np.nan

        return X


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
    """Cap numeric columns at training-learned quantile (reduces outlier leverage)."""

    def __init__(self, quantile: float = 0.99):
        self.quantile = quantile
        self.caps_ = None

    def fit(self, X, y=None):
        # Handle both DataFrame and array input
        Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        self.caps_ = Xdf.quantile(self.quantile, numeric_only=True)
        return self

    def transform(self, X):
        # Handle both DataFrame and array input (SimpleImputer returns array)
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for c in Xdf.columns:
            if self.caps_ is not None and c in self.caps_.index:
                Xdf[c] = np.minimum(Xdf[c], self.caps_[c])
        return Xdf.values  # Return array to match sklearn pipeline expectations


class ColumnAligner(BaseEstimator, TransformerMixin):
    """
    Ensure required columns exist for ColumnTransformer with explicit lists.

    CRITICAL: When using explicit column lists in ColumnTransformer (no callable selectors),
    missing columns cause KeyError. This transformer guarantees all required columns exist.

    Parameters:
    -----------
    required_cols : list[str]
        All columns that ColumnTransformer expects (union of all explicit lists)
    """

    def __init__(self, required_cols: list[str]):
        self.required_cols = required_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdf = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Add any missing required columns as NaN
        missing_cols = set(self.required_cols) - set(Xdf.columns)
        for col in missing_cols:
            Xdf[col] = np.nan

        # Return only required columns (drop extras, reorder to match required_cols)
        return Xdf[self.required_cols]
