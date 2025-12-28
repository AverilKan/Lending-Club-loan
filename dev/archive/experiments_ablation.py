# %% [markdown]
# # Ablation Study: int_rate + installment Dependency
#
# This script tests whether our model is truly "independent" or if it's heavily
# dependent on LendingClub's interest rate pricing signal.
#
# **Key Insight from GPT Feedback:**
# "int_rate is downstream of LC's risk model" - our model may be partially
# reverse-engineering LC's pricing strategy rather than predicting risk from
# applicant data alone.
#
# **Experiment Design:**
# 1. Define two feature sets:
#    - "Investor Features": WITH int_rate + installment (what investors see on LC platform)
#    - "Applicant Features": WITHOUT int_rate + installment (pure applicant characteristics)
#
# 2. Train best models (LR with linear pipeline, LGBM with tree pipeline) on both
#
# 3. Measure ΔAUC from removing int_rate + installment
#
# 4. Interpret: How much signal comes from LC's pricing vs pure applicant data?

# %%
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import skew

try:
    from src.transformers import EmpLengthConverter, CreditHistoryCalculator, InterestRateRiskTierTransformer
except ImportError:
    print("Warning: Custom transformers not available")

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

print("=" * 80)
print("ABLATION STUDY: int_rate + installment Dependency")
print("=" * 80)

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def run_ablation_study(X_train, X_test, y_train, y_test,
                      skewed_num_cols, standard_num_cols, ohe_cat_cols,
                      lr_preprocessor, lgbm_preprocessor):
    """
    Run ablation study comparing model performance WITH vs WITHOUT int_rate/installment.

    Parameters:
    -----------
    lr_preprocessor: Linear-optimized preprocessor (Yeo-Johnson + StandardScaler)
    lgbm_preprocessor: Tree-optimized preprocessor (minimal preprocessing)

    Returns:
    --------
    dict: Results with performance metrics for both feature sets
    """

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: ABLATION STUDY (Feature Set Dependency)")
    print("=" * 80)

    # Define feature sets
    all_features = X_train.columns.tolist()

    # Remove int_rate and installment for "applicant only" features
    features_to_remove = {'int_rate', 'installment'}
    applicant_only_features = [f for f in all_features if f not in features_to_remove]

    print(f"\nFeature Sets:")
    print(f"  Investor Features (WITH int_rate/installment): {len(all_features)} features")
    print(f"  Applicant Features (WITHOUT int_rate/installment): {len(applicant_only_features)} features")
    print(f"  Removed: {features_to_remove}")

    # Store results
    ablation_results = []

    # Train on BOTH feature sets
    for feature_set_name, features in [
        ('Investor Features (with LC pricing)', all_features),
        ('Applicant Features (no LC pricing)', applicant_only_features)
    ]:

        X_train_ablation = X_train[features]
        X_test_ablation = X_test[features]

        print(f"\n### Training on: {feature_set_name} ###")
        print(f"Feature count: {len(features)}")

        # --- Logistic Regression (Linear-Optimized Pipeline) ---
        print(f"\n  Logistic Regression (linear-optimized)...", end=" ")

        lr_pipeline = Pipeline([
            ('feature_engineering_emp', EmpLengthConverter()),
            ('feature_engineering_hist', CreditHistoryCalculator()),
            ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
            ('preprocessing', lr_preprocessor),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])

        lr_pipeline.fit(X_train_ablation, y_train)
        lr_test_auc = roc_auc_score(y_test, lr_pipeline.predict_proba(X_test_ablation)[:, 1])
        lr_test_pr = average_precision_score(y_test, lr_pipeline.predict_proba(X_test_ablation)[:, 1])

        print(f"ROC-AUC: {lr_test_auc:.4f}, PR-AUC: {lr_test_pr:.4f}")

        # --- LightGBM (Tree-Optimized Pipeline) ---
        print(f"  LightGBM (tree-optimized)...", end=" ")

        lgbm_pipeline = Pipeline([
            ('feature_engineering_emp', EmpLengthConverter()),
            ('feature_engineering_hist', CreditHistoryCalculator()),
            ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
            ('preprocessing', lgbm_preprocessor),
            ('classifier', LGBMClassifier(random_state=42, verbose=-1))
        ])

        lgbm_pipeline.fit(X_train_ablation, y_train)
        lgbm_test_auc = roc_auc_score(y_test, lgbm_pipeline.predict_proba(X_test_ablation)[:, 1])
        lgbm_test_pr = average_precision_score(y_test, lgbm_pipeline.predict_proba(X_test_ablation)[:, 1])

        print(f"ROC-AUC: {lgbm_test_auc:.4f}, PR-AUC: {lgbm_test_pr:.4f}")

        # Store results
        ablation_results.append({
            'Feature Set': feature_set_name,
            'LR ROC-AUC': lr_test_auc,
            'LR PR-AUC': lr_test_pr,
            'LGBM ROC-AUC': lgbm_test_auc,
            'LGBM PR-AUC': lgbm_test_pr
        })

    # Create results DataFrame
    df_ablation = pd.DataFrame(ablation_results)

    # Calculate effect sizes
    df_ablation['LR ΔAUC'] = df_ablation.loc[0, 'LR ROC-AUC'] - df_ablation.loc[1, 'LR ROC-AUC']
    df_ablation['LGBM ΔAUC'] = df_ablation.loc[0, 'LGBM ROC-AUC'] - df_ablation.loc[1, 'LGBM ROC-AUC']

    # Display results
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print("\n### ROC-AUC Comparison ###")
    print(df_ablation[['Feature Set', 'LR ROC-AUC', 'LGBM ROC-AUC']].to_string(index=False))

    print("\n### Effect Sizes (ΔAUC from removing int_rate + installment) ###")
    print(f"Logistic Regression: {df_ablation.loc[0, 'LR ΔAUC']:+.4f}")
    print(f"LightGBM: {df_ablation.loc[0, 'LGBM ΔAUC']:+.4f}")

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    lr_delta = df_ablation.loc[0, 'LR ΔAUC']
    lgbm_delta = df_ablation.loc[0, 'LGBM ΔAUC']

    print(f"\nApplicant-Only Performance:")
    print(f"  LR:   {df_ablation.loc[1, 'LR ROC-AUC']:.4f} AUC")
    print(f"  LGBM: {df_ablation.loc[1, 'LGBM ROC-AUC']:.4f} AUC")

    print(f"\nWith LendingClub's Pricing (int_rate + installment):")
    print(f"  LR:   {df_ablation.loc[0, 'LR ROC-AUC']:.4f} AUC")
    print(f"  LGBM: {df_ablation.loc[0, 'LGBM ROC-AUC']:.4f} AUC")

    print(f"\nAUC Lift from LC's Pricing:")
    print(f"  LR:   {lr_delta:+.4f} ({lr_delta*100:+.1f}% improvement)")
    print(f"  LGBM: {lgbm_delta:+.4f} ({lgbm_delta*100:+.1f}% improvement)")

    # Honest assessment
    avg_delta = (lr_delta + lgbm_delta) / 2
    print(f"\n### HONEST ASSESSMENT ###")
    if avg_delta > 0.15:
        print(f"⚠️  Model is HEAVILY dependent on LC's pricing (ΔAUC > 0.15)")
        print("→ We are significantly reverse-engineering LendingClub's risk assessment")
    elif avg_delta > 0.05:
        print(f"⚠️  Model shows MODERATE dependency on LC's pricing (ΔAUC 0.05-0.15)")
        print("→ Interest rate signal is important but not dominant")
    else:
        print(f"✓ Model shows WEAK dependency on LC's pricing (ΔAUC < 0.05)")
        print("→ Applicant-only features capture most predictive signal")

    print(f"\nApplicant-Only Capability:")
    applicant_only_auc_lr = df_ablation.loc[1, 'LR ROC-AUC']
    applicant_only_auc_lgbm = df_ablation.loc[1, 'LGBM ROC-AUC']
    print(f"  Can achieve {applicant_only_auc_lr:.4f} - {applicant_only_auc_lgbm:.4f} AUC")
    print(f"  with applicant data alone (no int_rate, no installment)")

    print(f"\nWith LC's Published Pricing:")
    full_auc_lgbm = df_ablation.loc[0, 'LGBM ROC-AUC']
    print(f"  Can achieve {full_auc_lgbm:.4f} AUC")
    print(f"  This represents our model's ability to refine LC's decisions")

    return df_ablation


# %% [markdown]
# # INTEGRATION NOTE
#
# To use this script, insert after running dual-pipeline experiments:
#
# ```python
# # Call ablation study
# ablation_df = run_ablation_study(
#     X_train, X_test, y_train, y_test,
#     skewed_num_cols, standard_num_cols, ohe_cat_cols,
#     lr_preprocessor, lgbm_preprocessor
# )
# ```
#
# Where lr_preprocessor and lgbm_preprocessor are the best pipelines identified
# from the dual-pipeline experiments.
