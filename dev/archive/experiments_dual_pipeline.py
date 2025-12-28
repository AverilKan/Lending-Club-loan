# %% [markdown]
# # Dual-Pipeline Experiments: Testing Feature Engineering Impact on Model Families
#
# This script tests the hypothesis that our feature engineering preprocessing pipeline
# (Yeo-Johnson + StandardScaler + OHE) inadvertently optimizes for linear models while
# handicapping tree-based models.
#
# **Hypothesis:** Tree-based models will show better (or comparable) performance with
# minimal preprocessing (no scaling, no transformation, sparse OHE), while LR will
# suffer without normalization.
#
# **Approach:**
# 1. Create two preprocessing pipelines
#    - Linear-optimized: Yeo-Johnson + StandardScaler + OHE(sparse=False)
#    - Tree-optimized: Median imputation only + OHE(sparse=True)
# 2. Train 4 models × 2 pipelines = 8 combinations
# 3. Report observed effect sizes (ΔAUC for each model)
# 4. Interpret what this reveals about data characteristics

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os, sys
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from scipy.stats import skew

# Import custom transformers from Phase 2
try:
    from src.transformers import EmpLengthConverter, CreditHistoryCalculator, InterestRateRiskTierTransformer
except ImportError:
    print("Warning: Custom transformers not available")

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

# =============================================================================
# SETUP: Load preprocessed data from Phase 2
# =============================================================================

# Assuming the data prep from dev/2_Modelling.py has already been run
# We'll load the processed datasets if they exist
print("=" * 80)
print("DUAL-PIPELINE EXPERIMENT: Testing Preprocessing Impact on Model Families")
print("=" * 80)

# For now, we'll need to run the Phase 2 preprocessing first
# This script is designed to be inserted INTO dev/2_Modelling.py after data loading
# but after the time-based split but BEFORE model training

print("\nNote: This script should be integrated into dev/2_Modelling.py")
print("after time-based data splitting (X_train, X_test, y_train, y_test defined)")
print("and custom feature engineering transformers are available.\n")

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def create_linear_optimized_pipeline(skewed_num_cols, standard_num_cols, ohe_cat_cols):
    """
    Create preprocessing pipeline optimized for linear models (Logistic Regression).

    Strategy:
    - Skewed numeric: impute(median) → Yeo-Johnson → StandardScaler
    - Standard numeric: impute(median) → StandardScaler
    - Categorical: impute(mode) → OHE(sparse=False, dense output)

    Rationale: Linear models benefit from normalized, transformed features
    """
    skewed_num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('transformer', PowerTransformer(method='yeo-johnson')),
        ('scaler', StandardScaler())
    ])

    standard_num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    ohe_cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_skewed', skewed_num_pipe, skewed_num_cols),
            ('num_standard', standard_num_pipe, standard_num_cols),
            ('cat_ohe', ohe_cat_pipe, ohe_cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def create_tree_optimized_pipeline(skewed_num_cols, standard_num_cols, ohe_cat_cols):
    """
    Create preprocessing pipeline optimized for tree-based models.

    Strategy:
    - All numeric: impute(median) only - NO scaling, NO transformation
    - Categorical: impute(mode) → OHE(sparse=True)

    Rationale:
    - Trees are scale-invariant, don't benefit from StandardScaler
    - Yeo-Johnson removes useful signal about feature distributions
    - Sparse OHE saves memory and preserves categorical structure
    - Trees prefer raw features to find their own optimal splits
    """
    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median'))
        # NO scaling, NO transformation
    ])

    ohe_cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipe, skewed_num_cols + standard_num_cols),
            ('cat_ohe', ohe_cat_pipe, ohe_cat_cols)
        ],
        remainder='drop'
    )
    return preprocessor


def run_dual_pipeline_experiments(X_train, X_test, y_train, y_test,
                                   skewed_num_cols, standard_num_cols, ohe_cat_cols):
    """
    Train all models on both preprocessing pipelines and report results.

    Returns:
    --------
    pd.DataFrame: Results with columns [Model, Pipeline, Val AUC, Test AUC, ΔAUC]
    """

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: DUAL-PIPELINE MATRIX (4 Models × 2 Pipelines)")
    print("=" * 80)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=100,
                                                min_samples_leaf=50, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                min_samples_split=100, random_state=42),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
    }

    # Define pipelines
    pipelines = {
        'Linear-Optimized': create_linear_optimized_pipeline(
            skewed_num_cols, standard_num_cols, ohe_cat_cols),
        'Tree-Optimized': create_tree_optimized_pipeline(
            skewed_num_cols, standard_num_cols, ohe_cat_cols)
    }

    results = []

    # Train matrix: 4 models × 2 pipelines = 8 experiments
    for model_name, model in models.items():
        for pipeline_name, preprocessor in pipelines.items():

            print(f"\nTraining {model_name:20s} with {pipeline_name:20s}...", end=" ")

            # Create full pipeline: custom FE → preprocessing → classifier
            full_pipeline = Pipeline([
                ('feature_engineering_emp', EmpLengthConverter()),
                ('feature_engineering_hist', CreditHistoryCalculator()),
                ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
                ('preprocessing', preprocessor),
                ('classifier', model)
            ])

            # Fit on train
            full_pipeline.fit(X_train, y_train)

            # Evaluate on validation
            y_val_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
            val_auc = roc_auc_score(y_test, y_val_pred_proba)

            # Also compute PR-AUC for imbalanced classification
            val_pr_auc = average_precision_score(y_test, y_val_pred_proba)

            results.append({
                'Model': model_name,
                'Pipeline': pipeline_name,
                'Test ROC-AUC': val_auc,
                'Test PR-AUC': val_pr_auc
            })

            print(f"ROC-AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}")

    # Create results DataFrame
    df_results = pd.DataFrame(results)

    # Compute effect sizes (ΔAUC for each model)
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    pivot_roc = df_results.pivot_table(
        index='Model', columns='Pipeline', values='Test ROC-AUC'
    )
    pivot_roc['ΔAUC'] = pivot_roc['Linear-Optimized'] - pivot_roc['Tree-Optimized']
    pivot_roc['Winner'] = pivot_roc.apply(
        lambda row: 'Linear' if row['ΔAUC'] > 0.01 else
                   'Tree' if row['ΔAUC'] < -0.01 else 'Tie', axis=1
    )

    print("\n### Preprocessing Strategy Impact on ROC-AUC ###")
    print(pivot_roc.to_string())

    print("\n### Interpretation ###")
    print(f"\nLinear-Optimized Pipeline (Yeo-Johnson + StandardScaler + OHE):")
    print(f"  - Best for: {pivot_roc[pivot_roc['Winner'] == 'Linear'].index.tolist()}")
    print(f"  - Average AUC: {pivot_roc['Linear-Optimized'].mean():.4f}")

    print(f"\nTree-Optimized Pipeline (Impute only + OHE(sparse)):")
    print(f"  - Best for: {pivot_roc[pivot_roc['Winner'] == 'Tree'].index.tolist()}")
    print(f"  - Average AUC: {pivot_roc['Tree-Optimized'].mean():.4f}")

    # Key insight
    lr_delta = pivot_roc.loc['Logistic Regression', 'ΔAUC']
    print(f"\n### KEY FINDING ###")
    print(f"Logistic Regression ΔAUC: {lr_delta:+.4f}")
    if lr_delta > 0.10:
        print("→ LR heavily relies on normalization (preprocessing matters enormously)")
    elif lr_delta > 0.05:
        print("→ LR benefits moderately from preprocessing")
    else:
        print("→ LR works reasonably well without preprocessing")

    tree_avg_delta = pivot_roc[['Logistic Regression', 'Decision Tree', 'Random Forest', 'LightGBM']].drop('Logistic Regression')
    tree_avg_delta = -tree_avg_delta['ΔAUC'].mean()  # Average benefit from tree-optimized
    if abs(tree_avg_delta) > 0.02:
        print(f"→ Trees prefer tree-optimized preprocessing ({tree_avg_delta:+.4f} average ΔAUC)")
    else:
        print(f"→ Trees perform similarly on both pipelines ({tree_avg_delta:+.4f} average ΔAUC)")

    return df_results, pivot_roc


# %% [markdown]
# # INTEGRATION NOTE
#
# To use this script, insert the following after line 857 (time-based split) in dev/2_Modelling.py:
#
# ```python
# # Call dual-pipeline experiments
# results_df, results_pivot = run_dual_pipeline_experiments(
#     X_train, X_test, y_train, y_test,
#     skewed_num_cols, standard_num_cols, ohe_cat_cols
# )
# ```
#
# This requires skewed_num_cols, standard_num_cols, ohe_cat_cols to be defined in the main script.
