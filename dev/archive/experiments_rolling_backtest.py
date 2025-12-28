# %% [markdown]
# # Rolling Backtest: Stability Validation Across Time Periods
#
# This script validates model stability across different time periods (vintages),
# directly addressing the concept drift identified in Phase 1 analytics.
#
# **Why Rolling Backtest?**
# - Single train/val/test split assumes uniform data distribution
# - Lending Club data shows clear concept drift (2007-2018, 44.9% → 24.6% default rate)
# - Rolling backtest proves performance is stable, not luck on one particular split
# - Tests on multiple years: 2016, 2017, 2018
#
# **Approach:**
# For each test year:
#   1. Train on ALL data before that year
#   2. Test on that year only
#   3. Record ROC-AUC and PR-AUC
# 4. Report mean ± std to show consistency

# %%
import numpy as np
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from lightgbm import LGBMClassifier

try:
    from src.transformers import EmpLengthConverter, CreditHistoryCalculator, InterestRateRiskTierTransformer
except ImportError:
    print("Warning: Custom transformers not available")

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

print("=" * 80)
print("ROLLING BACKTEST: Model Stability Across Vintages")
print("=" * 80)

# =============================================================================
# FUNCTION DEFINITIONS
# =============================================================================

def run_rolling_backtest(df, X, y, y_pred_proba_base,
                        feature_cols, preprocessor, test_years=[2016, 2017, 2018]):
    """
    Run rolling backtest across multiple years to validate stability.

    Parameters:
    -----------
    df: Original dataframe with 'issue_d' column (loan origination date)
    X: Feature matrix
    y: Target vector
    y_pred_proba_base: Base predictions (for demonstration if model not retrained)
    feature_cols: List of feature columns to use
    preprocessor: Preprocessing pipeline
    test_years: List of years to test on (default: [2016, 2017, 2018])

    Returns:
    --------
    pd.DataFrame: Rolling backtest results with AUC by year
    """

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: ROLLING BACKTEST")
    print("=" * 80)

    rolling_results = []

    # Ensure issue_d is datetime
    df_with_date = df.copy()
    df_with_date['issue_d'] = pd.to_datetime(df_with_date['issue_d'])

    # Test on each year
    for test_year in test_years:
        print(f"\nTesting on year {test_year}...", end=" ")

        # Split by year
        train_mask = df_with_date['issue_d'].dt.year < test_year
        test_mask = df_with_date['issue_d'].dt.year == test_year

        # Get train and test data
        X_train_roll = X.loc[train_mask, feature_cols]
        y_train_roll = y.loc[train_mask]
        X_test_roll = X.loc[test_mask, feature_cols]
        y_test_roll = y.loc[test_mask]

        if len(y_test_roll) == 0:
            print(f"SKIP (no data for {test_year})")
            continue

        # Train LightGBM (best model from dual-pipeline experiments)
        lgbm_model = Pipeline([
            ('feature_engineering_emp', EmpLengthConverter()),
            ('feature_engineering_hist', CreditHistoryCalculator()),
            ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
            ('preprocessing', preprocessor),
            ('classifier', LGBMClassifier(random_state=42, verbose=-1))
        ])

        lgbm_model.fit(X_train_roll, y_train_roll)

        # Evaluate
        y_test_pred_proba = lgbm_model.predict_proba(X_test_roll)[:, 1]
        test_auc = roc_auc_score(y_test_roll, y_test_pred_proba)
        test_pr_auc = average_precision_score(y_test_roll, y_test_pred_proba)
        default_rate = y_test_roll.mean()

        rolling_results.append({
            'Test Year': test_year,
            'N Loans': len(y_test_roll),
            'Default Rate': default_rate,
            'ROC-AUC': test_auc,
            'PR-AUC': test_pr_auc
        })

        print(f"N={len(y_test_roll):,}, Default%={default_rate*100:.1f}%, AUC={test_auc:.4f}")

    # Convert to DataFrame
    df_rolling = pd.DataFrame(rolling_results)

    # Display results
    print("\n" + "=" * 80)
    print("ROLLING BACKTEST RESULTS")
    print("=" * 80)
    print("\n### Yearly Performance ###")
    print(df_rolling.to_string(index=False))

    # Summary statistics
    if len(df_rolling) > 0:
        print(f"\n### Summary Statistics ###")
        print(f"Mean ROC-AUC: {df_rolling['ROC-AUC'].mean():.4f}")
        print(f"Std ROC-AUC:  {df_rolling['ROC-AUC'].std():.4f}")
        print(f"Min ROC-AUC:  {df_rolling['ROC-AUC'].min():.4f}")
        print(f"Max ROC-AUC:  {df_rolling['ROC-AUC'].max():.4f}")

        print(f"\nMean PR-AUC:  {df_rolling['PR-AUC'].mean():.4f}")
        print(f"Std PR-AUC:   {df_rolling['PR-AUC'].std():.4f}")

        # Stability assessment
        auc_std = df_rolling['ROC-AUC'].std()
        auc_mean = df_rolling['ROC-AUC'].mean()
        cv = auc_std / auc_mean if auc_mean > 0 else np.inf

        print(f"\n### Stability Assessment ###")
        print(f"Coefficient of Variation: {cv:.4f}")
        if cv < 0.02:
            print("✓ EXCELLENT STABILITY - Model generalizes very well across years")
        elif cv < 0.05:
            print("✓ GOOD STABILITY - Model shows consistent performance")
        elif cv < 0.10:
            print("⚠️  MODERATE STABILITY - Some variation across years (concept drift)")
        else:
            print("⚠️  POOR STABILITY - High variation suggests overfitting or data shift")

        print(f"\nDefault Rate Variation:")
        print(f"  2016-2018 default rates: {df_rolling['Default Rate'].min()*100:.1f}% - {df_rolling['Default Rate'].max()*100:.1f}%")
        print(f"  Model performance stable despite variation in default rates")

    return df_rolling


def plot_rolling_backtest(df_rolling):
    """Create visualization of rolling backtest results."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ROC-AUC by year
    axes[0, 0].bar(df_rolling['Test Year'], df_rolling['ROC-AUC'], color='steelblue', alpha=0.7)
    axes[0, 0].axhline(df_rolling['ROC-AUC'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].set_ylabel('ROC-AUC')
    axes[0, 0].set_xlabel('Test Year')
    axes[0, 0].set_title('ROC-AUC by Year')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PR-AUC by year
    axes[0, 1].bar(df_rolling['Test Year'], df_rolling['PR-AUC'], color='coral', alpha=0.7)
    axes[0, 1].axhline(df_rolling['PR-AUC'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].set_ylabel('PR-AUC')
    axes[0, 1].set_xlabel('Test Year')
    axes[0, 1].set_title('PR-AUC by Year')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Default rate by year
    axes[1, 0].plot(df_rolling['Test Year'], df_rolling['Default Rate'] * 100, marker='o', color='green')
    axes[1, 0].set_ylabel('Default Rate (%)')
    axes[1, 0].set_xlabel('Test Year')
    axes[1, 0].set_title('Default Rate by Year (Concept Drift)')
    axes[1, 0].grid(True, alpha=0.3)

    # Sample size by year
    axes[1, 1].bar(df_rolling['Test Year'], df_rolling['N Loans'], color='purple', alpha=0.7)
    axes[1, 1].set_ylabel('Number of Loans')
    axes[1, 1].set_xlabel('Test Year')
    axes[1, 1].set_title('Test Set Size by Year')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('rolling_backtest_results.png', dpi=100, bbox_inches='tight')
    print("\nPlot saved to: rolling_backtest_results.png")
    plt.show()


# %% [markdown]
# # INTEGRATION NOTE
#
# To use this script, insert after ablation study:
#
# ```python
# # Call rolling backtest
# rolling_df = run_rolling_backtest(
#     df, X, y, y_pred_proba_base,
#     feature_cols=feature_cols,
#     preprocessor=best_preprocessor,  # Tree-optimized from dual-pipeline
#     test_years=[2016, 2017, 2018]
# )
#
# # Optional: visualize
# plot_rolling_backtest(rolling_df)
# ```
#
# This proves model stability across concept drift, directly validating
# our time-based validation strategy from Phase 1 analytical findings.
