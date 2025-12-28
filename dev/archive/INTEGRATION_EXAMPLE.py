# %% [markdown]
# # Integration Example: Running All Three Experiments
#
# This file shows a practical example of how to integrate the three critical
# experiments into your modeling workflow. Copy-paste sections into dev/2_Modelling.py
# after your data preparation and time-based train/test split.
#
# **Requirements before running experiments:**
# - X_train, X_test defined (feature matrices after time-based split)
# - y_train, y_test defined (target vectors)
# - df (original dataframe with 'issue_d' column for rolling backtest)
# - skewed_num_cols, standard_num_cols, ohe_cat_cols defined
# - Custom transformers available (src.transformers)

# %% [markdown]
# ## Step 1: Import Experiment Modules

# %%
import sys
import os

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dev.run_all_experiments import ExperimentRunner

# %% [markdown]
# ## Step 2: Prepare Your Data (from dev/2_Modelling.py)
#
# By the time you're ready to run experiments, you should have:
# - Loaded and preprocessed data
# - Defined feature subsets
# - Performed time-based train/test split

# %%
# Example: Assuming these are already defined in your notebook
# X_train, X_test, y_train, y_test              <- Feature matrices
# df                                             <- Original dataframe
# skewed_num_cols, standard_num_cols, ohe_cat_cols <- Feature subsets

# %% [markdown]
# ## Step 3: Initialize and Run Experiment Runner
#
# This is the main call that orchestrates all three experiments.

# %%
print("=" * 80)
print("INITIALIZING EXPERIMENT RUNNER")
print("=" * 80)

runner = ExperimentRunner(
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    df_original=df,
    skewed_num_cols=skewed_num_cols,
    standard_num_cols=standard_num_cols,
    ohe_cat_cols=ohe_cat_cols
)

# Run all three experiments in sequence
print("\nStarting all three experiments...")
results = runner.run_all_experiments()

# %% [markdown]
# ## Step 4: Access and Interpret Results
#
# Results from all experiments are stored in runner.results dictionary

# %%
print("\n" + "=" * 80)
print("ACCESSING EXPERIMENT RESULTS")
print("=" * 80)

# Experiment 1: Dual-Pipeline Results
if 'dual_pipeline' in results:
    df_dual = results['dual_pipeline']
    print("\n### Experiment 1: Dual-Pipeline Matrix ###")
    print(f"Shape: {df_dual.shape}")
    print(f"Columns: {df_dual.columns.tolist()}")
    print(f"\nTop results:")
    print(df_dual.sort_values('Test ROC-AUC', ascending=False).head())

# Experiment 2: Ablation Study Results
if 'ablation' in results:
    df_ablation = results['ablation']
    print("\n### Experiment 2: Ablation Study ###")
    print(f"Shape: {df_ablation.shape}")
    print(f"\nResults:")
    print(df_ablation)

# Experiment 3: Rolling Backtest Results
if 'rolling' in results and results['rolling'] is not None:
    df_rolling = results['rolling']
    print("\n### Experiment 3: Rolling Backtest ###")
    print(f"Shape: {df_rolling.shape}")
    print(f"\nResults:")
    print(df_rolling)
    print(f"\nStability metrics:")
    print(f"  Mean AUC: {df_rolling['ROC-AUC'].mean():.4f}")
    print(f"  Std AUC:  {df_rolling['ROC-AUC'].std():.4f}")
    print(f"  CV:       {df_rolling['ROC-AUC'].std() / df_rolling['ROC-AUC'].mean():.4f}")

# %% [markdown]
# ## Step 5: Extract Key Findings for Your Narrative
#
# After experiments complete, extract key values for your project story

# %%
print("\n" + "=" * 80)
print("KEY FINDINGS FOR PROJECT NARRATIVE")
print("=" * 80)

# Extract key effect sizes
if 'dual_pipeline' in results:
    df_dp = results['dual_pipeline']

    # Get best LR performance and best tree performance
    lr_results = df_dp[df_dp['Model'] == 'Logistic Regression']

    if len(lr_results) > 0:
        lr_linear = lr_results[lr_results['Pipeline'] == 'Linear-Optimized']['Test ROC-AUC'].values[0]
        lr_tree = lr_results[lr_results['Pipeline'] == 'Tree-Optimized']['Test ROC-AUC'].values[0]
        lr_delta = lr_linear - lr_tree

        print(f"\n1. PREPROCESSING IMPACT ON LOGISTIC REGRESSION:")
        print(f"   - Linear-Optimized: {lr_linear:.4f} AUC")
        print(f"   - Tree-Optimized:   {lr_tree:.4f} AUC")
        print(f"   - Δ AUC: {lr_delta:+.4f} ({lr_delta*100:+.1f}%)")
        print(f"   → Interpretation: {'Large effect' if abs(lr_delta) > 0.05 else 'Moderate effect' if abs(lr_delta) > 0.02 else 'Small effect'}")

if 'ablation' in results:
    df_abl = results['ablation']

    if len(df_abl) >= 2:
        auc_with_intrate = df_abl.iloc[0]['LR ROC-AUC']
        auc_without_intrate = df_abl.iloc[1]['LR ROC-AUC']
        delta_abl = auc_with_intrate - auc_without_intrate

        print(f"\n2. FEATURE DEPENDENCY (int_rate + installment):")
        print(f"   - With LC pricing (int_rate + installment): {auc_with_intrate:.4f} AUC")
        print(f"   - Applicant data only: {auc_without_intrate:.4f} AUC")
        print(f"   - Δ AUC: {delta_abl:+.4f} ({delta_abl*100:+.1f}%)")
        print(f"   → Interpretation: {'Heavy dependency' if delta_abl > 0.15 else 'Moderate dependency' if delta_abl > 0.05 else 'Weak dependency'} on LC's pricing")

if 'rolling' in results and results['rolling'] is not None:
    df_roll = results['rolling']

    if len(df_roll) > 0:
        mean_auc = df_roll['ROC-AUC'].mean()
        std_auc = df_roll['ROC-AUC'].std()
        cv = std_auc / mean_auc

        print(f"\n3. MODEL STABILITY ACROSS TIME:")
        print(f"   - Mean AUC (2016-2018): {mean_auc:.4f}")
        print(f"   - Std Dev: {std_auc:.4f}")
        print(f"   - Coefficient of Variation: {cv:.4f}")
        print(f"   → Interpretation: {'Excellent stability' if cv < 0.02 else 'Good stability' if cv < 0.05 else 'Moderate stability'} across years")

# %% [markdown]
# ## Step 6: Update claude.md with Results
#
# After running experiments, update the "Expected Performance" section in claude.md:
#
# ```markdown
# ## Expected Performance (Observed from Experiments)
#
# **Dual-Pipeline Results:**
# - Logistic Regression improves [ΔAUC] with preprocessing (large effect)
# - Tree models improve [ΔAUC] with preprocessing (small effect)
# - Indicates high-quality feature engineering
#
# **Ablation Study:**
# - Applicant-only features: [AUC] (pure applicant data capability)
# - With LC's pricing: [AUC] (refined assessment)
# - int_rate + installment contribute [ΔAUC] AUC
#
# **Rolling Backtest Stability:**
# - Mean AUC: [value] ± [std]
# - Consistent across 2016-2018 vintages despite concept drift
# ```

# %% [markdown]
# ## Step 7: Optional - Save Results to CSV
#
# For documentation and portfolio sharing:

# %%
import os
from datetime import datetime

results_dir = 'experiment_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

if 'dual_pipeline' in results:
    results['dual_pipeline'].to_csv(
        f'{results_dir}/dual_pipeline_{timestamp}.csv',
        index=False
    )
    print(f"✓ Saved dual-pipeline results to {results_dir}/dual_pipeline_{timestamp}.csv")

if 'ablation' in results:
    results['ablation'].to_csv(
        f'{results_dir}/ablation_{timestamp}.csv',
        index=False
    )
    print(f"✓ Saved ablation results to {results_dir}/ablation_{timestamp}.csv")

if 'rolling' in results and results['rolling'] is not None:
    results['rolling'].to_csv(
        f'{results_dir}/rolling_backtest_{timestamp}.csv',
        index=False
    )
    print(f"✓ Saved rolling backtest results to {results_dir}/rolling_backtest_{timestamp}.csv")

# %% [markdown]
# ## Summary
#
# You have now:
# 1. ✓ Run dual-pipeline experiments (4 models × 2 pipelines = 8 combinations)
# 2. ✓ Run ablation study (measured int_rate + installment contribution)
# 3. ✓ Run rolling backtest (validated stability across 2016-2018)
# 4. ✓ Extracted key findings and effect sizes
# 5. ✓ Saved results for portfolio documentation
#
# Next steps:
# - Update claude.md with observed results
# - Update project narrative with honest assessment
# - Commit experiments and results to git
# - Convert to final Jupyter notebooks for portfolio
