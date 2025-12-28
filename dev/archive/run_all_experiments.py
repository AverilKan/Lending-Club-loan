# %% [markdown]
# # Master Experiment Runner: Dual-Pipeline, Ablation, Rolling Backtest
#
# This script orchestrates all three critical experiments to validate the
# modeling approach and provide rigorous evidence for project claims.
#
# **Execution Order:**
# 1. Dual-Pipeline Experiment (4 models × 2 pipelines = 8 combinations)
# 2. Ablation Study (Feature dependency: with vs without int_rate+installment)
# 3. Rolling Backtest (Stability across 2016, 2017, 2018)
#
# **Requirements:**
# - Run AFTER data loading and preprocessing setup in dev/2_Modelling.py
# - Requires: X_train, X_test, y_train, y_test (time-based splits)
# - Requires: skewed_num_cols, standard_num_cols, ohe_cat_cols
# - Requires: Custom transformers available

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import os, sys

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

try:
    from src.transformers import EmpLengthConverter, CreditHistoryCalculator, InterestRateRiskTierTransformer
except ImportError:
    print("Warning: Custom transformers not available - ensure src.transformers is in path")

warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

# =============================================================================
# IMPORT EXPERIMENT MODULES
# =============================================================================

# These would normally be imported from separate modules:
# from experiments_dual_pipeline import run_dual_pipeline_experiments, create_linear_optimized_pipeline, create_tree_optimized_pipeline
# from experiments_ablation import run_ablation_study
# from experiments_rolling_backtest import run_rolling_backtest, plot_rolling_backtest

# For now, we'll define them inline or import as needed

# =============================================================================
# MASTER EXPERIMENT RUNNER
# =============================================================================

class ExperimentRunner:
    """Orchestrates all three critical experiments."""

    def __init__(self, X_train, X_test, y_train, y_test, df_original,
                 skewed_num_cols, standard_num_cols, ohe_cat_cols):
        """
        Initialize experiment runner with preprocessed data.

        Parameters:
        -----------
        X_train, X_test: Feature matrices (already preprocessed by custom FE)
        y_train, y_test: Target vectors (time-based split: train ≤2015, test ≥2016)
        df_original: Original dataframe with 'issue_d' for rolling backtest
        skewed_num_cols, standard_num_cols, ohe_cat_cols: Feature subsets
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.df_original = df_original
        self.skewed_num_cols = skewed_num_cols
        self.standard_num_cols = standard_num_cols
        self.ohe_cat_cols = ohe_cat_cols

        self.results = {}
        self.pipelines = {}

        print("=" * 80)
        print("MASTER EXPERIMENT RUNNER INITIALIZED")
        print("=" * 80)
        print(f"\nData Summary:")
        print(f"  Train: X={X_train.shape}, y={y_train.shape}")
        print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
        print(f"  Class balance (train): {y_train.mean()*100:.1f}% positive")
        print(f"  Class balance (test):  {y_test.mean()*100:.1f}% positive")

    def create_linear_optimized_pipeline(self):
        """Create preprocessing pipeline optimized for linear models."""
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
                ('num_skewed', skewed_num_pipe, self.skewed_num_cols),
                ('num_standard', standard_num_pipe, self.standard_num_cols),
                ('cat_ohe', ohe_cat_pipe, self.ohe_cat_cols)
            ],
            remainder='drop'
        )
        self.pipelines['linear'] = preprocessor
        return preprocessor

    def create_tree_optimized_pipeline(self):
        """Create preprocessing pipeline optimized for tree models."""
        numeric_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])

        ohe_cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('numeric', numeric_pipe, self.skewed_num_cols + self.standard_num_cols),
                ('cat_ohe', ohe_cat_pipe, self.ohe_cat_cols)
            ],
            remainder='drop'
        )
        self.pipelines['tree'] = preprocessor
        return preprocessor

    def run_experiment_1_dual_pipeline(self):
        """Experiment 1: Dual-Pipeline Matrix."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: DUAL-PIPELINE MATRIX")
        print("=" * 80)

        # Create pipelines
        lr_pipe = self.create_linear_optimized_pipeline()
        tree_pipe = self.create_tree_optimized_pipeline()

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, min_samples_split=100,
                                                    min_samples_leaf=50, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                    min_samples_split=100, random_state=42),
            'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
        }

        pipelines = {'Linear-Optimized': lr_pipe, 'Tree-Optimized': tree_pipe}

        results = []
        start_time = time.time()

        for model_name, model in models.items():
            for pipeline_name, preprocessor in pipelines.items():
                print(f"{model_name:20s} × {pipeline_name:20s}...", end=" ")

                # Full pipeline
                full_pipeline = Pipeline([
                    ('feature_engineering_emp', EmpLengthConverter()),
                    ('feature_engineering_hist', CreditHistoryCalculator()),
                    ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
                    ('preprocessing', preprocessor),
                    ('classifier', model)
                ])

                full_pipeline.fit(self.X_train, self.y_train)
                y_pred_proba = full_pipeline.predict_proba(self.X_test)[:, 1]
                test_auc = roc_auc_score(self.y_test, y_pred_proba)
                test_pr = average_precision_score(self.y_test, y_pred_proba)

                results.append({
                    'Model': model_name,
                    'Pipeline': pipeline_name,
                    'Test ROC-AUC': test_auc,
                    'Test PR-AUC': test_pr
                })

                print(f"AUC={test_auc:.4f}")

        df_results = pd.DataFrame(results)
        self.results['dual_pipeline'] = df_results

        # Create pivot summary
        pivot = df_results.pivot_table(index='Model', columns='Pipeline', values='Test ROC-AUC')
        pivot['ΔAUC'] = pivot['Linear-Optimized'] - pivot['Tree-Optimized']

        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print(pivot.to_string())

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f} seconds")

        return df_results

    def run_experiment_2_ablation(self):
        """Experiment 2: Ablation Study."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: ABLATION STUDY")
        print("=" * 80)

        # Use best pipelines from dual-pipeline experiment
        lr_pipe = self.pipelines['linear']
        tree_pipe = self.pipelines['tree']

        # Feature sets
        all_features = self.X_train.columns.tolist()
        features_to_remove = {'int_rate', 'installment'}
        applicant_only = [f for f in all_features if f not in features_to_remove]

        results = []
        start_time = time.time()

        for feature_set_name, features in [
            ('Investor (with LC pricing)', all_features),
            ('Applicant Only (no LC pricing)', applicant_only)
        ]:

            X_train_abl = self.X_train[features]
            X_test_abl = self.X_test[features]

            print(f"\n{feature_set_name} ({len(features)} features):")

            # LR
            print(f"  Logistic Regression...", end=" ")
            lr_pipeline = Pipeline([
                ('feature_engineering_emp', EmpLengthConverter()),
                ('feature_engineering_hist', CreditHistoryCalculator()),
                ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
                ('preprocessing', lr_pipe),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ])
            lr_pipeline.fit(X_train_abl, self.y_train)
            lr_auc = roc_auc_score(self.y_test, lr_pipeline.predict_proba(X_test_abl)[:, 1])
            print(f"AUC={lr_auc:.4f}")

            # LGBM
            print(f"  LightGBM...", end=" ")
            lgbm_pipeline = Pipeline([
                ('feature_engineering_emp', EmpLengthConverter()),
                ('feature_engineering_hist', CreditHistoryCalculator()),
                ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
                ('preprocessing', tree_pipe),
                ('classifier', LGBMClassifier(random_state=42, verbose=-1))
            ])
            lgbm_pipeline.fit(X_train_abl, self.y_train)
            lgbm_auc = roc_auc_score(self.y_test, lgbm_pipeline.predict_proba(X_test_abl)[:, 1])
            print(f"AUC={lgbm_auc:.4f}")

            results.append({
                'Feature Set': feature_set_name,
                'LR AUC': lr_auc,
                'LGBM AUC': lgbm_auc
            })

        df_ablation = pd.DataFrame(results)
        self.results['ablation'] = df_ablation

        # Calculate deltas
        lr_delta = df_ablation.loc[0, 'LR AUC'] - df_ablation.loc[1, 'LR AUC']
        lgbm_delta = df_ablation.loc[0, 'LGBM AUC'] - df_ablation.loc[1, 'LGBM AUC']

        print("\n" + "=" * 80)
        print("ABLATION RESULTS")
        print("=" * 80)
        print(df_ablation.to_string(index=False))
        print(f"\nAUC Lift from int_rate + installment:")
        print(f"  LR:   {lr_delta:+.4f}")
        print(f"  LGBM: {lgbm_delta:+.4f}")

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f} seconds")

        return df_ablation

    def run_experiment_3_rolling_backtest(self):
        """Experiment 3: Rolling Backtest."""
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: ROLLING BACKTEST")
        print("=" * 80)

        # Use best tree pipeline
        tree_pipe = self.pipelines['tree']
        feature_cols = self.X_train.columns.tolist()

        # Ensure issue_d exists
        if 'issue_d' not in self.df_original.columns:
            print("⚠️  'issue_d' not found in dataframe - rolling backtest skipped")
            return None

        df_with_date = self.df_original.copy()
        df_with_date['issue_d'] = pd.to_datetime(df_with_date['issue_d'])

        results = []
        start_time = time.time()

        for test_year in [2016, 2017, 2018]:
            print(f"\nYear {test_year}...", end=" ")

            train_mask = df_with_date['issue_d'].dt.year < test_year
            test_mask = df_with_date['issue_d'].dt.year == test_year

            X_train_roll = self.X_train.loc[train_mask, feature_cols]
            y_train_roll = self.y_train.loc[train_mask]
            X_test_roll = self.X_test.loc[test_mask, feature_cols]
            y_test_roll = self.y_test.loc[test_mask]

            if len(y_test_roll) == 0:
                print("SKIP")
                continue

            # Train LightGBM
            lgbm_pipeline = Pipeline([
                ('feature_engineering_emp', EmpLengthConverter()),
                ('feature_engineering_hist', CreditHistoryCalculator()),
                ('feature_engineering_intrate', InterestRateRiskTierTransformer()),
                ('preprocessing', tree_pipe),
                ('classifier', LGBMClassifier(random_state=42, verbose=-1))
            ])

            lgbm_pipeline.fit(X_train_roll, y_train_roll)
            y_pred_proba = lgbm_pipeline.predict_proba(X_test_roll)[:, 1]
            test_auc = roc_auc_score(y_test_roll, y_pred_proba)

            results.append({
                'Year': test_year,
                'N Loans': len(y_test_roll),
                'Default Rate': y_test_roll.mean(),
                'ROC-AUC': test_auc
            })

            print(f"N={len(y_test_roll):,}, Default={y_test_roll.mean()*100:.1f}%, AUC={test_auc:.4f}")

        df_rolling = pd.DataFrame(results)
        self.results['rolling'] = df_rolling

        if len(df_rolling) > 0:
            print("\n" + "=" * 80)
            print("ROLLING BACKTEST RESULTS")
            print("=" * 80)
            print(df_rolling.to_string(index=False))
            print(f"\nStability: Mean AUC = {df_rolling['ROC-AUC'].mean():.4f} ± {df_rolling['ROC-AUC'].std():.4f}")

        elapsed = time.time() - start_time
        print(f"\nCompleted in {elapsed:.1f} seconds")

        return df_rolling

    def run_all_experiments(self):
        """Run all three experiments in sequence."""
        print("\n" + "=" * 80)
        print("RUNNING ALL EXPERIMENTS")
        print("=" * 80)

        start_all = time.time()

        # Experiment 1
        self.run_experiment_1_dual_pipeline()

        # Experiment 2
        self.run_experiment_2_ablation()

        # Experiment 3
        try:
            self.run_experiment_3_rolling_backtest()
        except Exception as e:
            print(f"Rolling backtest failed: {e}")

        total_time = time.time() - start_all

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED")
        print("=" * 80)
        print(f"\nTotal Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print("\nResults Summary:")
        for key in self.results.keys():
            print(f"  ✓ {key}")

        return self.results


# %% [markdown]
# # USAGE
#
# After running Phase 2 data preparation in dev/2_Modelling.py, run:
#
# ```python
# # Create runner
# runner = ExperimentRunner(
#     X_train, X_test, y_train, y_test, df,
#     skewed_num_cols, standard_num_cols, ohe_cat_cols
# )
#
# # Run all experiments
# results = runner.run_all_experiments()
# ```
