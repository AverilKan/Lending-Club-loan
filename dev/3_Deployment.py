# %% [markdown]
# # Phase 3: Deployment Simulation
# 
# Load the trained model pipeline (Phase 2 artifact) and demonstrate prediction on sample data using the `CreditPredictor` class.

# %% [markdown]
# ## 1. Setup and Imports

# %% 
import pandas as pd
import numpy as np
import os
import sys
import joblib # Use joblib as artifact was saved with it

# Add src directory to Python path
module_path = os.path.abspath(os.path.join('.', '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.predictor import CreditPredictor

# Explicitly import custom transformers needed for unpickling
try:
    from src.transformers import EmpLengthConverter, CreditHistoryCalculator, CountBinarizer
except ModuleNotFoundError:
    print("Warning: Could not import transformers directly.")

# %% [markdown]
# ## 2. Define Artifact Paths and Load Predictor

# %% 
# Define path to the saved pipeline artifact (Fixed path resolution)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir.endswith('/dev'):
        ARTIFACT_PATH = os.path.dirname(script_dir)  # Go up to project root
    else:
        ARTIFACT_PATH = script_dir
except NameError:
    # Handle interactive execution
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == 'dev':
        ARTIFACT_PATH = os.path.dirname(current_dir)  # Go up to project root
    else:
        ARTIFACT_PATH = current_dir

print(f"Artifact path: {ARTIFACT_PATH}") 

# Instantiate the predictor (loads the pipeline artifact)
try:
    predictor = CreditPredictor(artifact_path=ARTIFACT_PATH)
    print("CreditPredictor initialized and pipeline loaded.")
except Exception as e:
    print(f"Error loading predictor: {e}")
    predictor = None

# %% [markdown]
# ## 3. Prepare Sample Input Data for Prediction

# Input data must be in the raw format expected by the *original* pipeline (before FE/preprocessing).

# %% [markdown]
# ### 3.A Define Sample Data Directly (Hardcoded)

# %% 
# Single sample application data
sample_data_hardcoded = [
    {
        'loan_amnt': 15000, 'funded_amnt': 15000, 'funded_amnt_inv': 15000, 
        'term': ' 36 months', 'int_rate': 10.5, 'installment': 487.5, 
        'grade': 'B', 'sub_grade': 'B3', 'emp_length': '5 years', 
        'home_ownership': 'RENT', 'annual_inc': 65000.0, 'verification_status': 'Source Verified', 
        'issue_d': 'Dec-2016', # Needed for CreditHistoryCalculator
        'purpose': 'debt_consolidation', 'addr_state': 'NY', 'dti': 22.5, 
        'delinq_2yrs': 0.0, 
        'earliest_cr_line': 'Aug-2008', # Needed for CreditHistoryCalculator
        'fico_range_low': 680.0, 'fico_range_high': 684.0, 'inq_last_6mths': 0.0, 
        'open_acc': 12.0, 
        'pub_rec': 0.0, # Needed for CountBinarizer
        'revol_bal': 18000.0, 'revol_util': 75.2, 'total_acc': 30.0, 
        'initial_list_status': 'w', 'collections_12_mths_ex_med': 0.0, 
        'application_type': 'Individual', 'acc_now_delinq': 0.0, 
        'tot_coll_amt': 0.0, 'tot_cur_bal': 150000.0, 'total_rev_hi_lim': 25000.0, 
        'acc_open_past_24mths': 4.0, 'avg_cur_bal': 12500.0, 'bc_open_to_buy': 3000.0, 
        'bc_util': 80.0, 'chargeoff_within_12_mths': 0.0, 'delinq_amnt': 0.0, 
        'mo_sin_old_il_acct': 120.0, 'mo_sin_old_rev_tl_op': 150.0, 
        'mo_sin_rcnt_rev_tl_op': 5.0, 'mo_sin_rcnt_tl': 5.0, 
        'mort_acc': 1.0, # Needed for CountBinarizer
        'mths_since_recent_bc': 10.0, 'mths_since_recent_inq': 3.0, 
        'num_accts_ever_120_pd': 0.0, 'num_actv_bc_tl': 4.0, 'num_actv_rev_tl': 6.0, 
        'num_bc_sats': 4.0, 'num_bc_tl': 8.0, 'num_il_tl': 10.0, 'num_op_rev_tl': 6.0, 
        'num_rev_accts': 15.0, 'num_rev_tl_bal_gt_0': 6.0, 'num_sats': 12.0, 
        'num_tl_120dpd_2m': 0.0, 'num_tl_30dpd': 0.0, 'num_tl_90g_dpd_24m': 0.0, 
        'num_tl_op_past_12m': 2.0, 'pct_tl_nvr_dlq': 100.0, 'percent_bc_gt_75': 75.0, 
        'pub_rec_bankruptcies': 0.0, # Needed for CountBinarizer
        'tax_liens': 0.0, 'tot_hi_cred_lim': 180000.0, 'total_bal_ex_mort': 40000.0, 
        'total_bc_limit': 15000.0, 'total_il_high_credit_limit': 30000.0, 
        'disbursement_method': 'Cash' 
        # Ensure all 69 features expected by the pipeline are present
    }
]
sample_df_hardcoded = pd.DataFrame(sample_data_hardcoded)

print("Hardcoded Sample DataFrame prepared.")
print(f"Shape: {sample_df_hardcoded.shape}")

# %% [markdown]
# ### 3.B Load Sample Data from CSV

# %% 
# Construct path to sample CSV (Fixed path resolution)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # If running from dev/ directory, go up one level to project root
    if script_dir.endswith('/dev'):
        project_root = os.path.dirname(script_dir)
    else:
        project_root = script_dir
except NameError: 
    # Handle interactive execution - assume current working directory is project root
    project_root = os.getcwd()
    # If in dev subdirectory, go up one level
    if os.path.basename(project_root) == 'dev':
        project_root = os.path.dirname(project_root)

SAMPLE_CSV_PATH = os.path.join(project_root, 'data', 'sample_applications.csv')
print(f"\nPath to sample CSV: {SAMPLE_CSV_PATH}")
print(f"File exists: {os.path.exists(SAMPLE_CSV_PATH)}")

sample_df_csv = None
try:
    # Specify dtype for potential mixed-type columns if known
    sample_df_csv = pd.read_csv(SAMPLE_CSV_PATH)
    print(f"Successfully loaded sample data from CSV.")
    print(f"CSV Sample DataFrame shape: {sample_df_csv.shape}")
except FileNotFoundError:
    print(f"Error: Sample CSV file not found at {SAMPLE_CSV_PATH}.")
except Exception as e:
    print(f"Error loading sample CSV: {e}.")


# %% [markdown]
# ## 4. Make Predictions

# Use the loaded predictor to generate probabilities and binary classifications.

# %% [markdown]
# ### 4.A Predictions using Hardcoded Sample

# %% 
print("\n--- Predictions (Hardcoded Sample) ---")
if predictor is not None and predictor.pipeline is not None and sample_df_hardcoded is not None:
    try:
        probabilities_hardcoded = predictor.predict_proba(sample_df_hardcoded) 
        # Note: predict() method likely uses default 0.5 threshold unless modified in CreditPredictor
        predictions_hardcoded = predictor.predict(sample_df_hardcoded)
        
        print(f"Predicted Probability (is_bad=1): {probabilities_hardcoded[0]:.4f}")
        print(f"Binary Prediction (0=Good, 1=Bad): {predictions_hardcoded[0]}")
    except Exception as e:
        print(f"Error during prediction: {e}")
else:
    print("Skipping prediction: Predictor or sample data not available.")

# %% [markdown]
# ### 4.B Predictions using CSV Sample

# %% 
print("\n--- Predictions (CSV Sample) ---")
if predictor is not None and predictor.pipeline is not None and sample_df_csv is not None:
    try:
        probabilities_csv = predictor.predict_proba(sample_df_csv) 
        predictions_csv = predictor.predict(sample_df_csv)
        
        print("Predicted Probabilities (is_bad=1) for CSV records:")
        for i, prob in enumerate(probabilities_csv):
            print(f"  Record {i}: {prob:.4f}")
            
        print("\nBinary Predictions (0=Good, 1=Bad) for CSV records:")
        for i, pred in enumerate(predictions_csv):
            print(f"  Record {i}: {pred}")
            
    except Exception as e:
        print(f"Error during prediction on CSV data: {e}")
        
elif predictor is None or predictor.pipeline is None:
    print("Skipping CSV prediction: Predictor not initialized.")
else: 
    print("Skipping CSV prediction: Sample CSV data not loaded.")

# %% [markdown]
# ## 5. Discussion: Mapping to SageMaker Structure
# 
# This section discusses how the structure of the local `CreditPredictor` conceptually aligns with AWS SageMaker inference components.
# 
# - **`CreditPredictor.__init__()` & internal loading:** Maps to SageMaker's `model_fn(model_dir)`, responsible for loading model artifacts from the deployment environment.
# 
# - **Input Data Handling (within `predict_proba`/`predict`):** Internal checks/formatting (e.g., ensuring DataFrame, checking columns via schema validation added previously) map to SageMaker's `input_fn(request_body, request_content_type)`, which deserializes and prepares incoming request data.
# 
# - **Core Prediction (`self.pipeline.predict_proba(input_df)`):** Maps directly to SageMaker's `predict_fn(input_data, model)`, which performs inference using the loaded model and prepared data.
# 
# - **Output Formatting (Return value):** Maps to SageMaker's `output_fn(prediction, response_content_type)`, which serializes prediction results into the desired response format (e.g., JSON).
# 
# Organising local code within `CreditPredictor` facilitates adaptation to platforms like SageMaker by mapping these logical steps to the platform's required function signatures (e.g., in an `inference.py` script).

# %% [markdown]
# --- End of Phase 3: Deployment Simulation ---