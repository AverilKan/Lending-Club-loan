"""
Contains the predictor class for the credit risk model.
Encapsulates loading the model pipeline and making predictions.
"""

import os
import joblib
# import pickle # No longer needed for feature list
import pandas as pd
import numpy as np
# Need to import custom transformers so joblib can find their definitions
from src.transformers import EmpLengthConverter, CreditHistoryCalculator, CountBinarizer

# Define default artifact filenames (should match saving step in notebook)
DEFAULT_MODEL_FILENAME = 'credit_risk_pipeline_v1.joblib'
# DEFAULT_FEATURES_FILENAME = 'credit_risk_features_v1.pkl' # Feature list no longer loaded here

# --- Potentially maps to SageMaker's model_fn() ---
def load_model_artifacts(artifact_path='.'):
    """Loads the pipeline artifact."""
    model_file = os.path.join(artifact_path, DEFAULT_MODEL_FILENAME)
    
    print(f"Loading pipeline from: {model_file}")
    pipeline = joblib.load(model_file)
    print("Pipeline loaded successfully.")
    
    # print(f"Loading feature list from: {features_file}") # Removed
    # with open(features_file, 'rb') as f: # Removed
    #     features = pickle.load(f) # Removed
    # print("Feature list loaded successfully.") # Removed
    
    # return pipeline, features # Return only pipeline
    return pipeline

class CreditPredictor:
    """
    Manages loading the credit risk model pipeline and making predictions.
    Assumes the loaded pipeline handles all necessary preprocessing and feature engineering.
    """
    def __init__(self, artifact_path='.'):
        """
        Initializes the predictor by loading the pipeline.

        Args:
            artifact_path (str): Path to the directory containing model artifacts.
        """
        self.pipeline = None
        # self.features = None # Feature list no longer stored in predictor
        self.artifact_path = artifact_path
        try:
            # self.pipeline, self.features = load_model_artifacts(self.artifact_path) # Load only pipeline
            self.pipeline = load_model_artifacts(self.artifact_path)
            print(f"CreditPredictor initialized with pipeline.")
            # print(f"CreditPredictor initialized. Expecting {len(self.features)} features.") # Removed feature count log
        except FileNotFoundError:
            print(f"Error: Model artifact not found in {self.artifact_path}. "
                  f"Expected \'{DEFAULT_MODEL_FILENAME}\'.")
            # Depending on use case, might want to raise an error here
        except Exception as e:
            print(f"Error initializing CreditPredictor: {e}")
            # Depending on use case, might want to raise an error here

    # --- Potentially maps to SageMaker's predict_fn() ---
    def predict_proba(self, input_data):
        """
        Generates prediction probabilities for the input data.
        Assumes the loaded pipeline handles all preprocessing.

        Args:
            input_data (pd.DataFrame or list[dict]): Raw input data for prediction.
                                                   Must contain columns expected by the start of the pipeline.

        Returns:
            np.array: Array of probabilities for the positive class (is_bad=1).
                      Returns None if predictor is not initialized or input is invalid/pipeline fails.
        """
        # if self.pipeline is None or self.features is None: # Check only pipeline
        if self.pipeline is None:
            print("Error: Predictor pipeline not initialized. Cannot predict.")
            return None

        try:
            # --- Potentially maps to SageMaker's input_fn() transformation ---
            if isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)
            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data.copy()
            else:
                raise ValueError("Input data must be a pandas DataFrame or a list of dictionaries.")

            print(f"Received {input_df.shape[0]} records for prediction.")
            print("Input DataFrame columns:", input_df.columns.tolist()) # DEBUG
            print("Input DataFrame dtypes:\n", input_df.dtypes) # DEBUG


            # --- Prediction ---
            # The pipeline handles FE, imputation, scaling, encoding, and prediction
            probabilities = self.pipeline.predict_proba(input_df) # Pass raw DataFrame

            # Return probability of the positive class (usually index 1)
            positive_class_proba = probabilities[:, 1]
            print("Prediction probabilities generated.")

            # --- Potentially maps to SageMaker's output_fn() ---
            return positive_class_proba

        except KeyError as e:
             print(f"Error during prediction: Missing expected raw feature column: {e}")
             return None
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

    def predict(self, input_data, threshold=0.5):
        """
        Generates binary class predictions (0 or 1).

        Args:
            input_data (pd.DataFrame or list[dict]): Raw input data.
            threshold (float): Probability threshold for classifying as positive (1).

        Returns:
            np.array: Array of binary predictions (0 or 1). Returns None on error.
        """
        probabilities = self.predict_proba(input_data)
        if probabilities is None:
            return None

        return (probabilities >= threshold).astype(int)

# Example Usage (Optional - for testing within this script)
if __name__ == '__main__':
    # Assumes artifacts are in the project root directory AND this script is run FROM the root.
    # Ensure the saved pipeline ('credit_risk_pipeline_v1.joblib') includes FE steps.
    predictor = CreditPredictor(artifact_path='.') # Path relative to execution directory (root)

    if predictor.pipeline is not None:
        # Create RAW sample data (matching format BEFORE FE/preprocessing in Notebook 2)
        # Crucially includes raw fields needed for FE steps (like emp_length string, dates)
        sample_data = [
            {
                'loan_amnt': 10000, 'term': ' 36 months', 'int_rate': 12.0,
                'installment': 300.0, 'grade': 'B', 'sub_grade': 'B2',
                'emp_length': '10+ years', # Raw string format
                'home_ownership': 'MORTGAGE',
                'annual_inc': 75000, 'verification_status': 'Verified',
                # 'issue_d': pd.to_datetime('Dec-2015'), # Example date - NEEDED if FE requires it
                # 'earliest_cr_line': pd.to_datetime('Aug-2003'), # Example date - NEEDED if FE requires it
                'purpose': 'debt_consolidation', 'addr_state': 'CA', 'dti': 15.0,
                'delinq_2yrs': 0, 'fico_range_low': 700, 'fico_range_high': 704,
                'inq_last_6mths': 1, 'open_acc': 10,
                'pub_rec': 0, # Raw count
                'revol_bal': 15000, 'revol_util': 50.0, 'total_acc': 25,
                'initial_list_status': 'f',
                'collections_12_mths_ex_med': 0, 'application_type': 'Individual',
                'mort_acc': 2, # Raw count
                'pub_rec_bankruptcies': 0, # Raw count
                'acc_now_delinq': 0, 'tot_coll_amt': 0, 'tot_cur_bal': 250000,
                'total_rev_hi_lim': 30000, 'acc_open_past_24mths': 3,
                'avg_cur_bal': 25000, 'bc_open_to_buy': 5000, 'bc_util': 60.0,
                'chargeoff_within_12_mths': 0, 'delinq_amnt': 0,
                'mo_sin_old_il_acct': 120, 'mo_sin_old_rev_tl_op': 150,
                'mo_sin_rcnt_rev_tl_op': 10, 'mo_sin_rcnt_tl': 5,
                'mths_since_recent_bc': 15, 'mths_since_recent_inq': 6,
                'num_accts_ever_120_pd': 0, 'num_actv_bc_tl': 4, 'num_actv_rev_tl': 6,
                'num_bc_sats': 5, 'num_bc_tl': 8, 'num_il_tl': 10,
                'num_op_rev_tl': 12, 'num_rev_accts': 15, 'num_rev_tl_bal_gt_0': 6,
                'num_sats': 10, 'num_tl_120dpd_2m': 0, 'num_tl_30dpd': 0,
                'num_tl_90g_dpd_24m': 0, 'num_tl_op_past_12m': 2,
                'pct_tl_nvr_dlq': 95.0, 'percent_bc_gt_75': 25.0, 'tax_liens': 0,
                'tot_hi_cred_lim': 300000, 'total_bal_ex_mort': 50000,
                'total_bc_limit': 15000, 'total_il_high_credit_limit': 60000,
                'disbursement_method': 'Cash',

            },
            # Add more samples if needed
        ]

        # Convert sample data to DataFrame for prediction
        sample_df = pd.DataFrame(sample_data)

        # Important: Need to parse date columns if the FE step in the pipeline expects datetime objects
        # This parsing might ideally happen within the pipeline using a custom transformer,
        # but for now, we'll do it before calling predict if needed.
        date_cols_needed_by_pipeline = ['issue_d', 'earliest_cr_line'] # Example
        for col in date_cols_needed_by_pipeline:
             if col in sample_df.columns:
                 try:
                     sample_df[col] = pd.to_datetime(sample_df[col], errors='coerce')
                 except Exception as e:
                     print(f"Warning: Could not parse date column '{col}' in sample data: {e}")


        print("\n--- Making Sample Prediction ---")
        probabilities = predictor.predict_proba(sample_df)
        predictions = predictor.predict(sample_df)

        if probabilities is not None:
            print("Sample Probabilities (is_bad=1):")
            print(probabilities)
            print("Sample Predictions (0=Good, 1=Bad):")
            print(predictions)
