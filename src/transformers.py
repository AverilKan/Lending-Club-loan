import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector

# Custom transformer for emp_length conversion
class EmpLengthConverter(BaseEstimator, TransformerMixin):
    def __init__(self):
        # No parameters needed for this transformer
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        emp_length_mapping = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }
        
        if 'emp_length' in X_copy.columns:
            X_copy['emp_length_num'] = X_copy['emp_length'].map(emp_length_mapping)
            # Drop the original column
            X_copy = X_copy.drop('emp_length', axis=1) # Assign back instead of inplace
        else:
            pass # Silently pass
            
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Returns feature names after transformation."""
        if input_features is None:
            return np.array(['emp_length_num'], dtype=object)

        input_features = np.asarray(input_features, dtype=object)
        if 'emp_length' in input_features:
            # Correctly removes 'emp_length' and adds 'emp_length_num'
            return np.array([f for f in input_features if f != 'emp_length'] + ['emp_length_num'], dtype=object)
        else:
            # Returns input features unchanged if 'emp_length' wasn't present
            return input_features

# Custom transformer for credit history calculation
class CreditHistoryCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, issue_col='issue_d', earliest_col='earliest_cr_line'):
        self.issue_col = issue_col
        self.earliest_col = earliest_col
        self.output_col_ = 'credit_hist_years' # <-- Defined here
        self._input_features = None

    def fit(self, X, y=None):
        self._input_features = list(X.columns)
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Expected date format: Mon-YYYY (e.g., Dec-2016)
        date_format = '%b-%Y'

        if self.issue_col in X_copy.columns and self.earliest_col in X_copy.columns:
            # Attempt conversion, providing the format
            try:
                issue_dt = pd.to_datetime(X_copy[self.issue_col], format=date_format, errors='coerce')
                earliest_dt = pd.to_datetime(X_copy[self.earliest_col], format=date_format, errors='coerce')
            except Exception as e:
               print(f"Error parsing dates in CreditHistoryCalculator: {e}")
               X_copy[self.output_col_] = np.nan
               # Drop original columns even on error
               cols_to_drop = [c for c in [self.issue_col, self.earliest_col] if c in X_copy.columns]
               X_copy = X_copy.drop(columns=cols_to_drop) # Assign back
               # Impute the created NaN column immediately or let downstream imputer handle it?
               # Imputing here with 0 for consistency with original logic
               X_copy[self.output_col_] = X_copy[self.output_col_].fillna(0)
               return X_copy

            # Calculate difference in days, handling NaT
            time_diff = (issue_dt - earliest_dt)
            X_copy['credit_hist_days'] = time_diff.dt.days
            
            # Convert to years (approximate)
            X_copy[self.output_col_] = X_copy['credit_hist_days'] / 365.25
            
            # Handle potential negative values (data error) or NaNs from NaT dates
            X_copy.loc[X_copy[self.output_col_] < 0, self.output_col_] = 0 
            
            # Impute NaNs resulting from calculations or NaT dates
            # Using 0 as a default, could use median if calculated from fit
            X_copy[self.output_col_] = X_copy[self.output_col_].fillna(0) # Assign back instead of inplace
            
            # Drop intermediate and original date columns
            cols_to_drop = ['credit_hist_days', self.issue_col, self.earliest_col]
            cols_to_drop = [c for c in cols_to_drop if c in X_copy.columns]
            X_copy = X_copy.drop(columns=cols_to_drop) # Assign back
        else:
             # If columns are missing, create the output column filled with 0
             X_copy[self.output_col_] = 0
             pass # Silently pass
             
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Returns feature names after transformation."""
        # Use stored input features from fit if available and input_features is None
        if input_features is None and self._input_features is not None:
            input_features = self._input_features
        elif input_features is None:
            # Cannot determine output features without knowing input
            raise ValueError("input_features must be provided to get_feature_names_out if fit hasn't been called or didn't store features.")
        
        input_features = np.asarray(input_features, dtype=object)
        
        # Define columns that are dropped by this transformer
        cols_to_drop = [self.issue_col, self.earliest_col, 'credit_hist_days']
        
        # Start with input features, remove dropped ones, add the new one
        output_features = [f for f in input_features if f not in cols_to_drop]
        
        # Add the new feature only if the required input columns were present
        if self.issue_col in input_features and self.earliest_col in input_features:
             if self.output_col_ not in output_features:
                  output_features.append(self.output_col_)
        # Handle case where input cols missing: output col is added but filled with 0 in transform
        elif self.output_col_ not in output_features:
             output_features.append(self.output_col_)
            
        return np.array(output_features, dtype=object)

# Custom transformer for interest rate risk tiers
class InterestRateRiskTierTransformer(BaseEstimator, TransformerMixin):
    """
    Create interest rate risk tiers and binary flags.
    Captures non-linear relationship between interest rate and default risk.
    """

    def __init__(self):
        self.thresholds = {
            'low': 10.0,
            'medium': 15.0,
            'high': 20.0
        }
        self._input_features = None

    def fit(self, X, y=None):
        self._input_features = list(X.columns) if hasattr(X, 'columns') else None
        return self

    def transform(self, X):
        X_copy = X.copy()

        if 'int_rate' in X_copy.columns:
            # Create risk tiers
            X_copy['int_rate_tier'] = pd.cut(
                X_copy['int_rate'],
                bins=[-np.inf, self.thresholds['low'],
                      self.thresholds['medium'], self.thresholds['high'], np.inf],
                labels=['Low_Risk', 'Medium_Risk', 'High_Risk', 'Very_High_Risk']
            )

            # Binary high-risk indicators
            X_copy['high_int_rate'] = (X_copy['int_rate'] > self.thresholds['medium']).astype(int)
            X_copy['very_high_int_rate'] = (X_copy['int_rate'] > self.thresholds['high']).astype(int)
        else:
            # Graceful degradation if int_rate missing
            X_copy['int_rate_tier'] = 'Unknown'
            X_copy['high_int_rate'] = 0
            X_copy['very_high_int_rate'] = 0

        return X_copy

    def get_feature_names_out(self, input_features=None):
        if input_features is None and self._input_features is not None:
            input_features = self._input_features
        elif input_features is None:
            input_features = []

        input_features = np.asarray(input_features, dtype=object)
        new_features = ['int_rate_tier', 'high_int_rate', 'very_high_int_rate']
        output_features = list(input_features) + new_features

        return np.array(output_features, dtype=object)

