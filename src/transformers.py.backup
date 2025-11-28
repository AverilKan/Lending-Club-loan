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

# Custom transformer for binarizing count features
class CountBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        # Default columns to binarize
        self.columns = columns if columns is not None else ['pub_rec', 'mort_acc', 'pub_rec_bankruptcies']
        self._input_features = None
        self._output_features = None

    def fit(self, X, y=None):
        self._input_features = list(X.columns)
        # Determine output features based on input during fit
        output_features_list = list(self._input_features)
        self._new_cols = []
        for col in self.columns:
            if col in output_features_list:
                output_features_list.remove(col)
                new_col_name = f'{col}_binary'
                if new_col_name not in output_features_list:
                    output_features_list.append(new_col_name)
                    self._new_cols.append(new_col_name)
            else:
                 # If col not present, ensure dummy output col is tracked if created in transform
                 new_col_name = f'{col}_binary'
                 if new_col_name not in output_features_list:
                    output_features_list.append(new_col_name)
                    self._new_cols.append(new_col_name)
                    
        self._output_features = np.array(output_features_list, dtype=object)
        return self

    def transform(self, X):
        X_copy = X.copy()
        cols_to_drop = []
        for col in self.columns:
            if col in X_copy.columns:
                new_col_name = f'{col}_binary'
                # Ensure column is numeric before comparison
                numeric_col = pd.to_numeric(X_copy[col], errors='coerce')
                X_copy[new_col_name] = numeric_col.apply(lambda x: 1 if pd.notna(x) and x > 0 else 0)
                # Mark original column for dropping
                cols_to_drop.append(col)
            else:
                # Create dummy column if expected downstream, fill with 0
                new_col_name = f'{col}_binary'
                X_copy[new_col_name] = 0
                pass # Silently pass
        
        # Drop original columns after iteration
        if cols_to_drop:
            X_copy = X_copy.drop(columns=cols_to_drop) # Assign back
            
        return X_copy

    def get_feature_names_out(self, input_features=None):
        """Returns feature names after transformation."""
        # If fit was called, return the stored output features
        if self._output_features is not None:
            return self._output_features
            
        # Fallback if fit wasn't called (might be less accurate)
        if input_features is None:
            raise ValueError("input_features must be provided to get_feature_names_out if fit hasn't been called.")
            
        input_features = np.asarray(input_features, dtype=object)
        output_features_list = list(input_features)
        for col in self.columns:
            if col in output_features_list:
                output_features_list.remove(col)
                new_col_name = f'{col}_binary'
                if new_col_name not in output_features_list:
                    output_features_list.append(new_col_name)
            else:
                 new_col_name = f'{col}_binary'
                 if new_col_name not in output_features_list:
                    output_features_list.append(new_col_name)
                    
        return np.array(output_features_list, dtype=object) 

# Custom transformer for FICO-based risk tiers
class FICORiskTierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, fico_low_col='fico_range_low', fico_high_col='fico_range_high'):
        self.fico_low_col = fico_low_col
        self.fico_high_col = fico_high_col
        self._input_features = None
        
    def fit(self, X, y=None):
        self._input_features = list(X.columns)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Calculate average FICO score if both ranges available
        if self.fico_low_col in X_copy.columns and self.fico_high_col in X_copy.columns:
            X_copy['fico_avg'] = (X_copy[self.fico_low_col] + X_copy[self.fico_high_col]) / 2
        elif self.fico_low_col in X_copy.columns:
            X_copy['fico_avg'] = X_copy[self.fico_low_col]
        elif self.fico_high_col in X_copy.columns:
            X_copy['fico_avg'] = X_copy[self.fico_high_col]
        else:
            # If no FICO columns available, create dummy with median value
            X_copy['fico_avg'] = 680  # Typical median FICO
        
        # Create FICO-based risk tiers (based on standard credit score ranges)
        def get_fico_risk_tier(fico_score):
            if pd.isna(fico_score):
                return 'Unknown'
            elif fico_score >= 800:
                return 'Excellent'  # Very low risk
            elif fico_score >= 740:
                return 'Very Good'  # Low risk
            elif fico_score >= 670:
                return 'Good'       # Moderate risk
            elif fico_score >= 580:
                return 'Fair'       # High risk
            else:
                return 'Poor'       # Very high risk
        
        X_copy['fico_risk_tier'] = X_copy['fico_avg'].apply(get_fico_risk_tier)
        
        # Create binary indicators for high/low risk
        X_copy['fico_high_risk'] = (X_copy['fico_avg'] < 650).astype(int)
        X_copy['fico_excellent'] = (X_copy['fico_avg'] >= 800).astype(int)
        
        # Create FICO range width (measure of uncertainty)
        if self.fico_low_col in X_copy.columns and self.fico_high_col in X_copy.columns:
            X_copy['fico_range_width'] = X_copy[self.fico_high_col] - X_copy[self.fico_low_col]
        else:
            X_copy['fico_range_width'] = 4  # Typical range width
        
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """Returns feature names after transformation."""
        if input_features is None and self._input_features is not None:
            input_features = self._input_features
        elif input_features is None:
            raise ValueError("input_features must be provided if fit hasn't been called.")
        
        input_features = np.asarray(input_features, dtype=object)
        
        # Add new features created by this transformer
        new_features = ['fico_avg', 'fico_risk_tier', 'fico_high_risk', 'fico_excellent', 'fico_range_width']
        
        output_features = list(input_features) + new_features
        
        return np.array(output_features, dtype=object)

# Custom transformer for enhanced credit utilization features
class CreditUtilizationEnhancer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._input_features = None
        
    def fit(self, X, y=None):
        self._input_features = list(X.columns)
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Enhanced revolving utilization features
        if 'revol_util' in X_copy.columns:
            # Convert percentage to decimal if needed
            X_copy['revol_util_decimal'] = X_copy['revol_util'] / 100
            
            # Create utilization risk tiers
            def get_util_risk_tier(util_rate):
                if pd.isna(util_rate):
                    return 'Unknown'
                elif util_rate >= 90:
                    return 'Very High'  # >90% utilization
                elif util_rate >= 70:
                    return 'High'       # 70-90% utilization
                elif util_rate >= 30:
                    return 'Moderate'   # 30-70% utilization
                elif util_rate >= 10:
                    return 'Low'        # 10-30% utilization
                else:
                    return 'Very Low'   # <10% utilization
            
            X_copy['util_risk_tier'] = X_copy['revol_util'].apply(get_util_risk_tier)
            
            # Binary indicators for high utilization
            X_copy['high_utilization'] = (X_copy['revol_util'] >= 70).astype(int)
            X_copy['very_high_utilization'] = (X_copy['revol_util'] >= 90).astype(int)
        
        # Enhanced debt-to-income features
        if 'dti' in X_copy.columns:
            # DTI risk tiers
            def get_dti_risk_tier(dti_ratio):
                if pd.isna(dti_ratio):
                    return 'Unknown'
                elif dti_ratio >= 40:
                    return 'Very High'  # >40% DTI
                elif dti_ratio >= 30:
                    return 'High'       # 30-40% DTI
                elif dti_ratio >= 20:
                    return 'Moderate'   # 20-30% DTI
                elif dti_ratio >= 10:
                    return 'Low'        # 10-20% DTI
                else:
                    return 'Very Low'   # <10% DTI
            
            X_copy['dti_risk_tier'] = X_copy['dti'].apply(get_dti_risk_tier)
            
            # Binary indicators for high DTI
            X_copy['high_dti'] = (X_copy['dti'] >= 30).astype(int)
            X_copy['very_high_dti'] = (X_copy['dti'] >= 40).astype(int)
        
        # Credit account diversity features
        if 'total_acc' in X_copy.columns and 'open_acc' in X_copy.columns:
            # Account utilization rate (open/total)
            X_copy['acc_utilization_rate'] = X_copy['open_acc'] / (X_copy['total_acc'] + 1e-8)
            
            # Closed account ratio
            X_copy['closed_acc_ratio'] = (X_copy['total_acc'] - X_copy['open_acc']) / (X_copy['total_acc'] + 1e-8)
        
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """Returns feature names after transformation."""
        if input_features is None and self._input_features is not None:
            input_features = self._input_features
        elif input_features is None:
            raise ValueError("input_features must be provided if fit hasn't been called.")
        
        input_features = np.asarray(input_features, dtype=object)
        
        # Add new features created by this transformer
        new_features = [
            'revol_util_decimal', 'util_risk_tier', 'high_utilization', 'very_high_utilization',
            'dti_risk_tier', 'high_dti', 'very_high_dti',
            'acc_utilization_rate', 'closed_acc_ratio'
        ]
        
        output_features = list(input_features) + new_features
        
        return np.array(output_features, dtype=object)