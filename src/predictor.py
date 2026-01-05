"""
Production predictor for Lending Club default risk model.

Loads saved model artifacts and provides convenient methods for predictions.
All custom transformers are imported to enable joblib unpickling.
"""

import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Import custom transformers so joblib can find their definitions when unpickling
from src.transformers import (
    RawLendingClubCleaner,
    MissingnessDropper,
    Capper,
    ColumnAligner
)

# Default artifact filename (matches saving step in Notebook 2)
DEFAULT_ARTIFACT_FILENAME = "credit_risk_pipeline_investor.joblib"


class CreditRiskPredictor:
    """
    Production-ready wrapper for Lending Club default risk prediction.

    Loads a saved model artifact and provides convenient methods for:
    - Getting default probabilities
    - Making binary predictions
    - Detailed prediction results with decisions

    The artifact contains:
    - pipeline: Full sklearn pipeline (preprocessing + model), fitted on TRAIN+VAL
    - threshold: F2-optimized threshold selected on validation data
    - metadata: Training details, performance metrics, creation timestamp

    Usage:
        predictor = CreditRiskPredictor.from_artifact("path/to/artifact.joblib")
        proba = predictor.predict_proba(df_new_applications)
        decisions = predictor.predict(df_new_applications)
    """

    def __init__(self, pipeline, threshold, metadata):
        """
        Initialize predictor with pipeline and threshold.

        Parameters:
        -----------
        pipeline : sklearn.pipeline.Pipeline
            Fitted preprocessing + model pipeline
        threshold : float
            Decision threshold for binary classification (0-1)
        metadata : dict
            Model metadata (training date, performance, etc.)
        """
        self.pipeline = pipeline
        self.threshold = threshold
        self.metadata = metadata

    @classmethod
    def from_artifact(cls, artifact_path):
        """Load predictor from saved joblib artifact.

        Parameters:
        -----------
        artifact_path : str or Path
            Path to saved .joblib artifact file

        Returns:
        --------
        CreditRiskPredictor
            Initialized predictor ready for inference
        """
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")

        artifact = joblib.load(artifact_path)
        return cls(
            pipeline=artifact["pipeline"],
            threshold=artifact["threshold"],
            metadata=artifact["metadata"]
        )

    @classmethod
    def from_default_location(cls, search_paths=None):
        """Load predictor from default artifact location.

        Searches for the default artifact filename in multiple locations.

        Parameters:
        -----------
        search_paths : list[str], optional
            Paths to search for artifact. Defaults to ['.', './models', project_root/models]

        Returns:
        --------
        CreditRiskPredictor
            Initialized predictor ready for inference
        """
        if search_paths is None:
            search_paths = [
                Path.cwd() / "models",
                Path.cwd() / DEFAULT_ARTIFACT_FILENAME,
                Path.home() / ".lending-club" / DEFAULT_ARTIFACT_FILENAME,
            ]

        for search_path in search_paths:
            artifact_path = Path(search_path) / DEFAULT_ARTIFACT_FILENAME if Path(search_path).is_dir() else Path(search_path)
            if artifact_path.exists():
                return cls.from_artifact(artifact_path)

        raise FileNotFoundError(
            f"Could not find {DEFAULT_ARTIFACT_FILENAME} in any of: {search_paths}"
        )

    def predict_proba(self, X):
        """Return default probability predictions (0-1).

        Parameters:
        -----------
        X : pd.DataFrame or list[dict]
            Raw loan application data

        Returns:
        --------
        np.ndarray
            Default probabilities for each application
        """
        if isinstance(X, list):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a DataFrame or list of dicts")

        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, X):
        """Return binary default predictions (0=approve, 1=reject).

        Parameters:
        -----------
        X : pd.DataFrame or list[dict]
            Raw loan application data

        Returns:
        --------
        np.ndarray
            Binary predictions (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def predict_with_details(self, X):
        """Return DataFrame with probabilities, predictions, and decisions.

        Parameters:
        -----------
        X : pd.DataFrame or list[dict]
            Raw loan application data

        Returns:
        --------
        pd.DataFrame
            Results with columns:
            - default_probability: predicted probability (0-1)
            - predicted_default: binary prediction (0 or 1)
            - decision: human-readable decision (APPROVE or REJECT)
        """
        proba = self.predict_proba(X)
        pred = self.predict(X)

        return pd.DataFrame({
            "default_probability": proba,
            "predicted_default": pred,
            "decision": ["REJECT" if p == 1 else "APPROVE" for p in pred]
        })

    def get_metadata(self):
        """Return model metadata (training date, performance, etc.)."""
        return self.metadata

    def get_threshold(self):
        """Return decision threshold used for binary predictions."""
        return self.threshold

    def get_test_auc(self):
        """Return test set ROC-AUC."""
        return self.metadata.get("test_roc_auc", None)

# Example Usage (Optional - for testing within this script)
if __name__ == '__main__':
    """
    Example: Load model and make predictions on sample data.

    Run from project root:
        python -m src.predictor
    """
    print("="*80)
    print("CreditRiskPredictor Example Usage")
    print("="*80)

    try:
        # Load predictor from default location (models/credit_risk_pipeline_investor.joblib)
        print("\n1. Loading model from default location...")
        artifact_path = Path.cwd() / "models" / DEFAULT_ARTIFACT_FILENAME
        predictor = CreditRiskPredictor.from_artifact(artifact_path)

        print(f"✓ Model loaded successfully")
        print(f"  Threshold: {predictor.get_threshold():.4f}")
        print(f"  Test AUC: {predictor.get_test_auc():.4f}")

        # Create sample data (raw format - before preprocessing)
        print("\n2. Creating sample loan applications...")
        sample_data = [
            {
                'loan_amnt': 10000, 'term': '36 months', 'int_rate': 12.0,
                'emp_length': '10+ years',
                'home_ownership': 'MORTGAGE',
                'annual_inc': 75000, 'verification_status': 'Verified',
                'purpose': 'debt_consolidation', 'addr_state': 'CA', 'dti': 15.0,
                'delinq_2yrs': 0, 'fico_range_low': 700, 'fico_range_high': 704,
                'inq_last_6mths': 1, 'open_acc': 10, 'pub_rec': 0,
                'revol_bal': 15000, 'revol_util': 50.0, 'total_acc': 25,
            },
            {
                'loan_amnt': 25000, 'term': '60 months', 'int_rate': 14.5,
                'emp_length': '5 years',
                'home_ownership': 'RENT',
                'annual_inc': 50000, 'verification_status': 'Not Verified',
                'purpose': 'credit_card', 'addr_state': 'NY', 'dti': 25.0,
                'delinq_2yrs': 1, 'fico_range_low': 650, 'fico_range_high': 654,
                'inq_last_6mths': 2, 'open_acc': 8, 'pub_rec': 0,
                'revol_bal': 20000, 'revol_util': 75.0, 'total_acc': 15,
            },
        ]

        sample_df = pd.DataFrame(sample_data)
        print(f"✓ Created {len(sample_df)} sample applications")

        # Make predictions
        print("\n3. Making predictions...")
        predictions = predictor.predict_with_details(sample_df)

        print("\nPrediction Results:")
        print(predictions.to_string(index=False))

        print("\n" + "="*80)
        print("✅ Example completed successfully!")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"\nTo use this example:")
        print(f"1. Run Notebook 2 to generate the artifact")
        print(f"2. The artifact will be saved to: models/{DEFAULT_ARTIFACT_FILENAME}")
        print(f"3. Then run this script from the project root")
