# LendingClub Loan Default Prediction

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![scikit-learn 1.6.1](https://img.shields.io/badge/scikit--learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost 2.1.2](https://img.shields.io/badge/xgboost-2.1.2-red.svg)](https://xgboost.readthedocs.io/)

A probability of default (PD) prediction model trained on LendingClub's historical loan data. Achieved 0.69 AUC on out-of-time test data (2016-2017 originations) using only features available at loan origination. The key finding: simple Logistic Regression outperformed more complex models like XGBoost, prioritizing interpretability and deployment simplicity.

## What This Does

This project predicts which loans will default using only information available at loan origination—no hindsight features or proprietary risk scores.

**Data:** 2.26M loan applications, 887K approved loans from LendingClub (2007-2018)

**Approach:**
- Excludes post-origination data (payment history, recoveries, settlement outcomes) — these are outcome signals, not predictive features
- Excludes LendingClub's proprietary `grade`/`sub_grade` scores — would introduce circular reasoning by using their risk assessment as inputs
- Uses time-based validation (train ≤2015, test ≥2016) to account for concept drift rather than random splitting, which would overestimate generalization

**Result:** A trained sklearn pipeline that takes raw loan applications and returns default probability for each one.

## The Results

Test AUC: **0.6866**
Model: **Logistic Regression**
Decision Threshold: **0.3794** (F2-optimized to favor recall over precision)

At this threshold:
- Catches ~78% of defaults (high recall)
- But flags ~17% of good loans as risky (low precision)
- Tradeoff: Better to reject a good loan than approve a bad one from an investor's perspective

**Model Comparison:**
Tried multiple models on the same data:

| Model | Test AUC | Notes |
|-------|----------|-------|
| Logistic Regression | 0.6866 | Simple, interpretable, fast |
| Decision Tree | 0.6412 | Underperformed, prone to overfitting |
| Random Forest | 0.6853 | Nearly identical to LR, added complexity |
| XGBoost | 0.6906 | Only +0.004 improvement, not worth the complexity |

Logistic Regression won. The extra complexity from tree-based models didn't justify the marginal AUC gain.

## Key Design Decisions

**Time-based validation for robustness.** Random train/test splits would have overestimated performance by ~0.04 AUC in this domain. Lending markets exhibit concept drift—risk profiles in 2007-2012 differ materially from 2016-2017. Temporal validation ensures the model captures genuine predictive patterns rather than market artifacts.

**Deliberate exclusion of proprietary grades.** LendingClub's `grade` and `sub_grade` are their own risk assessments. Using them would produce circular logic (predicting their prediction). The model instead builds from raw applicant features available at loan decision time.

**Interest rate as a core feature.** Interest rate is published before loan origination and represents investor-accessible information. However, it implicitly encodes LendingClub's risk assessment. Removing it degraded AUC by 0.02—a material loss for minimal ethical gain, so it's retained.

**F2-score optimization for threshold selection.** The F2 metric (weighted recall/precision) favors recall over precision, reducing false negatives. At the chosen threshold (0.3794), the model captures ~78% of defaults at the cost of higher false positive rates (~83%)—an appropriate tradeoff for risk-conscious lending decisions.

## How It Works

**The Pipeline:**

Three notebooks build on each other:

1. **`1_EDA.ipynb`** — Exploratory Data Analysis
   - Define the target (Charged Off = 1, Fully Paid = 0)
   - Audit features for leakage (remove anything post-origination)
   - Handle missing values (drop ultra-sparse, median impute numeric)
   - Parse messy fields ("10+ years" → 10, "36 months" → 36)
   - Check for concept drift between train and test periods

2. **`2_Modelling.ipynb`** — Model Training & Selection
   - Load cleaned data
   - Time-based train/test split (≤2015 train, ≥2016 test)
   - Try 4 models (LR, DT, RF, XGBoost)
   - Optimize threshold using F2-score
   - Save pipeline + threshold + metadata as joblib artifact

3. **`3_Deployment.ipynb`** — Inference Demo
   - Load saved pipeline and threshold
   - Generate default probability predictions on new loans
   - Show deployment wrapper class for production use

**Technical Details:**

The pipeline uses custom sklearn transformers to handle data quality and feature engineering:

- `RawLendingClubCleaner` — Removes post-origination leakage, parses string fields, enforces an allowlist of valid features
- `ColumnAligner` — Ensures required columns exist during inference by adding missing features as NaN (handles incomplete new data robustly)
- `Capper` — Caps numeric features at the 99th percentile learned during training (reduces outlier leverage)
- `MissingnessDropper` — Removes ultra-sparse columns (>70% missing) learned during training

The entire pipeline is built with scikit-learn, enabling clean serialization with joblib and straightforward production deployment without custom preprocessing logic.

**Example Usage:**

```python
from src.predictor import CreditRiskPredictor
import pandas as pd

# Load the trained model
predictor = CreditRiskPredictor.from_artifact('models/credit_risk_pipeline_investor.joblib')

# Load new loan applications (same schema as training data)
new_loans = pd.read_csv('new_applications.csv')

# Get default probabilities (0-1)
probabilities = predictor.predict_proba(new_loans)

# Get binary decisions (0=approve, 1=reject)
decisions = predictor.predict(new_loans)

# Or get everything together
results = predictor.predict_with_details(new_loans)
print(results)
```

## Project Structure

```
├── 1_EDA.ipynb                      # Data exploration & cleaning
├── 2_Modelling.ipynb                # Model training & evaluation
├── 3_Deployment.ipynb               # Inference demo
├── src/
│   ├── transformers.py              # Custom sklearn transformers
│   └── predictor.py                 # CreditRiskPredictor class
├── models/
│   └── credit_risk_pipeline_investor.joblib  # Saved pipeline
├── data/
│   ├── sample_applications.csv      # Sample data for testing
│   └── [accepted_2007_to_2018Q4.csv not in repo — 1.6GB]
├── environment.yml                  # Conda environment
└── README.md                        # This file
```

## Running It

### Quick Start

Notebooks are saved with outputs — you can view results without running anything.

**If you want to explore the code:**

```bash
# Create conda environment
conda env create -f environment.yml
conda activate lending-club-ds

# Start Jupyter
jupyter notebook

# Open 1_EDA.ipynb, 2_Modelling.ipynb, or 3_Deployment.ipynb
```

### Re-training the Model

You'll need the full LendingClub dataset (2.26M applications, ~1.6GB).

1. Download from [LendingClub](https://www.lendingclub.com/info/download-data.action)
2. Save as `data/accepted_2007_to_2018Q4.csv`
3. Run `1_EDA.ipynb` to clean the data
4. Run `2_Modelling.ipynb` to train and save the model

Or just review the saved notebook outputs.

### Making Predictions

```python
from src.predictor import CreditRiskPredictor
predictor = CreditRiskPredictor.from_artifact('models/credit_risk_pipeline_investor.joblib')
results = predictor.predict_with_details(new_loans)
```

## Built With

- **Python 3.11** — Language
- **scikit-learn 1.6.1** — ML pipeline, preprocessing, metrics
- **XGBoost 2.1.2** — Gradient boosting (for comparison)
- **pandas 2.2.3** — Data wrangling
- **NumPy 2.0.1** — Numeric operations
- **Jupyter 1.1.1** — Development environment
- **conda** — Environment management

## Known Limitations

**60-month loan performance gap:**

Model achieves 0.62 AUC on 60-month loans compared to 0.69 on 36-month loans. Root cause: the dataset contains far fewer 60-month originations, limiting training signal. Recommended solutions include building term-specific models or flagging reduced predictive power for longer-duration loans in production systems.

**Dependency on interest rate feature:**

Interest rate is the single most predictive feature, reflecting LendingClub's proprietary pricing models. While interest rate is published before origination (legitimate investor information), the model partially leverages their risk assessment rather than providing fully independent predictions. Removing this feature degrades performance by 0.02 AUC—a material cost for limited benefit.

**High false positive rate at chosen threshold:**

The F2-optimized decision boundary (0.3794) rejects 83% of approved loans to capture 78% of defaults—reflecting the conservative risk posture appropriate for capital-constrained investors. More aggressive thresholds would improve loan approval rates but increase default risk exposure.

## Future Improvements

- **Term-specific modeling:** Develop separate pipelines for 36-month vs. 60-month loans to improve performance on underrepresented segments
- **Probability calibration:** Implement validation that predicted probabilities align with observed default rates in production data
- **Concept drift monitoring:** Establish automated detection for distributional shifts in new loan populations relative to training data
- **Business-driven threshold optimization:** Calibrate decision thresholds based on business metrics (e.g., expected portfolio loss) rather than statistical measures alone

## Contact

Add your contact information here (email, LinkedIn, GitHub, portfolio site). 
