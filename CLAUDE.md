# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Investor Portfolio Management Model** - Predict LendingClub loan defaults to maximize investment returns.

**Use Case:** Build an independent risk model to help investors select loans for their LC portfolio, without relying on LC's proprietary grade.

**Philosophy:** Focus on fundamentals, progressive model complexity, and honest metrics. This is a portfolio piece demonstrating data science skills, not production engineering.

**Key Decision:** Keep interest rate and installment (investor features) while removing grade/sub_grade (circular reasoning).

## Project Structure

### Development Notebooks (dev/ directory in Python format)
- **Purpose:** Work-in-progress notebooks in Python percent format (better git diffs)
- **Files:**
  - `dev/1_analytics.py`: EDA and data understanding
  - `dev/2_Modelling.py`: Feature engineering, progressive model training (5 models)
  - `dev/3_Deployment.py`: Simple prediction examples

### Finalized Notebooks (Root directory in Jupyter format)
- **Purpose:** Completed notebooks with outputs for portfolio viewing
- **Files:**
  - `1_analytics.ipynb`: EDA results (converted from dev/1_analytics.py)
  - `2_Modelling.ipynb`: Model results (converted from dev/2_Modelling.py)
  - `3_Deployment.ipynb`: Deployment demo (converted from dev/3_Deployment.py)

### Documentation (dev/docs/ directory)
- `REFACTORING_PLAN.md`: High-level refactoring strategy
- `00_Refactoring_Overview.md`: Why refactor (investor perspective, not circular reasoning)
- `01_Use_Case_And_Features.md`: Why keep int_rate, why remove grade
- `02_Feature_Engineering_Simplification.md`: Transformer selection rationale
- `03_Workflow_Guide.md`: dev/ Python format workflow
- `04_Model_Selection_Philosophy.md`: Progressive modeling approach (CS229)

### Source Code (`src/`)
- `predictor.py`: Load pipeline and make predictions
- `transformers.py`: Custom transformers (keep 3, remove 3)
  - **Keep:** EmpLengthConverter, CreditHistoryCalculator, InterestRateRiskTierTransformer
  - **Remove:** CountBinarizer, FICORiskTierTransformer, CreditUtilizationEnhancer

### Data (`data/`)
- `accepted_2007_to_2018Q4.csv`: Full dataset (887K loans)
- `sample_applications.csv`: Sample for testing predictions

## Development Workflow

### Directory Structure
- **dev/** - Development notebooks in Python percent format (.py files)
  - Python format for better git diffs and version control
  - Work-in-progress versions
  - **docs/** - Documentation and design decisions

- **Root Notebooks** - Finalized Jupyter notebooks (.ipynb)
  - Converted from dev/ Python files
  - Include executed outputs for portfolio viewing
  - Professional presentation versions

### Workflow: Edit → Test → Convert → Commit
1. **Edit:** Make changes in `dev/*.py` files
2. **Test:** Run `python dev/filename.py` or execute cells in IDE
3. **Convert:** `jupyter nbconvert --to notebook dev/filename.py --output filename.ipynb`
4. **Execute:** `jupyter nbconvert --execute --inplace filename.ipynb` (to generate outputs)
5. **Commit:** `git add dev/*.py && git add *.ipynb` (both versions)

**Why Python format?**
- Readable git diffs (no JSON metadata noise)
- Better IDE support (syntax highlighting, linting)
- Standard Python tooling works
- Easier code review and version control

## Environment Setup

This project uses a dedicated conda environment named `lending-club-ds`.

### Quick Setup
```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate lending-club-ds
```

### Manual Setup
```bash
# Create environment
conda create -n lending-club-ds python=3.11 -y

# Install core packages (14 essential)
conda install -n lending-club-ds \
  numpy>=2.0.1 \
  pandas>=2.2.3 \
  scikit-learn>=1.6.1 \
  scipy>=1.15.3 \
  matplotlib>=3.10.0 \
  seaborn>=0.13.2 \
  xgboost>=2.1.2 \
  lightgbm>=4.6.0 \
  jupyter>=1.1.1 \
  notebook>=7.4.4 \
  ipython>=9.1.0 \
  ipykernel \
  joblib>=1.4.2 \
  python-dateutil>=2.9.0 \
  -y

# Activate environment
conda activate lending-club-ds
```

### Key Dependencies (14 Essential Packages)
- **Core**: numpy, pandas, scikit-learn, scipy
- **ML**: xgboost, lightgbm (both for model comparison)
- **Visualization**: matplotlib, seaborn
- **Development**: jupyter, notebook, ipython, ipykernel
- **Utilities**: joblib, python-dateutil

## Key Features & Design Decisions

### Feature Selection: Investor Perspective

**INCLUDE (Legitimate Investor Features):**
- `int_rate` and `installment` - Published features when loan is listed
- Applicant data - FICO, DTI, employment, credit history, etc.

**EXCLUDE (Circular Reasoning):**
- `grade` and `sub_grade` - LC's proprietary risk scores (using their prediction to make our own is circular)

See `dev/docs/01_Use_Case_And_Features.md` for detailed explanation.

### Custom Transformers (Keep 3, Remove 3)

**KEEP (Domain-specific, unavailable in sklearn):**
1. EmpLengthConverter - String to numeric ("10+ years" → 10)
2. CreditHistoryCalculator - Date arithmetic (credit history in years)
3. InterestRateRiskTierTransformer - Bins and flags for interest rate (legitimate feature)

**REMOVE (Over-engineered, can use sklearn):**
1. CountBinarizer - Use sklearn.preprocessing.Binarizer
2. FICORiskTierTransformer - Over-complicated (5 features from 1 input)
3. CreditUtilizationEnhancer - Too many derived features (9 features from 2 inputs)

See `dev/docs/02_Feature_Engineering_Simplification.md` for rationale.

### Model Progression (CS229 Approach)

Build 5 models of increasing complexity:
1. **Logistic Regression** - Baseline (AUC ~0.53)
2. **Decision Tree** - Non-linearity (AUC ~0.55)
3. **Random Forest** - Ensemble stability (AUC ~0.57)
4. **XGBoost** - Boosting (AUC ~0.61)
5. **LightGBM** - Alternative boosting (AUC ~0.61)

Why this approach: Start simple, add complexity only when justified by empirical results.

See `dev/docs/04_Model_Selection_Philosophy.md` for full strategy.

### Making Predictions

```python
from src.predictor import CreditPredictor

# Initialize predictor
predictor = CreditPredictor(artifact_path='.')

# Make predictions on raw input data
probabilities = predictor.predict_proba(input_data)
predictions = predictor.predict(input_data, threshold=0.5)
```

## Common Tasks

### Development
```bash
# Activate environment
conda activate lending-club-ds

# Edit code in IDE
code dev/2_Modelling.py  # or your preferred editor

# Test code
python dev/2_Modelling.py
```

### Conversion & Execution
```bash
# Convert Python percent format to Jupyter notebook
jupyter nbconvert --to notebook dev/2_Modelling.py --output 2_Modelling.ipynb

# Execute notebook to generate outputs
jupyter nbconvert --execute --inplace 2_Modelling.ipynb
```

### Git Workflow
```bash
# Stage both versions
git add dev/*.py          # Python development versions
git add *.ipynb          # Jupyter presentation versions
git add dev/docs/*.md    # Documentation

# Commit
git commit -m "refactor: Message explaining changes"
```

### View Results
```bash
# View finalized notebooks (with outputs)
jupyter lab 2_Modelling.ipynb
```

## Related Documentation

See `dev/docs/` for detailed guides:
- `00_Refactoring_Overview.md` - Project philosophy and changes
- `01_Use_Case_And_Features.md` - Why include/exclude features
- `02_Feature_Engineering_Simplification.md` - Transformer selection
- `03_Workflow_Guide.md` - dev/ directory workflow
- `04_Model_Selection_Philosophy.md` - Progressive modeling approach
- `REFACTORING_PLAN.md` - Full refactoring timeline

## Expected Performance

**Investor Use Case Model:**
- **AUC:** 0.60-0.625 (realistic for independent assessment)
- **Features:** ~60 (no excessive binning/derivation)
- **Models:** 5 (progressive complexity)
- **Training Time:** <2 minutes (all 5 models)

This is respectable performance for a portfolio piece demonstrating:
✅ Proper feature selection (no circular reasoning)
✅ Understanding of data leakage and timing
✅ Clear business context (investor perspective)
✅ Progressive model complexity (CS229 philosophy)