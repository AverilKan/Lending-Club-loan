# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LendingClub Loan Default Prediction** - Predict loan defaults from investor perspective.

**Use Case:** Build an independent risk model to help investors select loans, without relying on LC's proprietary grade.

**Target Audience:** Junior data scientist portfolio demonstrating solid fundamentals.

**Philosophy:** Simplicity, correctness, and data-driven decisions over sophistication.

**Key Decision:** Keep interest rate and installment (investor features) while removing grade/sub_grade (circular reasoning).

---

## Project Structure

```
1_EDA.py                 # Exploratory Data Analysis (Python percent format)
2_Modeling.py            # Model training and evaluation (Python percent format)
README.md                # Project overview and findings
environment.yml          # Conda environment
pipeline.joblib          # Final saved model
data/
  ├── accepted_2007_to_2018Q4.csv     # Full dataset (887K loans)
  └── sample_applications.csv          # Sample for testing
dev/
  ├── archive/           # Previous complex implementation (reference)
  └── docs/
      └── notebook1_guide.md  # Detailed EDA checklist and structure
```

---

## Workflow

### 1. Exploratory Data Analysis (1_EDA.py)

Follow the structured approach in `dev/docs/notebook1_guide.md`:

- **Define target:** Binary classification (Charged Off vs Fully Paid), drop Current status
- **Leakage audit:** Remove post-origination features (payments, recoveries, settlement, hardship fields)
- **Missing values:** Simple rules (drop >50% missing, median impute numeric, "missing" impute categorical)
- **Data type parsing:** Convert dates, strings ("10+ years" → numeric), ensure numeric columns are clean
- **Univariate EDA:** Understand distributions for core features (FICO, DTI, annual_inc, loan_amnt, int_rate)
- **Bivariate EDA:** Plot default rate by feature bins to inform modeling approach
- **Time drift check:** Validate that time-based split (≤2015 train, ≥2016 test) is justified

**Output:** Cleaned dataset, validation of approach, EDA-backed hypotheses for modeling.

### 2. Model Training & Evaluation (2_Modeling.py)

Data-driven approach based on EDA findings:

- **Load cleaned data** from 1_EDA.py output
- **Train/test split:** Time-based (train ≤2015, test ≥2016) to address concept drift
- **Feature engineering:** sklearn only (StandardScaler, OneHotEncoder, simple numeric parsing)
  - No custom transformers
  - No excessive binning or derived features
- **Model selection:** TBD based on EDA findings
  - Start with Logistic Regression (baseline)
  - Add complexity only if EDA supports it (e.g., if bivariate EDA shows nonlinear patterns)
  - 2-3 models total (not 5)
- **Evaluation:** ROC-AUC, PR-AUC, confusion matrix, feature coefficients
- **Save pipeline:** joblib format for predictions

### 3. Working with Python Percent Format (.py files)

The project uses `.py` files with `%%` cell separators (Python percent format) for these advantages:

```python
# %% [markdown]
# # Section Title
# Markdown cells use # %% [markdown]

# %%
# Code cells use # %%
import pandas as pd

# %%
# Another code cell
data = pd.read_csv('data.csv')
```

**Advantages:**
- Run directly: `python 1_EDA.py`
- Better git diffs (no JSON metadata noise)
- Standard IDE support (syntax highlighting, linting)
- Can convert to .ipynb when needed: `jupyter nbconvert --to notebook 1_EDA.py`
- Same format as archived reference files

---

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

# Install core packages
conda install -n lending-club-ds \
  numpy>=2.0.1 \
  pandas>=2.2.3 \
  scikit-learn>=1.6.1 \
  scipy>=1.15.3 \
  matplotlib>=3.10.0 \
  seaborn>=0.13.2 \
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

### Key Dependencies

- **Core:** numpy, pandas, scikit-learn, scipy
- **Visualization:** matplotlib, seaborn
- **Development:** jupyter, notebook, ipython, ipykernel
- **Utilities:** joblib, python-dateutil

---

## Feature Selection Principle

### INCLUDE (Legitimate Investor Features)

- `int_rate` and `installment` - Published investor-facing features at loan origination
- Applicant data - FICO score, DTI, employment length, credit history, etc.

### EXCLUDE (Circular Reasoning)

- `grade` and `sub_grade` - LC's proprietary risk scores (using their prediction to make our own is circular)
- Post-origination features - Payment history, recoveries, settlement, hardship fields (outcome signals, not predictors)

---

## Key Guidelines

### 1. No Custom Transformers
- Use only sklearn.preprocessing built-ins (StandardScaler, OneHotEncoder, Binarizer, etc.)
- Parse messy strings (e.g., "10+ years", "36 months") with pandas before pipeline if needed
- Keep it simple - demonstrate understanding of data, not engineering complexity

### 2. Data-Driven Decisions
- Model selection is TBD and informed by EDA findings
- Don't plan 5 models upfront - let data guide you
- If Logistic Regression explains the data well, that's the right choice
- Only add complexity if EDA clearly supports it (nonlinear patterns, feature interactions)

### 3. Simple, Correct, Clear
- Demonstration of fundamentals matters more than sophistication
- Avoid over-engineering
- Honest about feature dependencies (int_rate encodes LC's risk assessment)
- Use correct terminology (ROC-AUC ≠ accuracy)

### 4. Time-Based Validation
- Train on loans issued ≤2015, test on loans issued ≥2016
- Addresses concept drift in lending data (risk profiles change over time)
- Standard practice for temporal prediction problems

---

## Expected Outcomes

- **Size:** ~2 Python notebooks with %% separators (~1,000-1,500 lines total)
- **Models:** 2-3 models (baseline + improvements based on EDA findings)
- **Pipeline:** Clean, reproducible sklearn-only pipeline
- **Runnable:** Can execute with `python 1_EDA.py` or use IDE with Jupyter extension
- **Demonstrates:** Leakage awareness, time-based validation, data-driven modeling, honest feature dependencies

---

## Common Tasks

### Development

```bash
# Activate environment
conda activate lending-club-ds

# Edit code
code 1_EDA.py  # or your preferred editor

# Run directly
python 1_EDA.py

# Run in IDE with Jupyter extension (Jupyter notebooks in VS Code, PyCharm, etc.)
# The IDE will recognize %% cell separators
```

### Convert to Jupyter Notebook (Optional)

```bash
# Convert Python percent format to Jupyter notebook
jupyter nbconvert --to notebook 1_EDA.py --output 1_EDA.ipynb

# Execute notebook to generate outputs
jupyter nbconvert --execute --inplace 1_EDA.ipynb
```

### Git Workflow

```bash
# Stage and commit
git add 1_EDA.py 2_Modeling.py
git add README.md environment.yml
git commit -m "feat: Add EDA and modeling notebooks with findings"
```

---

## References

- **EDA Guidance:** See `dev/docs/notebook1_guide.md` for detailed 8-section checklist (target definition, leakage audit, missing values, data types, univariate, bivariate, time drift, conclusions)
- **Data Dictionary:** Lending Club data dictionary CSV maps all columns to descriptions
- **Previous Implementation:** `dev/archive/` contains reference from over-engineered approach (useful for understanding what NOT to do)

---

## Summary

This is a **junior data scientist portfolio piece** demonstrating:

✅ Proper feature selection (no circular reasoning with LC's grades)
✅ Understanding of data leakage (no post-origination features)
✅ Time-based validation (concept drift awareness)
✅ Data-driven modeling (model selection from EDA, not assumptions)
✅ Clear communication (code is readable, decisions are documented)
