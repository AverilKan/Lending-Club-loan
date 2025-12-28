# Development Workflow Guide

**Status:** Reference
**Purpose:** Explain dev/ directory structure and Python notebook format
**Audience:** Developers, maintainers, portfolio reviewers

---

## Why Python Format for Notebooks?

### The Problem with .ipynb Files

```json
{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 42,
      "metadata": {
        "collapsed": false,
        "tags": []
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Result: 42\n"
          ]
        }
      ],
      "source": [
        "print('Result:', 42)"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    ...nested metadata...
  }
}
```

**Issues:**
- Git diffs are unreadable (JSON noise)
- Code review is difficult
- Merge conflicts are painful
- Version control becomes messy

### The Solution: Python Percent Format

```python
# %%
# # Data Loading and Exploration
# This section loads the raw data and performs initial checks

import pandas as pd
import numpy as np

# %%
# Load data
data = pd.read_csv('data/loans.csv')
print(f"Loaded {len(data)} rows")

# %%
# Explore structure
print(data.head())
print(data.info())

# %%
# ## Summary Statistics
print(data.describe())
```

**Benefits:**
- ✅ Standard Python files (.py extension)
- ✅ Readable git diffs
- ✅ Works with version control
- ✅ IDE support (syntax highlighting, linting)
- ✅ Can use standard Python tools

---

## Directory Structure

```
Lending-Club-Loan/
├── dev/
│   ├── docs/                          ← You are here
│   │   ├── REFACTORING_PLAN.md
│   │   ├── 00_Refactoring_Overview.md
│   │   ├── 01_Use_Case_And_Features.md
│   │   ├── 02_Feature_Engineering_Simplification.md
│   │   ├── 03_Workflow_Guide.md
│   │   └── 04_Model_Selection_Philosophy.md
│   │
│   ├── 1_analytics.py                 ← Work-in-progress notebooks
│   ├── 2_Modelling.py
│   └── 3_Deployment.py
│
├── 1_analytics.ipynb                  ← Finalized notebooks (with outputs)
├── 2_Modelling.ipynb
├── 3_Deployment.ipynb
│
├── src/
│   ├── predictor.py
│   ├── transformers.py
│   └── __pycache__/
│
├── data/
│   ├── accepted_2007_to_2018Q4.csv
│   └── sample_applications.csv
│
├── CLAUDE.md                          ← Project guidance
├── README.md
├── environment.yml
├── requirements.txt
└── .gitignore
```

---

## Workflow: Editing → Testing → Conversion → Commit

### Step 1: Edit in dev/ (Python Format)

**File:** `dev/2_Modelling.py`

```python
# %%
# # Model Training and Evaluation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# %%
# ## Load and Prepare Data

data = pd.read_csv('data/loans.csv')
print(f"Loaded {len(data)} rows, {len(data.columns)} columns")

# %%
# ### Feature Selection

features = ['fico_range_low', 'dti', 'annual_inc', 'int_rate']
X = data[features]
y = data['loan_status']  # 0 = repaid, 1 = default

# %%
# ## Train Logistic Regression

model = LogisticRegression()
model.fit(X, y)
score = model.score(X, y)
print(f"Model accuracy: {score:.3f}")
```

**Tips:**
- Use `# %%` to mark cell boundaries
- Use `# ## Title` for markdown cells (h2 headers)
- Write docstrings for clarity
- Comments should explain WHY, not WHAT

### Step 2: Test Locally

**Run the Python file directly:**
```bash
python dev/2_Modelling.py
```

**OR use Jupyter with percent format:**
```bash
# Open in VS Code, Jupyter, or compatible IDE
# Run cells interactively
# Debug issues before committing
```

**OR use percent format converter:**
```bash
# Convert to notebook for testing
jupyter nbconvert --to notebook dev/2_Modelling.py --output test_2_Modelling.ipynb
```

### Step 3: Convert to Notebook Format

Once code is tested and working:

```bash
# Convert Python percent format to Jupyter notebook
jupyter nbconvert \
  --to notebook \
  dev/1_analytics.py \
  --output 1_analytics.ipynb

jupyter nbconvert \
  --to notebook \
  dev/2_Modelling.py \
  --output 2_Modelling.ipynb

jupyter nbconvert \
  --to notebook \
  dev/3_Deployment.py \
  --output 3_Deployment.ipynb
```

### Step 4: Execute Notebooks

```bash
# Activate environment
conda activate lending-club-ds

# Execute and save outputs
jupyter nbconvert \
  --to notebook \
  --inplace \
  --execute \
  --ExecutePreprocessor.timeout=600 \
  1_analytics.ipynb
```

### Step 5: Commit Both Versions

**Git workflow:**
```bash
# Stage both Python and notebook versions
git add dev/1_analytics.py
git add dev/2_Modelling.py
git add dev/3_Deployment.py
git add 1_analytics.ipynb
git add 2_Modelling.ipynb
git add 3_Deployment.ipynb

# Commit with meaningful message
git commit -m "refactor: Simplify feature engineering and styling

- Remove 3 over-engineered transformers
- Keep interest rate as legitimate investor feature
- Implement progressive model training
- Expected AUC: 0.60-0.625 (investor perspective)"
```

---

## Key Points About Python Percent Format

### Cell Markers

```python
# %%
# This is a cell boundary
# Everything below belongs to this cell

code_here = True

# %%
# This starts a new cell
# Previous cell ends here

different_code = True
```

### Markdown Cells

```python
# %%
# # Title (H1)
# This becomes markdown

# Regular comments (starting with #) become markdown
# Multiple lines work fine

# ## Subtitle (H2)
# Even more markdown

# %%
# Code resumes here
print("This is Python code")
```

### Mixed Content

```python
# %%
# # Data Loading
# Load and explore the raw dataset

import pandas as pd

# %%
# ### Read CSV
# This is a subheading with explanation

data = pd.read_csv('loans.csv')
print(f"Loaded {len(data)} rows")

# %%
# ### Initial Checks

print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\nMissing values:")
print(data.isnull().sum())
```

---

## Tools & IDE Support

### VS Code (Recommended)

**Extensions:**
- Python (Microsoft)
- Jupyter (Microsoft)
- Pylance

**Features:**
- Run `.py` files with Python extension
- View cells with `# %%` markers
- Interactive execution
- Syntax highlighting

**Shortcut:**
- `Ctrl+Shift+Enter` - Run cell and move to next

### Jupyter / JupyterLab

**Convert and open:**
```bash
jupyter nbconvert --to notebook dev/2_Modelling.py --output temp.ipynb
jupyter lab temp.ipynb
```

### PyCharm

**Built-in support** for percent format:
- Right-click cell → Run cell
- View cells in gutter

---

## Git Best Practices with dev/

### What to Commit

✅ **DO commit:**
- `dev/*.py` files (work in progress)
- Root `.ipynb` files (finalized with outputs)
- Documentation in `dev/docs/`
- Code changes, bug fixes, feature additions

### What to Ignore

❌ **DON'T commit:**
- `__pycache__/` directories
- `.ipynb_checkpoints/`
- `.pyc` files
- Temporary notebooks (e.g., `test_*.ipynb`)
- Large output files

**In .gitignore:**
```
__pycache__/
*.pyc
.ipynb_checkpoints/
.DS_Store
*.joblib
test_*.ipynb
temp_*.py
```

### Diff-Friendly Workflow

**Python files are diff-friendly:**
```bash
# git diff dev/2_Modelling.py - READABLE
# Shows exact code changes

# git diff 2_Modelling.ipynb - UNREADABLE (JSON)
# Avoid diffing notebooks directly
```

**Strategy:**
- Work in `dev/*.py` (generates readable diffs)
- Convert to `.ipynb` only when ready
- Keep `.ipynb` files for viewing (don't diff them)

---

## Conversion Commands Reference

### Single File
```bash
jupyter nbconvert --to notebook dev/2_Modelling.py --output 2_Modelling.ipynb
```

### All Files
```bash
for file in dev/*.py; do
  base=$(basename "$file" .py)
  jupyter nbconvert --to notebook "$file" --output "$base.ipynb"
done
```

### With Execution
```bash
jupyter nbconvert \
  --to notebook \
  --inplace \
  --execute \
  --ExecutePreprocessor.timeout=600 \
  2_Modelling.ipynb
```

### Preserve Metadata
```bash
# Conversion preserves markdown, code cells, and structure
# Outputs are generated during execution
```

---

## Troubleshooting

### Issue: Cells not recognized

**Problem:** `# %%` not creating cells

**Solution:**
- Make sure `# %%` is at start of line
- No leading spaces
- Blank line after comment optional

```python
# %%
✅ This works

  # %%
❌ This doesn't (leading space)
```

### Issue: Markdown not rendering

**Problem:** Comments not converted to markdown

**Solution:**
- Comments starting with `#` become markdown
- Use `# ` (with space) for readability
- Use `# ## heading` for subheadings

```python
# %%
# This is markdown  ✅
#This is code comment (no space) ❌
```

### Issue: Slow notebook execution

**Problem:** Conversion takes too long

**Solution:**
- Set timeout higher: `--ExecutePreprocessor.timeout=600`
- Run steps separately (not all at once)
- Check for infinite loops or heavy computation

---

## Summary

| Task | Command | Where |
|------|---------|-------|
| Write code | Edit `dev/*.py` | Local IDE |
| Test code | Run `python dev/*.py` | Terminal/IDE |
| Convert | `jupyter nbconvert --to notebook` | Terminal |
| Execute | `jupyter nbconvert --execute` | Terminal |
| Commit | `git add dev/*.py && git add *.ipynb` | Git |
| View | Open `*.ipynb` in browser/JupyterLab | Browser |

---

## Next Steps

1. Edit notebooks in `dev/*.py` format
2. Test locally
3. Convert to `*.ipynb` when ready
4. Execute to generate outputs
5. Commit both versions
6. Push to GitHub

See `04_Model_Selection_Philosophy.md` for model training approach.
