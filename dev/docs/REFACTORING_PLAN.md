# Lending Club Loan Project - Refactoring Plan

**Status:** Approved & Ready for Implementation
**Date Created:** 2025-11-27
**Goal:** Transform project from over-engineered implementation into clean, fundamental-focused portfolio piece

---

## Executive Summary

Transform the Lending Club credit risk project into a clean, fundamental-focused portfolio piece demonstrating core data science skills.

**Primary Goals:**
1. **Fix Critical Data Leakage** - Remove int_rate, installment (assigned by LC based on grade)
2. **Simplify Codebase** - Reduce custom transformers from 6 to 2, eliminate styling bloat
3. **Remove Library Bloat** - Keep only essential dependencies (~14 packages)
4. **Document Workflow** - Clarify dev/ directory purpose and refactoring decisions
5. **Achieve Realistic Metrics** - Expected AUC: 0.625 → 0.55-0.58 (honest, no leakage)
6. **Progressive Modeling** - Follow CS229 philosophy: start simple, add complexity only when justified

---

## Critical Issue: Data Leakage

### The Problem
Model currently uses `int_rate` and `installment` which are **assigned by Lending Club based on their proprietary grade**, creating circular reasoning:

**Lending Club's Workflow:**
```
Applicant submits data
    ↓
LC assigns grade (proprietary algorithm)
    ↓
LC determines int_rate based on grade
    ↓
Calculates installment from loan_amnt + int_rate
    ↓
Loan is issued
```

**Current Code Error (dev/2_Modelling.py, line 533):**
```python
# int_rate is NOT circular reasoning - it's an independent market variable
```

**Reality:** int_rate is NOT independent - it's determined by LC's proprietary pricing model based on grade.

### Features to Remove
| Feature | Status | Reason |
|---------|--------|--------|
| `grade` | ✅ Already excluded | LC's proprietary risk score |
| `sub_grade` | ✅ Already excluded | LC's fine-grained risk score |
| `int_rate` | ❌ Currently included, MUST REMOVE | Assigned by LC based on grade |
| `installment` | ❌ Currently included, MUST REMOVE | Calculated from loan_amnt + int_rate |

### Expected Impact
- **AUC Drop:** 0.625 → 0.55-0.58 (7-12% decrease)
- **Feature Reduction:** ~80 → ~50 features
- **Why Good:** Demonstrates honest modeling without circular reasoning

---

## Modeling Strategy: Progressive Complexity (CS229 Philosophy)

Following Stanford CS229 teaching methodology: **start simple, add complexity only when justified by performance gains**.

### Model Progression Framework

| # | Model | Purpose | Expected AUC | Narrative |
|---|-------|---------|--------------|-----------|
| 1 | Logistic Regression | Interpretable baseline, coefficient analysis | 0.52-0.54 | "Simple linear model as foundation" |
| 2 | Decision Tree | Capture non-linear relationships | 0.54-0.56 | "Single tree finds non-linear patterns but overfits" |
| 3 | Random Forest | Reduce overfitting through bagging | 0.56-0.57 | "Ensemble reduces variance, improves stability" |
| 4 | XGBoost | Systematic error reduction through boosting | 0.57-0.58 | "Boosting systematically corrects errors" |
| 5 | LightGBM | Compare alternative boosting implementation | 0.57-0.58 | "Alternative boosting for comparison" |

**Key Insight:** 5% AUC improvement costs 50x training time and interpretability loss. Trade-offs are documented.

---

## Implementation Plan

### Phase 1: Documentation Setup

#### 1.1 Update CLAUDE.md
- Add Development Workflow section explaining dev/ Python format
- Update Project Overview to emphasize portfolio focus
- Simplify Dependencies section (14 essential packages)

#### 1.2 Create dev/docs/ Files
1. **00_Refactoring_Overview.md** - High-level refactoring philosophy
2. **01_Data_Leakage_Analysis.md** - Detailed leakage analysis
3. **02_Feature_Engineering_Simplification.md** - Transformer reduction strategy
4. **03_Workflow_Guide.md** - dev/ directory Python format explanation
5. **04_Model_Selection_Philosophy.md** - Progressive modeling approach

### Phase 2: Source Code Refactoring

#### 2.1 src/transformers.py (367 → ~120 lines)
**KEEP:** EmpLengthConverter, CreditHistoryCalculator
**DELETE:** CountBinarizer, FICORiskTierTransformer, CreditUtilizationEnhancer, InterestRateRiskTierTransformer

#### 2.2 src/predictor.py
- Remove complex error handling
- Simplify interface
- Remove SageMaker comments
- Focus on demonstration clarity

### Phase 3: Modeling Notebook (dev/2_Modelling.py)

**CRITICAL DATA LEAKAGE FIXES:**

| Line(s) | Action | Details |
|---------|--------|---------|
| 528-529 | Add to cols_to_drop | `'int_rate'`, `'installment'` |
| 98-149 | DELETE entire class | InterestRateRiskTierTransformer |
| 82 | Update imports | Remove unused transformers |
| 750-752 | REMOVE from pipeline | int_rate derived features |
| 76-95 | DELETE imports | psutil, gc, tqdm, memory utilities |
| 156-286 | SIMPLIFY styling | 250+ lines → ~20 lines |
| Throughout | Add models | Logistic → Tree → Forest → XGB → LightGBM |

**Expected:** 2,132 → ~900 lines, ~50 features, 5 models trained

### Phase 4: Analytics Notebook (dev/1_analytics.py)

- Simplify styling (250+ lines → ~20 lines)
- Remove memory management imports
- Update correlation analysis (exclude leakage features)
- Add narrative about independent modeling

**Expected:** 2,335 → ~1,200 lines

### Phase 5: Deployment Notebook (dev/3_Deployment.py)

- Update sample data (remove leakage columns)
- Simplify predictor demonstration
- Remove production comments

**Expected:** 207 → ~120 lines

### Phase 6: Environment Simplification

**environment.yml changes:**
- REMOVE: tqdm, psutil, pytz (3 packages)
- KEEP: xgboost, lightgbm (both for comparison)
- RESULT: 17 → 14 packages

### Phase 7: Final Integration

- Convert dev/*.py → root *.ipynb
- Execute all notebooks
- Final validation
- Portfolio-ready state

---

## Feature Engineering Simplification

### Current (Over-Engineered)
- 6 custom transformers
- ~30 derived features
- 250+ lines styling code
- Excessive memory management

### Target (Fundamental-Focused)
- 2 custom transformers (EmpLengthConverter, CreditHistoryCalculator)
- ~50 base features (no excessive binning)
- ~20 lines styling code
- Clean, minimal design

### Feature Selection (Final ~50 Features)

**Tier 1 - Core Credit (raw values):**
- FICO: fico_range_low, fico_range_high (NO binning)
- DTI: dti (NO tiers)
- Credit history: credit_hist_years (from CreditHistoryCalculator)
- Delinquency: delinq_2yrs
- Inquiries: inq_last_6mths

**Tier 2 - Loan Characteristics:**
- loan_amnt, term, purpose, home_ownership, annual_inc, verification_status

**Tier 3 - Credit Accounts (raw values):**
- open_acc, total_acc, revol_bal, revol_util (NO tiers), pub_rec, pub_rec_bankruptcies

**REMOVED:**
- grade, sub_grade (circular reasoning)
- int_rate, installment (circular reasoning)
- All *_risk_tier features (over-engineering)
- All high_*/very_high_* binary indicators (unnecessary)

---

## Quantitative Targets

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Custom Transformers | 6 | 2 | -67% |
| Total Code Lines | ~4,900 | ~2,400 | -51% |
| Dependencies | 17 | 14 | -18% |
| Features | ~80 | ~50 | -38% |
| Styling Code | ~500 | ~60 | -88% |
| Models Trained | 1-2 | 5 | +150-400% |
| AUC | 0.625 | 0.55-0.58 | -7% to -12% |

---

## Validation Checklist

### Critical: No Data Leakage
```python
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
leakage_terms = ['grade', 'sub_grade', 'int_rate', 'installment']
found_leakage = [f for f in feature_names if any(term in f.lower() for term in leakage_terms)]
assert len(found_leakage) == 0, f"LEAKAGE DETECTED: {found_leakage}"
```

### Performance Expectations
- **Before:** AUC ~0.625 (inflated by leakage)
- **After:** AUC ~0.55-0.58 (realistic, honest)
- **Baseline:** AUC 0.50 (random guessing)
- **Improvement:** 10-16% better than random ✅

---

## Critical Files to Modify

### Priority 1: Data Leakage Fixes
1. **dev/2_Modelling.py**
   - Remove int_rate, installment from features
   - Delete InterestRateRiskTierTransformer
   - Update imports
   - Implement 5-model progression

2. **src/transformers.py**
   - Keep: EmpLengthConverter, CreditHistoryCalculator
   - Delete: 4 over-engineered transformers

### Priority 2: Simplification
3. **dev/1_analytics.py** - Simplify styling, remove memory code
4. **dev/3_Deployment.py** - Update sample data, remove production cruft
5. **environment.yml** - Remove 3 packages, keep both XGB and LightGBM

### Priority 3: Documentation
6. **CLAUDE.md** - Add dev/ workflow section
7. **dev/docs/** - Create 5 markdown files

---

## Expected Portfolio Value

### Demonstrations of Skill
✅ **Data Understanding** - Clean EDA with clear insights
✅ **Feature Engineering** - Thoughtful, minimal (2 custom transformers)
✅ **Model Selection** - Progressive approach showing maturity
✅ **Model Comparison** - Side-by-side metrics, justified trade-offs
✅ **Data Leakage Awareness** - Critical thinking about feature availability
✅ **Communication** - Clear documentation throughout
✅ **CS229 Philosophy** - Start simple, justify complexity

### Final Metrics
- **AUC:** 0.55-0.58 (realistic, honest, no leakage)
- **Features:** ~50 (focused, interpretable)
- **Models:** 5 (progression shows maturity)
- **Code:** ~2,400 lines (clean, focused)
- **Dependencies:** 14 (minimal, essential)

---

## Timeline & Execution

**Total:** 7 phases, ~4-5 hours of work

| Phase | Focus | Files | Time |
|-------|-------|-------|------|
| 1 | Documentation | CLAUDE.md, dev/docs/ | 30m |
| 2 | Source code | src/*.py | 20m |
| 3 | Modeling (CRITICAL) | dev/2_Modelling.py | 60m |
| 4 | Analytics | dev/1_analytics.py | 40m |
| 5 | Deployment | dev/3_Deployment.py | 15m |
| 6 | Environment | environment.yml | 15m |
| 7 | Integration | Conversion & testing | 30m |

---

## References & Context

**Based on Analysis Files:**
- `00_Refactoring-Plan.md` - Strategic refactoring direction
- `01_Data-Leakage-Audit.md` - Detailed leakage identification

**Related Documentation:**
- See individual files in dev/docs/ for detailed analysis:
  - 00_Refactoring_Overview.md
  - 01_Data_Leakage_Analysis.md
  - 02_Feature_Engineering_Simplification.md
  - 03_Workflow_Guide.md
  - 04_Model_Selection_Philosophy.md

**Kaggle Best Practices:**
- [Lending Club Data Analysis](https://www.kaggle.com/datasets/urstrulyvikas/lending-club-loan-data-analysis)
- [Complete Lending Club Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- [Hitchhiker's Guide to LC Data](https://www.kaggle.com/code/pragyanbo/a-hitchhiker-s-guide-to-lending-club-loan-data)

---

## Next Steps

1. ✅ Plan approved
2. ⬜ Phase 1: Create documentation files
3. ⬜ Phase 2: Refactor source code
4. ⬜ Phase 3: Fix modeling notebook (data leakage)
5. ⬜ Phase 4: Simplify analytics notebook
6. ⬜ Phase 5: Update deployment notebook
7. ⬜ Phase 6: Simplify environment
8. ⬜ Phase 7: Final integration

**Ready to begin Phase 1?**
