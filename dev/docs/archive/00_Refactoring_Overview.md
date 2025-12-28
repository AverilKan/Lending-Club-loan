# Refactoring Overview

**Date:** 2025-11-27
**Status:** In Progress
**Use Case:** Investor Portfolio Management & Independent Risk Assessment
**Goal:** Clean, fundamental-focused portfolio piece demonstrating independent credit risk modeling

---

## Use Case: Investor Perspective

**Scenario:** You're an investor on the LendingClub platform

**Timeline:**
1. Applicant submits loan request → LC originates and lists loan
2. **[You are here]** Loan appears on platform with all details (applicant data + grade + int_rate + installment)
3. You decide whether to fund this loan
4. Loan is issued, borrower repays or defaults
5. You analyze: did YOUR model predict correctly?

**Goal:** Build independent risk model to identify profitable loans, WITHOUT blindly trusting LC's grade

---

## Why Refactor?

### Current Problems

1. **Circular Reasoning (grade/sub_grade)**
   - `grade` and `sub_grade` are LC's proprietary **risk scores**
   - Using them = using LC's prediction to make our own prediction
   - ❌ Remove these

2. **Over-Engineering**
   - 6 custom transformers creating ~30 derived features
   - 500+ lines of styling configuration
   - Memory management utilities for manageable dataset
   - Excessive complexity obscures fundamentals

3. **Missing Narrative**
   - Jumps straight to complex ensemble models
   - No justification for complexity choices
   - Doesn't demonstrate understanding of trade-offs

---

## Key Feature Decision: Why KEEP int_rate and installment?

**int_rate and installment are LEGITIMATE investor features because:**

1. **Availability:** Published for every loan on the platform (you see them as investor)
2. **Information Content:** Reflect LC's pricing strategy + market conditions, not just risk
3. **Business Value:** Help investors understand loan economics
4. **Independent Signal:** May contain additional signal beyond just grade

**Example:** Two loans might have same grade (same risk) but different rates (market conditions, borrower negotiation)

**Grade vs int_rate:**
- `grade` = "This is risky" (circular)
- `int_rate` = "This costs 15%" (operational fact)

---

## Refactoring Goals

### 1. Remove Circular Reasoning (grade/sub_grade ONLY)

**Remove:** grade, sub_grade (LC's risk scores - circular)
**Keep:** int_rate, installment (investor features - legitimate)
**Impact:** AUC maintains ~0.60-0.625 (realistic for independent investor assessment)

### 2. Simplify Feature Engineering

**Current:** 6 custom transformers → ~30 derived features
**Target:** 3 custom transformers → ~60 base features
**Keep:** EmpLengthConverter, CreditHistoryCalculator, InterestRateRiskTierTransformer
**Remove:** CountBinarizer, FICORiskTierTransformer, CreditUtilizationEnhancer

### 3. Remove Code Bloat

- **Styling:** 500+ lines → ~60 lines (use seaborn defaults)
- **Memory Management:** Delete psutil, tqdm, gc utilities
- **Dependencies:** 17 → 14 packages (remove unnecessary tools)
- **Total Code:** ~4,900 → ~2,400 lines (-51%)

### 4. Implement Progressive Modeling

**Follow CS229 Philosophy:** Start simple, justify complexity

**Model Progression:**
1. Logistic Regression (baseline, interpretable) → AUC ~0.52-0.54
2. Decision Tree (non-linearity) → AUC ~0.54-0.56
3. Random Forest (ensemble stability) → AUC ~0.56-0.57
4. XGBoost (boosting) → AUC ~0.57-0.58
5. LightGBM (comparison) → AUC ~0.57-0.58

**Key Insight:** 5% AUC gain costs 50x training time + interpretability loss

### 5. Clear Documentation

**Create:**
- Workflow guide (dev/ directory explanation)
- Data leakage analysis (why features removed)
- Feature engineering rationale (why only 2 transformers)
- Model selection philosophy (progressive complexity)

---

## Expected Outcomes

### Quantitative Changes

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Lines of Code | ~4,900 | ~2,400 | -51% |
| Custom Transformers | 6 | 2 | -67% |
| Dependencies | 17 | 14 | -18% |
| Features | ~80 | ~50 | -38% |
| Styling Code | ~500 | ~60 | -88% |
| Models Trained | 1-2 | 5 | +150-400% |

### Qualitative Improvements

✅ **Honest Metrics** - AUC 0.55-0.58 (realistic, no leakage)
✅ **Clear Narrative** - Progressive modeling shows maturity
✅ **Clean Code** - Focused, minimal, readable
✅ **Strong Foundation** - Demonstrates core DS fundamentals
✅ **Professional** - Well-documented, reproducible

---

## Success Criteria

### Must Have
- [x] Plan created and approved
- [ ] No data leakage (grade, sub_grade, int_rate, installment removed)
- [ ] 2 custom transformers only
- [ ] 5 models trained (progressive approach)
- [ ] AUC 0.55-0.58 (realistic range)
- [ ] ~50 features (focused, interpretable)
- [ ] Complete documentation (5 files)

### Nice to Have
- [ ] Model comparison visualizations
- [ ] Feature importance analysis across models
- [ ] Performance vs complexity trade-off chart

---

## Implementation Phases

1. **Documentation** (30 min) - Create supporting docs, update CLAUDE.md
2. **Source Code** (20 min) - Simplify transformers.py, predictor.py
3. **Modeling** (60 min) - Fix leakage, implement progressive models
4. **Analytics** (40 min) - Simplify styling, update analysis
5. **Deployment** (15 min) - Update sample data, simplify demo
6. **Environment** (15 min) - Reduce dependencies
7. **Integration** (30 min) - Convert to notebooks, execute, validate

**Total:** ~3.5-4 hours

---

## Key Decisions

### Why Only 2 Custom Transformers?

**EmpLengthConverter:** Domain-specific string parsing ("10+ years" → 10)
**CreditHistoryCalculator:** Date arithmetic for credit history years

**Everything else:** Use sklearn (SimpleImputer, StandardScaler, OneHotEncoder, Binarizer)

**Rationale:** Custom code only when sklearn truly insufficient

### Why Keep Both XGBoost and LightGBM?

- Demonstrates model comparison skills
- Shows understanding of different implementations
- Portfolio depth (not just "I used one library")
- Minimal cost (~15MB, similar performance)

### Why Accept Lower AUC?

**Integrity over metrics.** Removing leakage is the right thing to do:
- Enables true pre-origination modeling
- Shows critical thinking about features
- Demonstrates honest data science practices
- AUC 0.55-0.58 is respectable for credit risk without proprietary scores

---

## Related Documentation

- **REFACTORING_PLAN.md** - Comprehensive implementation plan
- **01_Data_Leakage_Analysis.md** - Detailed leakage explanation
- **02_Feature_Engineering_Simplification.md** - Transformer reduction strategy
- **03_Workflow_Guide.md** - dev/ directory Python format explanation
- **04_Model_Selection_Philosophy.md** - Progressive modeling approach

---

## Next Steps

1. ✅ Create refactoring overview (this file)
2. ⬜ Complete remaining 4 documentation files
3. ⬜ Update CLAUDE.md
4. ⬜ Begin source code refactoring

**Current Phase:** Documentation (1 of 5 files complete)
