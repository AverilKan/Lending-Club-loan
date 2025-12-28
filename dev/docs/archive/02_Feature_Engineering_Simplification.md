# Feature Engineering Simplification

**Status:** Implementation Guide
**Purpose:** Justify removing 3 of 6 transformers while keeping 3
**Philosophy:** "Only custom code when sklearn truly insufficient"

---

## Current State: Over-Engineered (6 Transformers)

### Transformer Inventory

| # | Transformer | Purpose | Lines | Features Created | Status |
|---|-------------|---------|-------|------------------|--------|
| 1 | EmpLengthConverter | "10+ years" → 10 | ~40 | 1 | ✅ KEEP |
| 2 | CreditHistoryCalculator | Date diff → years | ~50 | 1 | ✅ KEEP |
| 3 | CountBinarizer | num → binary | ~50 | ? | ❌ REMOVE |
| 4 | InterestRateRiskTierTransformer | Bins & flags | ~100 | 5 | ✅ KEEP |
| 5 | FICORiskTierTransformer | Bins & flags | ~100 | 5 | ❌ REMOVE |
| 6 | CreditUtilizationEnhancer | Tiers & ratios | ~100 | 9 | ❌ REMOVE |

**Total Removed:** 3 transformers, ~50 lines, ~14 derived features

---

## Keep: 3 Custom Transformers (Domain-Specific Logic)

### 1. EmpLengthConverter ✅

**What it does:**
```
Input:  "10+ years", "5 years", "< 1 year", "1 year", "2 years", ...
Output: 10.0, 5.0, 0.5, 1.0, 2.0, ...
```

**Why custom code is needed:**
- String parsing with special rules ("10+ years" → 10, "< 1 year" → 0.5)
- Not available in sklearn
- Domain-specific (employment context)

**Why keep it:**
- Shows understanding of preprocessing
- Demonstrates custom transformer pattern
- Necessary step (no sklearn equivalent)

**Refactor:** Already clean, minimal lines

---

### 2. CreditHistoryCalculator ✅

**What it does:**
```
Input:  earliest_cr_line = "Dec-2008", issue_d = "Nov-2020"
Output: credit_hist_years = 12.08
```

**Why custom code is needed:**
- Date arithmetic (difference in years)
- Parses string dates
- Not a standard sklearn feature

**Why keep it:**
- Important financial indicator (years of credit experience)
- Requires date parsing logic
- Can't be done with simple imputation/scaling

**Refactor:** Already clean, minimal lines

---

### 3. InterestRateRiskTierTransformer ✅ (NOW JUSTIFIED)

**What it does:**
```
Creates 5 derived features from interest rate:
1. int_rate_tier (binned: low/medium/high/very_high)
2. high_int_rate (binary flag: >15%)
3. very_high_int_rate (binary flag: >22%)
4. int_rate_premium (centered around mean)
5. int_rate_risk_score (custom scoring)
```

**Why keep it (REVISED):**
- Interest rate is now a legitimate investor feature
- Non-linear relationship with default (higher rate → higher risk)
- Bins capture natural breaks in pricing strategy
- Demonstrates feature engineering skill on continuous variables

**Why simplify it:**
- 5 derived features may be excessive
- Could use sklearn.preprocessing.KBinsDiscretizer instead of custom bins
- Consider: Keep only tier + binary flags (2-3 features max)

**Refactor Strategy:**
```python
# OLD (custom, excessive):
int_rate_tier (custom bins)
high_int_rate (>15%)
very_high_int_rate (>22%)
int_rate_premium (custom centering)
int_rate_risk_score (custom formula)

# NEW (simplified):
int_rate_tier (using KBinsDiscretizer or custom bins)
high_int_rate (>15% binary flag)
```

**Justification:** Keep 2-3 features max, drop the others

---

## Remove: 3 Over-Engineered Transformers

### 1. CountBinarizer ❌

**What it does:**
```
Input:  open_acc = 5, total_acc = 12, etc.
Output: open_acc_binary = 1, total_acc_binary = 1, etc.
        (threshold at some value)
```

**Why remove:**
- Loses information (converts numeric → boolean)
- Linear relationship to default typically works better
- sklearn has better binning: KBinsDiscretizer, pd.cut()
- Adding complexity for questionable gain

**Replacement:**
```python
# Instead of custom CountBinarizer:
from sklearn.preprocessing import KBinsDiscretizer

binned = KBinsDiscretizer(n_bins=3, encode='onehot-dense').fit_transform(X)
```

---

### 2. FICORiskTierTransformer ❌

**What it does:**
```
Creates 5 derived features from FICO range:
1. fico_avg (average of low/high)
2. fico_range_width (high - low)
3. fico_tier (binned: poor/fair/good/very_good/excellent)
4. low_fico_flag (< 650?)
5. high_fico_flag (> 740?)
```

**Why remove:**
- Over-engineering: 5 features from 2 inputs
- Can't justify 5 derived features for portfolio
- FICO has well-known linear relationship with default
- sklearn preprocessing is cleaner

**Replacement:**
```python
# Instead of custom FICORiskTierTransformer:
from sklearn.preprocessing import StandardScaler

# Just use FICO directly (linear relationship):
scaler = StandardScaler()
fico_scaled = scaler.fit_transform(X[['fico_range_low', 'fico_range_high']])

# If binning needed:
from sklearn.preprocessing import KBinsDiscretizer
fico_binned = KBinsDiscretizer(n_bins=5).fit_transform(fico)
```

**Better Approach:** FICO is already well-studied; use it raw with scaling

---

### 3. CreditUtilizationEnhancer ❌

**What it does:**
```
Creates 9 derived features from revol_util and dti:
1. revol_util_tier (bins)
2. high_revol_util (>80%)
3. revol_util_premium
4. dti_tier (bins)
5. high_dti (>40%)
6. dti_premium
7. util_dti_ratio
8. util_dti_interaction
9. util_dti_combined_risk
```

**Why remove:**
- 9 features is excessive (feature explosion)
- Interactions and combined scores are ML's job, not feature engineering
- Creates redundant features
- Diminishes portfolio value (looks like over-engineering)
- sklearn has better tools for interactions (polynomial features)

**Replacement:**
```python
# Instead of custom CreditUtilizationEnhancer:
from sklearn.preprocessing import PolynomialFeatures

# Let the model learn interactions:
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X[['revol_util', 'dti']])

# Or just use raw features (Linear models capture linear relationships):
# revol_util (raw)
# dti (raw)
# Let the model determine their importance
```

---

## The Principle: When to Use Custom Transformers

### ✅ USE Custom Code When:
1. **String Parsing Required** - Employment length ("10+ years" → numeric)
2. **Date Arithmetic** - Credit history calculation
3. **Domain Knowledge** - Lending-specific transformations
4. **sklearn Has No Equivalent** - Special business logic
5. **Minimal Features** - 1-2 outputs per transformer

### ❌ DON'T Use Custom Code When:
1. **sklearn Has It** - Use KBinsDiscretizer, OneHotEncoder, etc.
2. **Information Loss** - Binarizing numeric continuous variables
3. **Feature Explosion** - Creating 5-9 features from 1-2 inputs
4. **ML Can Learn It** - Models handle interactions, polynomial features, etc.
5. **Adds Complexity** - Extra code means more maintenance, harder to debug

---

## Target Pipeline Architecture

### Current (Over-Engineered)
```
Input Data
    ↓
[6 Custom Transformers]
├─ EmpLengthConverter
├─ CreditHistoryCalculator
├─ CountBinarizer
├─ InterestRateRiskTierTransformer
├─ FICORiskTierTransformer
└─ CreditUtilizationEnhancer
    ↓
~30 Derived Features
    ↓
StandardScaler + OneHotEncoder
    ↓
Model Training
```

### Target (Simplified)
```
Input Data
    ↓
[3 Custom Transformers]
├─ EmpLengthConverter
├─ CreditHistoryCalculator
└─ InterestRateRiskTierTransformer (simplified)
    ↓
[Standard sklearn]
├─ SimpleImputer (missing values)
├─ StandardScaler (continuous variables)
├─ OneHotEncoder (categorical variables)
└─ KBinsDiscretizer (if binning needed)
    ↓
~60 Base Features (not derived)
    ↓
Model Training
```

---

## Implementation Checklist

### Phase 1: Keep These
- [ ] Keep EmpLengthConverter (line 8-44 in transformers.py)
- [ ] Keep CreditHistoryCalculator (line 47-129 in transformers.py)
- [ ] Keep InterestRateRiskTierTransformer (refactored, fewer features)

### Phase 2: Remove These
- [ ] Delete CountBinarizer class entirely
- [ ] Delete FICORiskTierTransformer class entirely
- [ ] Delete CreditUtilizationEnhancer class entirely

### Phase 3: Update Pipeline
- [ ] Remove deleted transformer imports from dev/2_Modelling.py
- [ ] Remove deleted transformer usages from pipeline
- [ ] Replace with sklearn equivalents where needed

### Phase 4: Testing
- [ ] Verify pipeline builds without removed transformers
- [ ] Check feature count (~60 instead of ~80)
- [ ] Validate AUC remains in expected range (0.60-0.625)
- [ ] Confirm code runs end-to-end

---

## Expected Outcomes

### Quantitative
- **Lines of Code:** ~367 → ~250 (remove 117 lines)
- **Custom Transformers:** 6 → 3 (-50%)
- **Features:** ~80 → ~60 (-25%)
- **Complexity:** High → Medium
- **Maintainability:** Harder → Easier

### Qualitative
✅ **Cleaner portfolio** - Shows judgment about when custom code is justified
✅ **Better signal** - Fewer derived features, clearer relationships
✅ **Professional** - Aligns with best practices (use sklearn when possible)
✅ **Maintainable** - Easier to debug and modify

---

## File References

**Files to Modify:**
- `src/transformers.py` - Delete 3 classes (lines ~130-367)
- `dev/2_Modelling.py` - Update imports and pipeline construction

**Status:** Ready for implementation in Phase 2

---

**Next:** See `03_Workflow_Guide.md` for dev directory best practices.
