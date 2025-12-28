### Pre-flight checklist (before touching “analysis”)

* Lock the **prediction moment** (what info is allowed at prediction time) and define the **target** accordingly.
* Run a **leakage audit**: separate origination features from outcome/post-origination features.
* Confirm **time column** quality (parseable dates) and decide the **validation scheme** (time-based).
* Quantify **missingness + data type mess** and decide simple, defensible cleaning rules.
* Produce a short list of **EDA-backed modeling hypotheses** (not “try 10 models”).

---

## Notebook 1 (EDA): Outline + what I already see in your sample

First: your attached `sample_applications.csv` is **6 rows** with **151 columns**. That’s enough to validate schema and spot leakage, but it’s **not enough to infer distributions or relationships**. Also: your sample has only `Fully Paid` and `Current` statuses — **no defaults** — so any “default modeling” conclusions from this sample would be nonsense.

### 0) Setup and data dictionary wiring

**Goal:** load data + join descriptions so every column is explainable.

**What you do**

* Read the sample CSV.
* Read the data dictionary CSV (it requires `latin1` encoding).
* Create a `dict[col] -> description` lookup.

**Immediate sample findings (real, from your file)**

* `loan_status` exists (good).
* `issue_d` exists and looks like `"Dec-15"` → needs explicit parsing.
* Tons of post-origination variables exist (more below).

**Output**

* A quick schema table: columns, dtype, % missing, # unique, dictionary description.

✅ **Validation:** Data loads, dictionary maps to key fields (`loan_status`, `issue_d`, `int_rate`, etc.).
➡️ **Next:** define target + allowed features (prediction moment), otherwise EDA is pointless.

---

### 1) Define the modeling problem properly (or everything after is garbage)

**You must choose one:**

**Option A (cleanest for a junior portfolio):**
Predict **default outcome** using only loans with final status:

* Keep: `Fully Paid` vs `Charged Off` (and optionally `Default`)
* Drop: `Current` (unknown outcome)

**Option B (messier):**
Predict “bad loan” including late statuses. This is harder because “late” depends on observation window.

Given you want fundamentals and credibility: **choose Option A**.

**Output**

* A clear label definition: `y = 1` for charged-off/default, `y = 0` for fully paid.
* A table of loan_status counts before/after filtering.

✅ **Validation:** Target is binary and based on final outcomes only; class imbalance quantified.
➡️ **Next:** leakage audit (non-negotiable).

---

### 2) Leakage audit (this is where most LendingClub projects fail)

**Goal:** remove features that wouldn’t exist at origination.

From your sample alone, I can already flag obvious leakage columns like:

* `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`
* `last_pymnt_d`, `last_pymnt_amnt`, `out_prncp`
* hardship/settlement fields (`hardship_*`, `settlement_*`)
  These are literally outcomes or post-outcome signals.

**What you do**

* Create a leakage rule list (pattern-based + explicit list).
* Produce two feature sets:

  1. **Origination-only** (strict)
  2. **Investor-view** (includes `int_rate`, because investors see it — but you will later ablate it)

**Output**

* A “dropped for leakage” list, with reasons.
* A final candidate feature list.

✅ **Validation:** No post-origination fields remain in the modeling dataset.
➡️ **Next:** missingness + types cleanup.

---

### 3) Missing values: simple rules, no heroics

**Goal:** decide what to drop vs impute, based on missingness and meaning.

**What you do**

* Missingness by column (% missing).
* Categorize features:

  * drop if >X% missing **and** no strong justification
  * impute numeric with median
  * impute categorical with “missing”
* Add a “missingness indicator” only for a handful of meaningful columns (don’t spam 100 flags).

**Caution from your sample**
Many joint/hardship/settlement fields are 100% missing in the sample. That might not hold in full data, but it’s a red flag: these columns are either niche or irrelevant for your baseline.

**Output**

* Missingness plot (top 30 columns).
* Final imputation plan.

✅ **Validation:** After cleaning, no columns are “mostly empty” without justification; missingness strategy documented.
➡️ **Next:** data type normalization.

---

### 4) Data types and parsing (you’re not allowed to hand-wave this)

**Goal:** convert messy strings into usable numeric/time features.

Minimum transformations:

* `issue_d` → datetime
* `term` `"36 months"` → 36
* `emp_length` (`"10+ years"`, `"< 1 year"`) → numeric
* `earliest_cr_line` → datetime; derive credit history length at issue
* Ensure numeric columns aren’t secretly strings with `%` (e.g., `revol_util` sometimes)

**Output**

* Before/after dtype counts.
* A few sanity-check prints (min/max) for parsed fields.

✅ **Validation:** Dates parse cleanly (no mass NaTs), and key fields are numeric.
➡️ **Next:** univariate + bivariate EDA to shape modeling decisions.

---

### 5) Univariate EDA: what matters, no plot spam

**Goal:** understand distributions, outliers, and whether simple models are reasonable.

Focus on a small core set:

* credit: `fico_range_low`, `dti`, `revol_util`, `delinq_2yrs`, `inq_last_6mths`
* income: `annual_inc`
* loan: `loan_amnt`, `term`, `purpose`
* optional investor-visible: `int_rate`

**Output**

* Histograms/boxplots for numeric
* Bar plots for top categoricals
* Target rate

✅ **Validation:** You can explain what “typical” looks like and what the outliers are.
➡️ **Next:** relationship checks (does anything move default rate?).

---

### 6) Bivariate EDA: prove or kill modeling hypotheses

**Goal:** decide whether logistic regression is a sensible baseline and whether nonlinear models are worth it.

Simple, defensible tests:

* Bin numeric features into deciles and plot **default rate by bin**
* For categoricals: plot default rate by category (top N, rest grouped)
* Compare Pearson vs Spearman *only after cleaning* (optional)

This is where you decide:

* Is default risk mostly monotonic in core features? (good for LR)
* Are there obvious nonlinear patterns worth capturing? (might justify trees)

✅ **Validation:** You have 3–6 plots that clearly show how risk changes with the main features.
➡️ **Next:** time drift and split strategy.

---

### 7) Time drift check (because LendingClub is not stationary)

**Goal:** justify time-based validation.

**What you do**

* Plot default rate over `issue_d` (monthly or quarterly)
* Check distribution shift for 3–5 key features across years

**Output**

* A drift plot and a short written conclusion: “We must use time split.”

✅ **Validation:** Drift is quantified, not implied; split plan is locked.
➡️ **Next:** finalize “EDA decisions” section and export cleaned dataset.

---

### 8) EDA conclusions: decisions you’ll carry into modeling

This section should be short and blunt:

* Target definition (final outcomes only)
* Leakage dropped features (with examples)
* Feature set chosen (origination-only + investor-view for ablation)
* Validation scheme (time split)
* Baseline model expectation (LR first, then one tree model only if EDA supports it)

**Output**

* Save `cleaned_dataset.parquet` (or CSV) for Notebook 2.

✅ **Validation:** Notebook 2 can be built without re-deciding anything fundamental.
➡️ **Next:** start modeling with LR baseline + honest evaluation.

---

## One more reality check (because you asked for a restart)

If you don’t have the **full dataset** available locally, don’t pretend Notebook 1 is “EDA.” With 6 rows and no defaults, it’s just schema inspection. That’s fine as a warm-up, but the real EDA must run on the real data.

If you want, I’ll draft the exact **Notebook 1 structure with markdown + code cell skeleton** (clean, minimal, scikit-learn friendly) so you can paste it into Jupyter and run it immediately on the full CSV.
