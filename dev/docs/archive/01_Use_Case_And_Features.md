# Use Case & Feature Selection: Investor Perspective

**Status:** Reference Guide
**Audience:** Portfolio reviewers, future maintainers
**Goal:** Clarify feature selection decisions and why this is legitimate investor modeling

---

## The Investor Use Case

### Real-World Scenario

**You:** Individual investor on LendingClub's peer-to-peer lending platform
**Your Goal:** Maximize investment returns by selecting loans with low default risk
**Available Information:** Everything LC publishes (applicant data + LC's grade + interest rate)
**Your Challenge:** Should you trust LC's grade, or build your own model?

### Timeline & Information Availability

```
TIMELINE:

Day 0 (Applicant Day):
├─ Borrower applies with: income, FICO, employment, credit history, etc.
├─ LC's internal process: Analyzes all data, assigns grade A-G
├─ LC pricing: Uses grade to set interest rate (5%-30%)
└─ Loan published: Listed on platform with all above info

         ↓

Day 1 (Investor Day) [YOU ARE HERE]
├─ Loan appears on platform
├─ You see: All applicant data + grade + int_rate + installment
├─ You must decide: "Should I fund this loan?"
├─ Your question: "Will this person default?"
└─ Your goal: Make better predictions than LC to beat market returns

         ↓

Days 1-60 (Loan Months):
└─ Borrower makes payments or defaults

         ↓

Future (Your Analysis):
└─ Did your model predict correctly?
```

---

## Features Available at YOUR Decision Point (Day 1)

### Applicant Data (Pre-Origination)
✅ **Always Available to You**
- FICO score (fico_range_low, fico_range_high)
- Employment length (emp_length)
- Annual income (annual_inc)
- Debt-to-income ratio (dti)
- Home ownership (home_ownership)
- Verification status (verification_status)
- Credit history (earliest_cr_line)
- Account diversity (open_acc, total_acc, revol_acc, mort_acc, etc.)
- Recent activity (inq_last_6mths, acc_open_past_24mths)
- Credit utilization (revol_bal, revol_util)
- Public records (pub_rec, pub_rec_bankruptcies)
- Delinquency (delinq_2yrs)

### Loan Terms (Published by LC)
✅ **Always Available to You**
- Loan amount (loan_amnt)
- Term (36 vs 60 months)
- Purpose (debt_consolidation, credit_card, etc.)
- **Interest rate (int_rate)** ← Published feature
- **Monthly installment (installment)** ← Calculated from term + amount + rate
- **Grade (grade, sub_grade)** ← LC's risk classification

### Post-Origination Data (NOT Available to You Yet)
❌ **Not Available When You Decide**
- Funded amount (funded_amnt)
- Last payment amount
- Last payment date
- Payment status
- Collections data
- Delinquency beyond application
- Interest paid to date
- Total payment received

---

## The Feature Decision: grade vs int_rate

### Grade: Why EXCLUDE it?

**What is `grade`?**
- LC's internal credit risk classification (A1-G5)
- Derived from their analysis of borrower data
- Essentially: "We predict this person will default with X probability"

**Why it's circular:**
```
LC's Process:
grade = f(FICO, DTI, income, employment, credit_history, ...)
default = g(FICO, DTI, income, employment, credit_history, ...)

Our Model WITH grade:
default = h(grade, FICO, DTI, ...)
        = h(LC's prediction, applicant data, ...)

This is circular! We're using LC's prediction to make our own prediction.
```

**Better Approach:**
- Don't use `grade`
- Use the **same inputs** LC used (FICO, DTI, etc.)
- Build our **own model**
- **Compare our predictions to LC's grade** to see if we do better

**What We Gain:**
- Independent risk assessment
- Portfolio depth (show we can replicate LC's thinking)
- Ability to identify overpriced/underpriced loans
- Demonstrates modeling skill, not just trusting someone else

### Interest Rate: Why INCLUDE it?

**What is `int_rate`?**
- Lending Club's pricing decision
- Influenced by: grade, market conditions, risk policy, portfolio management
- Published information (every investor sees it)

**Why it's NOT circular:**
```
int_rate reflects:
- LC's risk assessment (grade)
- PLUS: Market conditions
- PLUS: LC's business strategy
- PLUS: Macroeconomic factors
```

Interest rate contains information **beyond just risk**. Two loans with same risk might have different rates due to:
- When they were originated (market conditions changed)
- Borrower negotiation
- LC's portfolio composition needs
- Risk-adjusted pricing strategy

**Real Example:**
- Loan A: Grade B, 12% rate → LC says "moderate risk, normal pricing"
- Loan B: Grade B, 10% rate → LC says "moderate risk, but we're pricing it low"
- Your Model: Could these two loans have different default rates despite same grade?

**What We Gain:**
- Additional information about LC's view (higher rate = higher risk)
- Market signal (economic factors embedded in rates)
- Allows model to learn if LC's pricing is good or not
- Realistic investor scenario (you DO see the interest rate)

---

## Feature Categories for Our Model

### Category 1: Core Credit Indicators
**These directly measure financial health and payment capacity**

- FICO score (primary indicator of payment history)
- DTI (ability to service debt)
- Employment length (job stability)
- Annual income (income stability)
- Credit history (experience with credit)

### Category 2: Credit Account Profile
**These show diversification and historical behavior**

- Number of open accounts
- Number of installment accounts
- Number of revolving accounts
- Revolving balance and utilization
- Public records (negative signals)
- Recent delinquencies

### Category 3: Recent Activity
**These indicate financial stress signals**

- Credit inquiries in last 6 months
- Accounts opened in last 24 months
- New account activity

### Category 4: Loan Economics
**These reflect loan characteristics and LC's pricing**

- Loan amount
- Loan term (36 vs 60 months)
- Loan purpose
- Interest rate (LC's pricing decision)
- Monthly installment (repayment burden)

---

## Features Explicitly EXCLUDED & Why

### Excluded: `grade` and `sub_grade`
- **Why:** LC's proprietary credit score (circular reasoning)
- **Impact:** Don't use their risk assessment to predict default
- **Alternative:** Use their inputs (FICO, DTI, etc.) to make our own assessment

### Excluded: Post-Origination Features
These would only be available AFTER the loan is already issued:
- `funded_amnt` - shows if loan was actually funded
- `issue_d` - origination date (but available in historical data)
- Payment history, collections, delinquency beyond 2 years
- Interest paid, total received

**Why:** You must predict BEFORE the loan reaches these states

### NOT Excluded: `int_rate` and `installment`
- **These are published features** when loan is listed
- **They're operational data**, not predictions
- **They contain signal** about market conditions and pricing strategy
- **Investors use them** to evaluate loan economics

---

## Model Building Strategy

### Step 1: Use Applicant Data to Build Foundation
- FICO, DTI, income, employment, credit history
- Build model: "How well do fundamentals predict default?"
- Expected: Moderate performance (0.52-0.58 AUC)

### Step 2: Add Loan Economics
- Include interest rate and installment
- Build enhanced model: "Does pricing signal help?"
- Expected: Slight improvement (0.58-0.62 AUC)

### Step 3: Model Comparison
- Compare to LC's grade: How well does our model align?
- Identify outliers: Where do we disagree with LC?
- Business insight: Are there over/under priced loans?

### Step 4: Investment Strategy
- Use model to select loans for portfolio
- Monitor performance: Do we beat the market?
- Refine: Does our model actually predict better?

---

## Validation Strategy

### Ask These Questions of Your Model

1. **Feature Check:** "Are we using legitimate, available features?"
   - ✅ No grade (circular)
   - ✅ Yes to applicant data (fundamental credit indicators)
   - ✅ Yes to int_rate (investor pricing information)

2. **Leakage Check:** "Can we make this prediction at investor decision time?"
   - ✅ All features are published on loan listing
   - ❌ Would fail if using payment history (not available yet)
   - ❌ Would fail if using collections (not available yet)

3. **Business Check:** "Does this make business sense?"
   - ✅ Investor can use this to select loans
   - ✅ Model can identify underpriced loans
   - ✅ Independent assessment (not copying LC's grade)

4. **Portfolio Value:** "Does this demonstrate data science skill?"
   - ✅ Proper feature selection (don't use circular features)
   - ✅ Understanding of leakage and timing
   - ✅ Clear business context and narrative
   - ✅ Progressive model building and comparison

---

## Summary Table: Feature Decision Matrix

| Feature | Category | Include? | Why |
|---------|----------|----------|-----|
| grade, sub_grade | Credit Score | ❌ NO | Circular (LC's prediction) |
| int_rate | Pricing | ✅ YES | Published, contains signal |
| installment | Economics | ✅ YES | Published, loan burden indicator |
| fico_score | Fundamentals | ✅ YES | Core credit indicator |
| dti | Fundamentals | ✅ YES | Debt capacity |
| employment_length | Stability | ✅ YES | Job stability |
| inq_last_6m | Stress Signal | ✅ YES | Financial stress indicator |
| last_payment_amt | Post-Origination | ❌ NO | Not available at decision time |
| collections | Post-Origination | ❌ NO | Not available at decision time |
| delinq_2yrs | History | ✅ YES | Past behavior (available at application) |

---

## Research Context

This approach aligns with best practices in peer-to-peer lending research:

> "Out of loans classified as safe by predictive models, only 15% defaulted, allowing a decrease in default frequency by almost 50%."

Investors who build independent risk models can significantly beat market average returns.

---

**Next Step:** See `04_Model_Selection_Philosophy.md` for progressive model building approach.
