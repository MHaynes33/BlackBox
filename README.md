# ACME Reimbursement Engine — Reverse Engineering Legacy Business Logic (working on the title)

### East Carolina University • Machine Learning / Data Science Graduate Project

*This repository showcases our full machine learning workflow for reverse engineering ACME’s legacy reimbursement engine. It includes all datasets, Jupyter notebooks, modeling scripts, and reports required to recreate the system and understand the underlying business logic, covering the complete analysis from Phase 1 through Phase 4.*

**Team Members:**
- **Ayushi Bohra** — Technical Lead
- **Mike Haynes** — Business Analyst
- **Matthew Fernald** — Quality Analyst
- **Colyn Martin** — Documentation & Communication Lead

- ------------------------------------------------------------------------

## I. Project Overview

ACME’s internal reimbursement engine has operated for over 60 years with undocumented formulas, nonlinear rules, hidden thresholds, department-specific quirks, and unpredictable rounding behaviors. The goal of this project was to **reverse engineer** this black-box system using quantitative analysis, qualitative insights, and machine learning.

Across four phases, our team:

-   Validated and explored ACME’s public/private reimbursement datasets, identifying receipts, mileage, and duration as the core statistical drivers of reimbursement.
-   Engineered efficiency-based features that capture diminishing returns, travel balance, and real-world spending behaviors.
-   Trained baseline and advanced nonlinear models, ultimately developing a **stacking ensemble** that most accurately replicated ACME’s historical outputs.
-   Interpreted the final model to reconstruct ACME’s underlying rules, confirming alignment with stakeholder interviews and PRD expectations.

The resulting model achieves **high predictive accuracy (\~0.95 R²)** while maintaining interpretability, providing ACME a modern, transparent foundation for replacing/recreating its legacy reimbursement engine.

## II. Repository Structure (work in progress)

## III. Phase by Phase Workflow Summary

### Phase 1: Discovery, Data Quality, and Business Logic Hypothesis Formation

**Technology Used**

---

## Phase 2: Feature Engineering and Baseline Modeling

#### **Goal**

*Translate Phase 1 behavioral findings into engineered efficiency
features and evaluate how well baseline models approximate ACME’s legacy
reimbursement logic*

---

#### **Engineered Features**

To capture nonlinear travel behavior and business rules described in interviews, we engineered four efficiency-based features:

| Feature           | Definition                             | Business Purpose |
|-------------------|-----------------------------------------|------------------|
| `cost_per_day`    | `total_receipts_amount / trip_duration_days` | Spending intensity; identifies overspend/underspend penalties |
| `cost_per_mile`   | `total_receipts_amount / miles_traveled`     | Efficiency per mile; flags abnormal travel cost patterns |
| `miles_per_day`   | `miles_traveled / trip_duration_days`        | Travel rate; reflects mileage tiers discussed in interviews |
| `cost_ratio`      | `cost_per_day / cost_per_mile`               | Balance indicator between day-based and distance-based costs |

**Findings:**
- All engineered features were valid (no NaN or infinite values).
- Distributions were right-skewed, matching real-world travel behavior.
- Features added nonlinear nuance necessary for downstream modeling.

---

#### **Extended Exploratory Data Analysis**

Key observations from the consolidated EDA:

- No missing values across input features.  
- Range validation confirmed all values fall within plausible travel limits.  
- Outlier detection (1.5×IQR) found **zero outliers** in public or private datasets.  
- Public and private datasets show **aligned distributions**, confirming no domain drift.  
- Strongest data-driven predictors of reimbursement:
  - **Total receipts**
  - **Miles traveled**
  - **Trip duration**

These trends reinforce Phase 1’s business insights and support nonlinear behavior modeling.

---

#### **Baseline Models Evaluated**

The following models were implemented using the engineered + original features:

- **Linear Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Polynomial Regression (Degree 2)**

Train/test split: **75% / 25%**

---

#### **Baseline Model Performance**

The polynomial regression model produced the strongest approximation of ACME’s legacy behavior.

| Model                     | R²     | RMSE     | MAE       | Notes |
|---------------------------|---------|----------|-----------|-------|
| Linear Regression         | 0.784   | 199.85   | 159.59    | Captures broad linear effects |
| Ridge Regression          | 0.784   | 199.84   | —         | Stabilizes coefficients |
| Lasso Regression          | 0.784   | 199.85   | —         | Produces simpler model |
| Polynomial (Degree 2)     | 0.892   | 141.64   | —         | Best baseline; captures nonlinear interactions |

**Interpretation:**
- Linear-style models recover ~78% of reimbursement variance.  
- Polynomial modeling increases explanatory power to ~89%, revealing:
  - diminishing returns,  
  - threshold effects,  
  - nonlinear business logic.  

These patterns mirror stakeholder descriptions of ACME’s historical reimbursement system.

---

#### **Phase 2 Conclusion**

Phase 2:

- Integrated Phase 1 statistical and interview findings  
- Engineered behavioral features reflecting business rules  
- Established strong baseline performance benchmarks  
- Demonstrated that nonlinear modeling is essential to replicate ACME’s legacy engine  

The dataset, engineered features, and baseline performance results fully prepared the pipeline for **Phase 3’s nonlinear and ensemble modeling strategies.**

---

#### Phase 2 Relevant Sources:

**Files Relevant to this Phase**

**Technical Tools Used**

---

## Phase 3: Modeling Outlook & Integration Plan.

Goal:

*Advance beyond Phase 2 polynomial baselines by evaluating nonlinear and
ensemble modeling strategies capable of capturing ACME’s tiered, nonlinear,
and diminishing-return reimbursement logic. Phase 3 introduces a structured
modeling pipeline, evaluates individual nonlinear regressors, integrates PRDdriven business logic features, and develops a calibrated stacking ensemble
that most closely replicates ACME’s 60-year-old legacy system.*

---

**Models Trained**

1. **Decision Tree**
2. **Random Forest**
3. **Gradient Boosting**
4. **Support Vector Regression (SVR)**
5. **MLP Neural Network**
6. **Stacking Ensemble (Final Model)**

---

#### Model Performance

| Model              | MAE    | RMSE    | R²      |
|--------------------|--------|---------|---------|
| Decision Tree      | 113.02 | 173.45  | 0.8561  |
| Random Forest      | 72.98  | 110.01  | 0.9421  |
| Gradient Boosting  | 72.09  | 109.71  | 0.9424  |
| SVR                | 93.26  | 136.25  | 0.9112  |
| MLP Neural Net     | 134.88 | 177.22  | 0.8498  |
| **Stacking Ensemble** | **66.56** | **102.34** | **0.9499** |

---
#### Granular Hit Rates

~0% within $0.01

~1.6% within $1.00

_ACME’s legacy engine adds random cent-level noise, making exact matches impossible._

---

#### Business Alignment

~ Trees capture threshold rules → like internal ACME policy.

~ Boosting models stacked exceptions → like decades of patches.

~ SVR/MLP capture smooth diminishing-return curves → matching spend patterns.

~ Stacking Ensemble combines all rule behaviors → closest to ACME’s logic.

---

#### Phase 3 Outcome

The stacking ensemble provides the most accurate and business-aligned reconstruction of ACME’s historical reimbursement behavior.

#### Phase 3 Relevant Sources:

**Files Relevant to this Phase**

**Technical Tools Used**

---

## Phase 4: Model Interpretability & Feature-Impact


**Technology Used**

## 4. How to Run this Project

## 5. Key Findings

## 6. Acknowledgements

## 7. Final Notes
