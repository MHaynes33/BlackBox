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

### Phase 1

**Technology Used**

### Phase 2

**Technology Used**

### Phase 3: Modeling Outlook & Integration Plan

Goal:

_Advance beyond Phase 2 polynomial baselines by evaluating nonlinear and
ensemble modeling strategies capable of capturing ACME’s tiered, nonlinear,
and diminishing-return reimbursement logic. Phase 3 introduces a structured
modeling pipeline, evaluates individual nonlinear regressors, integrates PRDdriven business logic features, and develops a calibrated stacking ensemble
that most closely replicates ACME’s 60-year-old legacy system.

**Models Trained**

Decision Tree

Random Forest

Gradient Boosting

Support Vector Regression (SVR)

MLP Neural Network

Stacking Ensemble (Final Model)

#### Model Performance

| Model              | MAE    | RMSE    | R²      |
|--------------------|--------|---------|---------|
| Decision Tree      | 113.02 | 173.45  | 0.8561  |
| Random Forest      | 72.98  | 110.01  | 0.9421  |
| Gradient Boosting  | 72.09  | 109.71  | 0.9424  |
| SVR                | 93.26  | 136.25  | 0.9112  |
| MLP Neural Net     | 134.88 | 177.22  | 0.8498  |
| **Stacking Ensemble** | **66.56** | **102.34** | **0.9499** |

#### Granular Hit Rates

~0% within $0.01

~1.6% within $1.00

_ACME’s legacy engine adds random cent-level noise, making exact matches impossible._

#### Business Alignment

~ Trees capture threshold rules → like internal ACME policy.

~ Boosting models stacked exceptions → like decades of patches.

~ SVR/MLP capture smooth diminishing-return curves → matching spend patterns.

~ Stacking Ensemble combines all rule behaviors → closest to ACME’s logic.
 
#### Phase 3 Outcome

The stacking ensemble provides the most accurate and business-aligned reconstruction of ACME’s historical reimbursement behavior.

#### Phase 3 Relevant Sources:

**Files Relvant to this Phase**

**Technical Tools Used**

### Phase 4

**Technology Used**

## 4. How to Run this Project

## 5. Key Findings

## 6. Acknowledgements

## 7. Final Notes
