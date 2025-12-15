# Reverse Engineering ACME’s Legacy Reimbursement System Using Machine Learning and Business Logic Analysis

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
- ------------------------------------------------------------------------
## II. Repository Structure

*This repository is organized to support a phase-by-phase reverse-engineering workflow used to reconstruct ACME’s legacy reimbursement engine. The structure separates exploratory notebooks, data artifacts, formal documentation, evaluation scripts, and the final executable reimbursement engine while maintaining clear traceability across Phases 1–4.*

---

### Main Structure

```text
├── .idea/
├── Notebooks/
├── data/
├── presentation/
├── reports/
├── scripts/
├── src/
├── .gitignore
├── LICENSE
├── README.md
└── deleteme.txt

```

---

### File Guide

*Listed below are the key files found within the repository*

------------------------------------------------------------------------

### `Notebooks/` — Phase-Driven Analysis, Modeling, and Diagnostics

**Purpose**  
Narrative-driven Jupyter notebooks used to explore data, engineer features, train models, evaluate performance, and interpret results.

**Phases Supported:** 1–4

#### Phase 1 — Data Validation & Statistical EDA
- `week1_data_cleaning.ipynb`  
- `01_EDA_Reimbursement.ipynb`

#### Phase 2 — Feature Engineering & Baseline Modeling
- `02_Feature_Engineering_and_Baseline_Model.ipynb`  
- `08_Feature_Correlation_and_Visualization.ipynb`  
- `06_Performance_Summary.ipynb`  
- `09_Model_Evaluation_Checklist.ipynb`

**Purpose:**  
Sanity-check baseline model behavior, validate assumptions, and ensure feature engineering and evaluation steps meet modeling standards before advancing to nonlinear methods.

#### Phase 3 — Nonlinear & Ensemble Modeling
- `07_Model_Development_and_Integration.ipynb`  
- `03_Ensemble_Model_Performance.ipynb`  
- `04_Ensemble_Model_Metrics.ipynb`

**Purpose:**  
Train and evaluate nonlinear and ensemble regressors and generate official holdout predictions.

#### Phase 4 — Interpretability & Diagnostic Analysis
- `05_Model_Interpretability_and_Feature_Impact.ipynb`  
- `Calibrated Ensemble Performance (with Residual Model).ipynb`

**Important:**  
The calibrated residual notebook is a **Phase 4 diagnostic and interpretability experiment**.  
It does **not** replace the official Phase 3 stacking ensemble model.

------------------------------------------------------------------------

### `data/` — Raw Inputs, Cleaned Data, and Modeling Artifacts

**Phases Supported:** 1–3

**Raw Inputs**
- `public_cases.json`
- `private_cases.json`

**Cleaned & Validated Datasets**
- `public_clean.csv`
- `private_clean.csv`
- `combined_clean.csv`

**Feature-Engineered Modeling Table**
- `phase2_features_baseline_models.csv`

**Phase 3 Predictions**
- `phase3_predictions.csv`

**Reference**
- `TeamProjectInstructions.qmd`

------------------------------------------------------------------------

### `reports/` — Formal Documentation & Deliverables

**Phases Supported:** 1–4

**Business Logic Documentation**
- `Business_logic_summary.qmd`
- `Business_logic_summary.html`

**EDA & Baseline Modeling**
- `EDA & Baseline Model Info.qmd`
- `EDA & Baseline Model Info.html`

**Feature Engineering Rationale**
- `Feature Definitions and Rationales Table Updated.qmd`
- `Feature Definitions and Rationales Table Updated.html`

**Phase 3–4 Addenda & Tuning**
- `phase3_phase4_addendum.qmd`
- `phase3_phase4_addendum.pdf`
- `postprocessing_tuning.qmd`

**Final Summary**
- `project_Summary.qmd`
- `project_Summary.pdf`

------------------------------------------------------------------------

### `scripts/` — Evaluation & Automation Utilities

**Phases Supported:** 3

- `phase3_performance_metrics.py` — computes MAE, RMSE, R², and match-rate thresholds  
- `predict_test.py` — end-to-end prediction testing  
- `predict_rules_test.py` — rule-based benchmark comparisons  

------------------------------------------------------------------------

### `src/` — Executable Reimbursement Engine

**Phases Supported:** 3–4

- `final_model.pkl` — **official Phase 3 stacking ensemble**
- `predict.py`
- `predict_rules.py`
- Execution helpers:
  - `eval.sh`
  - `generate_results.sh`
  - `run.sh.template`
- Business references:
  - `INTERVIEWS.md`
  - `PRD.md`
- Engine-specific `README.md`

------------------------------------------------------------------------

### `presentation/` — Final Project Synthesis

- `ACME_Project_Presentation.pdf`

------------------------------------------------------------------------

### Repository Metadata

- `.idea/` — IDE configuration (non-essential)
- `.gitignore`, `LICENSE`
- Root `README.md`
- `deleteme.txt`


- ------------------------------------------------------------------------
## III. Phase by Phase Project Pipeline Overview

*This project follows a four-phase workflow to reverse-engineer ACME Corporation’s
60-year-old legacy reimbursement engine, progressing from discovery to modeling
and interpretability.*

---

### Phase 1: Discovery, Data Quality, & Business Logic Hypothesis

**Goal:**  
Understand dataset structure, validate data quality, and synthesize statistical
patterns with interview and PRD insights to hypothesize the legacy system’s logic.

**Key Results:**
- Public (1,000 rows) and private (5,000 rows) datasets show no missing values
  in input features and no statistical outliers.
- Public and private distributions align closely (no domain drift).
- Reimbursement behavior is:
  - Receipts-dominant
  - Nonlinear with diminishing returns
  - Influenced by mileage and duration tiers
  - Subject to rounding quirks and stochastic noise

**Outcome:**  
A validated dataset and a testable hypothesis that ACME’s reimbursement engine
is nonlinear, threshold-based, and efficiency-driven.

---

### Phase 2: Feature Engineering & Baseline Modeling

**Goal:**  
Encode Phase 1 insights into engineered features and establish baseline performance.

**Engineered Features:**
- `cost_per_day`
- `cost_per_mile`
- `miles_per_day`
- `cost_ratio`

**Baseline Models Evaluated:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Polynomial Regression (degree 2)

**Key Results:**
- Polynomial regression achieved the strongest baseline performance:
  - **R² ≈ 0.892**
- Linear models captured global trends but failed to model nonlinear behavior.

**Outcome:**  
Confirmed that nonlinear modeling is required before advancing to ensembles.

---

### Phase 3: Nonlinear & Ensemble Modeling (Official Model Selection)

**Goal:**  
Develop advanced nonlinear and ensemble models capable of replicating ACME’s
tiered and diminishing-return reimbursement logic.

**Models Trained:**
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- MLP Neural Network
- **Stacking Ensemble (Final Phase 3 Model)**

**Model Performance (Holdout Test Set):**

| Model              | MAE   | RMSE   | R²     |
|-------------------|-------|--------|--------|
| Decision Tree     | 113.02 | 173.45 | 0.8561 |
| Random Forest     | 72.98  | 110.01 | 0.9421 |
| Gradient Boosting | 72.09  | 109.71 | 0.9424 |
| SVR               | 93.26  | 136.25 | 0.9112 |
| MLP Neural Net    | 134.88 | 177.22 | 0.8498 |
| **Stacking Ensemble** | **66.56** | **102.34** | **0.9499** |

**Granular Match Rates (Stacking Ensemble):**
- ~0% within $0.01
- ~1.6% within $1.00

These low exact-match rates are expected due to cent-level randomness embedded
in ACME’s legacy system.

**Outcome:**  
The stacking ensemble was selected as the **official Phase 3 model**, providing
the best balance of accuracy, generalization, and alignment with business logic.

---

### Phase 4: Interpretability, Diagnostics, & Validation

**Goal:**  
Explain *why* the Phase 3 model works, validate alignment with interviews and PRD
documentation, and analyze residual behavior.

**Methods Used:**
- Tree-based feature importance
- Permutation-style importance
- SHAP-inspired nonlinear reasoning
- Residual diagnostics

**Most Influential Features:**
1. `total_receipts_amount` (primary driver; diminishing returns)
2. `miles_traveled` (tiered mileage effects)
3. `trip_duration_days` (stepwise, per-diem-like influence)

**Key Insight:**
Residuals are centered around zero with no systematic bias, confirming that
remaining error reflects intentional legacy randomness rather than model failure.

**Additional Diagnostic Work (Non-Production):**
A calibrated stacking + residual model was explored **for diagnostic and
interpretability purposes only**, yielding slightly improved metrics
(MAE ≈ 66.44, RMSE ≈ 92.07), but this model is **not** the official Phase 3 model.

**Outcome:**  
Phase 4 confirms that the Phase 3 ensemble faithfully reproduces ACME’s historical
reimbursement logic while providing transparency and auditability.

---

### Final Takeaway

Across four phases, this project transformed an undocumented, rule-based legacy
system into a transparent, explainable, and high-performing machine learning
pipeline—preserving historical behavior while enabling modernization and trust.


- ------------------------------------------------------------------------

## IV. How to Run this Project (Phase by Phase)


### Overview

This project is organized into four sequential phases.  Each phase corresponds directly to notebooks, scripts, and reports in the repository and can be executed independently once required inputs are available.

---

## Phase 1:  Data Validation, Statistical EDA, and Business Logic Review

### Purpose
*Prepare the ACME reimbursement datasets for downstream modeling by validating data integrity, performing exploratory analysis, and reviewing business-rule context derived from interviews and PRD documentation.*

### Inputs
- `data/public_cases.json`
- `data/private_cases.json`

---

### Step 1: Data Validation & Cleaning
**Notebook:** [week1_data_cleaning.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/week1_data_cleaning.ipynb)

**What this does**
- Loads and flattens both JSON datasets
- Performs structured data-quality checks
- Validates public-case expected outputs
- Generates exploratory visual diagnostics

**Outputs**
- Inline plots and data-quality summaries  
- No datasets written

**How to run**
- Open the notebook in JupyterLab and run all cells.

---

### Step 2: Statistical EDA & Dataset Comparison
**Notebook:** [01_EDA_Reimbursement .ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/01_EDA_Reimbursement%20.ipynb)

**What this does**
- Combines public and private datasets
- Computes descriptive statistics
- Generates public vs. private comparisons

**Outputs**
- `combined_clean.csv`
- `public_clean.csv`
- `private_clean.csv`
- Inline visualizations

**How to run**
- Open the notebook and run all cells sequentially.

---

### Step 3: Business Logic Summary Review
**Report:** [Business_logic_summary.html](https://github.com/MHaynes33/BlackBox/blob/main/reports/Business_logic_summary.html)

**What this contains**
- Interview- and PRD-derived reimbursement logic hypotheses

**Outputs**
- No datasets written

**How to run**
- Open the HTML file in a browser.

---

### Phase 1 Outputs Summary
- Validated datasets
- EDA context for modeling
- Business-logic hypotheses guiding feature engineering

---

## Phase 2: Feature Engineering & Baseline Modeling

### Purpose

*Engineer derived features and establish baseline models and benchmarks.*

### Inputs
- Phase 1 validated data 

---

### Step 1: Feature Engineering & Baseline Models
**Notebook:** [02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb)

**What this does**
- Engineers derived efficiency features
- Performs feature sanity checks
- Trains baseline models

**Outputs**
- `data/phase2_features_baseline_models.csv`

**How to run**
- Open the notebook and run all cells.

---

### Step 2: Correlation & Driver Analysis
**Notebook:** [08_Feature_Correlation_and_Visualization.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/%2008_Feature_Correlation_and_Visualization.ipynb)

**What this does**
- Visualizes feature relationships and drivers

**Outputs**
- Inline plots

**How to run**
- Open the notebook and run all cells.

---

### Step 3: Baseline Performance Summary
**Notebook:** [06_Performance_Summary.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/%2006_Performance_Summary.ipynb)

**What this does**
- Evaluates baseline model performance using regression metrics

**Outputs**
- Inline evaluation tables

**How to run**
- Open the notebook and run all cells.

---

### Step 4: Review Phase 1 EDA & Baseline Model Report
**Report:** [EDA & Baseline Model Info.html](https://github.com/MHaynes33/BlackBox/blob/main/reports/EDA%20%26%20Baseline%20Model%20Info.html)

**What this contains**
- Phase 1 EDA recap
- Statistical summaries
- Baseline model documentation and results

**Outputs**
- No datasets written

**How to run**
- Open the HTML file in a browser.

---

### Step 5: Review Feature Definitions & Rationales
**Report:** [Feature Definitions and Rationales Table Updated .html](https://github.com/MHaynes33/BlackBox/blob/main/reports/Feature%20Definitions%20and%20Rationales%20Table%20Updated%20.html)

**What this contains**
- Feature definitions
- Feature engineering rationale
- Target variable documentation

**Outputs**
- No datasets written

**How to run**
- Open the HTML file in a browser.

---

### Phase 2 Outputs Summary
- Engineered modeling dataset
- Baseline benchmarks
- Feature reference documentation

---

## Phase 3: Nonlinear & Ensemble Modeling

### Purpose
Evaluate nonlinear and ensemble regression models capable of capturing ACME’s
tiered, nonlinear, and diminishing-return reimbursement logic, and select a
final production-quality model for interpretability analysis in Phase 4.

This phase produces the **official Phase 3 stacking ensemble model** and
generates holdout predictions for downstream analysis.

---

### Inputs
- `data/phase2_features_baseline_models.csv`

---

### Step 1: Model Development & Integration

**Notebook:**  [07_Model_Development_and_Integration.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/%2007_Model_Development_and_Integration.ipynb)

#### What this does
- Loads the Phase 2 engineered feature dataset
- Applies a 75% / 25% train–test split
- Trains multiple nonlinear regression models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Regression (SVR)
  - MLP Neural Network
- Trains a **stacking ensemble** combining tree-based and smooth learners
- Evaluates all models using MAE, RMSE, and R²
- Generates Actual vs. Predicted plots for visual assessment

#### Outputs
- Inline evaluation metrics and comparison tables
- Diagnostic plots (no files written)

#### How to run
Open the notebook in JupyterLab and run all cells sequentially.

---

### Step 2: Holdout Metrics & Prediction Generation

**Script:**  
`scripts/phase3_performance_metrics.py`  

#### What this does
- Loads the Phase 2 modeling dataset
- Re-trains the **final Phase 3 stacking ensemble**
- Evaluates performance on a 75% / 25% holdout split
- Reports:
  - MAE
  - RMSE
  - R²
  - Granular match rates (≤ $0.01, ≤ $1.00, ± $5.00)
- Writes holdout predictions for auditability and interpretability

#### Outputs
- `data/phase3_predictions.csv`  

#### How to run
From the repository root:
```bash
python scripts/phase3_performance_metrics.py
```

---

## Phase 4: Model Interpretability & Feature-Impact Analysis

### Purpose
Explain Phase 3 model behavior, validate alignment with PRD documentation and employee interviews, and diagnose why cent-level matching remains rare even when global accuracy is high.

### Inputs
- `data/phase2_features_baseline_models.csv`
- `data/phase3_predictions.csv`

------------------------------------------------------------------------

### Step 1: Interpretability & Feature-Impact Analysis

**Notebook:** [05_Model_Interpretability_and_Feature_Impact.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/%2005_Model_Interpretability_and_Feature_Impact.ipynb)

**What this does**
- Reviews feature importance and feature impact patterns
- Examines threshold-like behavior learned by tree-based models
- Interprets residual behavior and ties model behavior back to PRD/interview business logic

**Outputs**
- Interpretive plots and narrative explanations (in-notebook)
- No datasets written

**How to run**
- Open the notebook and run all cells sequentially

------------------------------------------------------------------------

### Step 2: Diagnostic Calibration & Residual Modeling (Post-Processing Experiment)

**Notebook:** [Calibrated Ensemble Performance (with Residual Model).ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Calibrated%20Ensemble%20Performance%20(with%20Residual%20Model).ipynb)

> **Important:** This notebook is Phase 4 *diagnostic post-processing* and does **not** replace the official Phase 3 stacking ensemble model.  
> It is used to evaluate generalization behavior, residual structure, and whether small post-hoc corrections reduce error without changing the underlying learned logic.

**What this does**
- Adds PRD-inspired diagnostic features (e.g., spend bands, receipt-cent flags, efficiency sweet-spot indicators)
- Trains a **regularized stacking ensemble** and reports MAE/RMSE/$R^2$
- Trains a **regularized residual correction model** to test whether remaining errors are systematic
- Runs train vs. test diagnostics to assess over/underfitting and generalization gaps
- Computes match-rate thresholds (Exact ≤ \$0.01, ≤ \$1, and ±\$5)

**Outputs**
- Prints calibrated performance metrics and match rates inside the notebook
- Diagnostic evidence supporting whether residual mismatch is structured or consistent with legacy rounding/noise behavior
- No required project artifacts for downstream phases (Phase 4 is interpretability/diagnostic-focused)

**How to run**
- Open the notebook and run all cells sequentially
- Confirm the input path points to `data/phase2_features_baseline_models.csv` 

---
### Project Phase by Phase Steps Complete!
- ------------------------------------------------------------------------

## V. How to Run the Final Model: Stacking Ensemble
- ------------------------------------------------------------------------

# ACME Legacy Reimbursement – Final Model Overview

This repository contains multiple experiments and notebooks, but only one model artifact and one execution path represent the final reported results for Phase 3 and Phase 4.

## What matters

- **Final model artifact:** `src/final_model.pkl`  
  This is the tuned stacking ensemble used in the report. It is a binary file and is **not meant to be opened in an editor**.

- **Official Phase 3 evaluation:** `scripts/phase3_performance_metrics.py`  
  This script loads the final model and computes the metrics reported in the project summary.

## How to reproduce reported metrics

From the **project root** (the folder containing `src/`, `data/`, `scripts/`):

```
python scripts/phase3_performance_metrics.py
```

This script:
- Loads `src/final_model.pkl` (no retraining)
- Uses `data/phase2_features_baseline_models.csv`
- Applies a 75/25 holdout split
- Prints MAE, RMSE, R², and close/exact match rates
- Writes predictions to `data/phase3_predictions.csv`

## Expected results (approximate)

- MAE: ~62  
- RMSE: ~95  
- R²: ~0.95  
- Exact matches (≤ $0.01): ~0%  
- Close matches (≤ $1.00): ~1.6%  

If you see materially different values (e.g., MAE ~79), you are likely evaluating a retrained baseline model rather than the final saved artifact.

## Notebooks

- `Notebooks/04_Ensemble_Model_Metrics.ipynb`  
  Interactive version of the same evaluation. Useful for inspection and learning. Results should match the script because it loads the same final model.

Other notebooks may retrain models for exploration and should **not** be used for final reporting.

## Documentation

Detailed business context, modeling rationale, interpretability findings, and limitations are documented in the project summary QMD and rendered PDFs under `reports/`.


## VI. Final Notes

~ To get a more detailed overview of the project read through our presentation: [ACME Project Presentation](https://github.com/MHaynes33/BlackBox/blob/main/presentation/ACME_Project_Presentation.pdf) or watch it: [ACME_Project_Recorded_Presentation](

~ For all other questions reach out to any of the members listed above
