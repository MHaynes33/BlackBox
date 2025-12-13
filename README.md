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

- ------------------------------------------------------------------------
## III. Phase by Phase Workflow Summary

## Phase 1: Discovery, Data Quality, and Business Logic Hypothesis Formation

### **Goal**

*Understand the structure and quality of the public/private reimbursement datasets and synthesize statistical patterns along with interview
insights to form a testable hypothesis of the ACME legacy system’s reimbursement rules.*

---

### **Data Ingestion & Preparation**

- Imported and flattened the two JSON files:
  - **public_cases** (1,000 rows; 4 variables)
  - **private_cases** (5,000 rows; 3 variables)
- Combined into a unified **6,000-row** table.
- Reimbursement missing in private cases by design.

---

### **Data Quality Validation**

#### **Missing Values**
- No missing values in any input fields for either dataset.
- Expected: reimbursement is missing for private cases only.

#### **Range Validation**
All values fell within realistic constraints:
- Duration: **1 to 14** days  
- Mileage: **5 to 1,348** miles  
- Receipts: **$0.27 to $2,503.46**  
- Public reimbursement: **$117.24 to $2,337.73**

No negative, zero, or implausible entries.

#### **Outlier Detection**
- Applied the **1.5 × IQR** rule to all numeric features.  
- **Zero outliers** detected in public or private datasets.

#### **Public vs Private Comparison**
- Both datasets demonstrated **near-identical distributions** across duration, mileage, and receipts.  
- Statistical behavior strongly aligns → **no evidence of domain drift**.

---

### **Statistical Insights**

- **Key Predictive Variables** (based on correlations and distributions):
  - `total_receipts_amount`
  - `miles_traveled`
  - `trip_duration_days`

- **Receipts** show the strongest relationship with reimbursement.
- **Mileage** exhibits nonlinear effects (diminishing returns).
- **Duration** has moderate influence with tapering beyond longer trips.

Distribution characteristics (right-skewed, long-tailed) support the presence of nonlinear business logic.

---

### **Business Logic Insights (Stakeholder Interviews & PRD)**

Interviews with ACME staff revealed critical behavioral rules embedded within the legacy reimbursement engine:

#### **Trip Duration**
- Optimal reimbursement zone around **4–6 days**.
- **5-day trips** often yield the most favorable output.
- Penalties begin appearing **after 7 days**.

#### **Mileage Behavior**
- Ideal daily mileage window: **~180–220 miles/day**.
- Excessive mileage likely penalized.
- Very low mileage may trigger under-efficiency penalties.

#### **Receipts Behavior**
- System is **receipts-driven**, but not linearly:
  - Diminishing returns at higher spending levels.
  - Penalties for extremely low spend.

#### **Efficiency Bonuses**
- Balanced trips (reasonable mileage, moderate duration, moderate spend) receive positive adjustments.

#### **Legacy System Quirks**
- Department-specific weighting and “memory effects.”
- Rounding patterns using **.49** and **.99** endings.
- **±5–10% pseudo-random adjustments** built into the system’s historical logic.

These qualitative insights match the statistical patterns observed in the data.

---

### **Phase 1 Takeaway**

Phase 1 concludes that ACME’s legacy reimbursement engine is:

- **Nonlinear**  
- **Threshold-based and tiered**  
- **Receipts-dominant** but subject to diminishing returns  
- Influenced by **trip efficiency patterns**  
- Embedded with **historical quirks and random adjustments**  

These findings directly informed the engineered features in Phase 2 and the nonlinear modeling strategies in Phases 3 and 4.


**Files Relevant to this Phase**

- Notebooks/01_EDA_Reimbursement (3).ipynb
- BlackBox/reports/Business_logic_summary.html
- Notebooks/week1_data_cleaning.ipynb
- data/private_cases.json
- data/public_cases.json
- data/combined_clean.csv
- src/INTERVIEWS.md
- src/PRD.md

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

- data/phase2_features_baseline_models.csv
- Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb
- Notebooks/Feature Correlation and Visualization.ipynb
- Notebooks/Performance Summary.ipynb
-  reports/EDA & Baseline Model Info.html
- reports/Feature Definitions and Rationales Table Updated .html

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

### **Goal**

*Explain the Phase 3 model’s behavior, confirm that it matches the interviews/PRD expectations, and surface the business rules it appears to learn*

---

#### **Interpretability Methods Used**

- Tree-based feature importance  
- Permutation importance  
- SHAP-style nonlinear reasoning  
- Residual diagnostics  

---

#### **Most Influential Features (Consistent Across Models)**

1. **total_receipts_amount**  
   - Primary driver of reimbursement  
   - Exhibits nonlinear, diminishing-return behavior  
   - Matches stakeholder statement: *“The system mainly pays back receipts.”*

2. **miles_traveled**  
   - Contributes through **tiered mileage bands**  
   - Reflects historical mileage tables described in interviews

3. **trip_duration_days**  
   - Influences model in **step-wise tiers**  
   - Similar to per-diem structures with tapering after long trips

4. **Engineered Features (Phase 2)**  
   - `cost_per_day`, `cost_per_mile`, `miles_per_day`, `cost_ratio`  
   - Add nuance and stability, especially for edge cases  
   - Represent efficiency and balance patterns described by stakeholders

---

### **Reconstructed Business Rules Learned by the Model**

Interpretability reveals the ensemble model learned rules that match both interview accounts and Phase 1–2 patterns:

- **Receipts-driven reimbursement** with diminishing returns  
- **Mileage tier adjustments** (low, medium, high travel efficiency)  
- **Duration tiers** (sweet spots at ~4–6 days, tapering after 7)  
- **Efficiency bonuses** for balanced trips  
- **Randomized rounding/noise** consistent with:
  - “random adjustments”
  - “unique rounding logic”
  - pseudo-random behavior noted in interviews

---

### **Residual Analysis Findings**

- Residuals are centered around zero → **no systematic bias**.  
- No structural pattern across mileage, receipts, or duration.  
- Cent-level randomness matches the legacy system’s described behavior.  
- This explains why granular match rates remain low (e.g., ~0% within \$0.01, ~1.6% within \$1.00).

---

### **Phase 4 Conclusion**

Phase 4 confirms that the final stacking ensemble:

- **Accurately models the true business behavior** of ACME’s reimbursement engine.  
- Does **not invent new logic**, but instead **reveals the hidden structure** of the legacy system.  
- Provides transparency, auditability, and interpretability necessary for modernization.  
- Successfully reconstructs a **60-year-old black box** into an understandable, data-driven framework.

---

**File Relevant to this Phase**

- Notebooks/Model Interpretability & Feature-Impact Analysis.ipynb

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
**Notebook:** `Notebooks/week1_data_cleaning.ipynb`  
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/week1_data_cleaning.ipynb

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
**Notebook:** `Notebooks/01_EDA_Reimbursement (3).ipynb`  
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/01_EDA_Reimbursement%20(3).ipynb

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
**Report:** `reports/Business_logic_summary.html`

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
**Notebook:** `Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb`  
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb

**What this does**
- Engineers derived efficiency features
- Performs feature sanity checks
- Trains baseline models

**Outputs**
- `data/phase2_features_baseline_models.csv`  
https://github.com/MHaynes33/BlackBox/blob/main/data/phase2_features_baseline_models.csv

**How to run**
- Open the notebook and run all cells.

---

### Step 2: Correlation & Driver Analysis
**Notebook:** `Notebooks/Feature Correlation and Visualization.ipynb`  
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Feature%20Correlation%20and%20Visualization.ipynb

**What this does**
- Visualizes feature relationships and drivers

**Outputs**
- Inline plots

**How to run**
- Open the notebook and run all cells.

---

### Step 3: Baseline Performance Summary
**Notebook:** `Notebooks/Performance Summary.ipynb`  
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Performance%20Summary.ipynb

**What this does**
- Evaluates baseline model performance using regression metrics

**Outputs**
- Inline evaluation tables

**How to run**
- Open the notebook and run all cells.

---

### Step 4: Review Phase 1 EDA & Baseline Model Report
**Report:** `reports/EDA & Baseline Model Info.html`  
https://github.com/MHaynes33/BlackBox/blob/main/reports/EDA%20%26%20Baseline%20Model%20Info.html

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
**Report:** `reports/Feature Definitions and Rationales Table Updated.html`  
https://github.com/MHaynes33/BlackBox/blob/main/reports/Feature%20Definitions%20and%20Rationales%20Table%20Updated%20.html

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

## Phase 3: Nonlinear & Ensemble Modeling (need help!)

### Purpose
Evaluate nonlinear and ensemble regression models and generate holdout predictions.

### Inputs
- `data/phase2_features_baseline_models.csv`

---

### Step 1: Model Development & Integration
**Notebook:** `Notebooks/Model Development & Integration.ipynb`  
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Model%20Development%20%26%20Integration.ipynb

**What this does**
- Trains nonlinear and ensemble models
- Generates predicted vs. actual plots

**Outputs**
- Inline evaluation results

**How to run**
- Open the notebook and run all cells.

---

### Step 2: Holdout Metrics & Predictions
**Script:** `scripts/phase3_performance_metrics.py`  
https://github.com/MHaynes33/BlackBox/blob/main/scripts/phase3_performance_metrics.py

**What this does**
- Runs a 75/25 holdout evaluation
- Reports MAE, RMSE, R², and within-$ thresholds
- Writes holdout predictions

**Outputs**
- `data/phase3_predictions.csv`  
https://github.com/MHaynes33/BlackBox/blob/main/data/phase3_predictions.csv

python scripts/phase3_performance_metrics.py

Phase 3 Outputs Summary

Holdout predictions

Ensemble performance metrics

---

## Phase 4: Model Interpretability & Feature-Impact Analysis

### Purpose

Explain model behavior and validate alignment with PRD documentation and employee interviews.

### Inputs

- `data/phase2_features_baseline_models.csv`
- `data/phase3_predictions.csv`

### Step 1: Interpretability Analysis

Notebook: Notebooks/Model Interpretability & Feature-Impact Analysis.ipynb
https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Model%20Interpretability%20%26%20Feature-Impact%20Analysis.ipynb

**What this does**

- Reviews feature importance and impact
- Examines thresholds and residual patterns
- Connects model behavior to business logic

**Outputs**

- Interpretive plots and narrative explanations
- No datasets written

**How to run**

- Open the notebook and run all cells.

---
### Project Phase by Phase Steps Complete!
- ------------------------------------------------------------------------

## V. Key Findings & How to Run the Model
- ------------------------------------------------------------------------

## VI. Final Notes

~ To get a more detailed overview of the project check our presentation: https://github.com/MHaynes33/BlackBox/blob/main/presentation/Final-Presentation-Phase-1-4.pdf

~ For all other questions reach out to any of the members listed above
