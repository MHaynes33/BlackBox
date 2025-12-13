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

## II. Repository Structure (work in progress)

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
~ Notebooks/01_EDA_Reimbursement (3).ipynb
~ BlackBox/reports/Business_logic_summary.html
~ reports/Phase1_summary.qmd
~ Notebooks/week1_data_cleaning.ipynb
~ data/private_cases.json
~ data/public_cases.json
~ data/combined_clean.csv
~ src/INTERVIEWS.md
~ src/PRD.md

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
~ data/phase2_features_baseline_models.csv
~ Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb
~ Notebooks/Feature Correlation and Visualization.ipynb
~ Notebooks/Performance Summary.ipynb
~ reports/EDA & Baseline Model Info.html
~ reports/Feature Definitions and Rationales Table Updated .html

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

**Relevant File for Phase 4**

## 4. How to Run this Project (Phase by Phase)
## Phase 1 — How to Run the Data Validation, Statistical EDA, and Business Logic Review

Phase 1 prepares the ACME reimbursement datasets for all downstream modeling.  
It validates data integrity, performs exploratory statistical analysis, and integrates 
business-rule insights extracted from interviews and the PRD.

Phase 1 consists of **three components**, each mapped to a notebook or report.

---

##  Step 1 — Run Data Validation & Cleaning

**Notebook:**  
[`Notebooks/week1_data_cleaning.ipynb`](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/week1_data_cleaning.ipynb)

### What this notebook does
- Loads and flattens both JSON datasets (`public_cases.json` and `private_cases.json`)
- Performs structured data-quality checks:
  - Missing values  
  - Duplicates  
  - Invalid numeric ranges  
  - Public expected_output validation  
- Generates visual diagnostics:
  - Histograms and KDE curves  
  - Boxplots  
  - Correlation heatmaps  
  - Pairplots  
- Runs IQR-based outlier detection (result: **0 outliers detected**)

### Outputs
- Produces no modified datasets  
- Displays plots & data-quality diagnostics  
- Confirms the raw datasets are clean and modeling-ready  

### How to run
Open the notebook in JupyterLab and run all cells.

---

## Step 2 — Run Statistical EDA & Public/Private Dataset Comparison

**Notebook:**  
[`Notebooks/01_EDA_Reimbursement (3).ipynb`](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/01_EDA_Reimbursement%20(3).ipynb)

### What this notebook does
- Loads both datasets using a flexible JSON parser  
- Combines public + private into a single 6000-row dataframe  
- Computes descriptive statistics & missing-value profiling  
- Generates:
  - Correlation heatmap  
  - Individual feature distributions  
  - Pairwise (public vs private) feature plots  
  - Regression-based trend analyses  
  - Public-vs-private reimbursement boxplots  
- Saves cleaned output files:
  - `combined_clean.csv`  
  - `public_clean.csv`  
  - `private_clean.csv`  

### Outputs
- CSV exports for optional use in later phases  
- Visualizations supporting statistical understanding of the datasets

### How to run
Open the notebook and run all cells sequentially. All figures will render inline.

---

##  Step 3 — Review the Business Logic Summary (Interview + PRD Insights)

**Report:**  
[`reports/Business_logic_summary.html`](https://github.com/MHaynes33/BlackBox/blob/main/reports/Business_logic_summary.html)

### What this report contains
A narrative breakdown of the undocumented rules embedded in ACME’s legacy reimbursement engine, including:

- Per-diem base rates  
- Duration sweet spots  
- Mileage efficiency curves  
- Spend-rate penalties & diminishing returns  
- Receipt nonlinearity  
- Rounding anomalies  
- Timing effects (weekday, month, quarter)  
- Department weighting  
- User-profile memory  
- Intentional ±5–10% stochastic noise  

### Purpose
This document forms the **business-rule foundation** for feature engineering in Phase 2  
and the nonlinear modeling strategy in Phases 3 and 4.

### How to run
Open the HTML file in a browser. No execution is required.

---

## Phase 1 Outputs Summary

After completing all three components, you will have:

- Clean, validated datasets with no missing values or outliers  
- Statistical understanding of feature behavior  
- Visual evidence of public/private alignment  
- Cleaned CSV files for later use  
- A clear business-rule hypothesis map for feature engineering  

Phase 1 ensures that the raw ACME datasets, their statistical properties,  
and their business context are fully understood before building models in Phase 2.

________________

# Phase 2: Feature Engineering & Baseline Modeling

Phase 2 transforms the Phase 1 cleaned dataset into newly engineered features and evaluates baseline regression models to establish a quantitative performance benchmark.

Each step below corresponds directly to a project notebook, ensuring full traceability and reproducibility across the workflow.

---

## Phase 2 Workflow Overview

| Step | Notebook |
|-----|----------|
| Load & Validate Data | [02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb) |
| Engineer Derived Features | [02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb) |
| Outlier & Distribution Checks | [02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb) |
| Correlation & Driver Analysis | [Feature Correlation and Visualization.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Feature%20Correlation%20and%20Visualization.ipynb) |
| Baseline Modeling (Linear → Polynomial) | [02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb) |
| Baseline Performance Evaluation | [Performance Summary.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Performance%20Summary.ipynb) |
| Data Output for Phase 3 | [phase2_features_baseline_models.csv](https://github.com/MHaynes33/BlackBox/blob/main/data/phase2_features_baseline_models.csv) |

*This structure ensures transparency and repeatability for every analytical step.*

---

## 1. Loading & Validating the Clean Dataset  
**Notebook:**  
[02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb)

Phase 2 begins by loading the Phase 1 output dataset.

The notebook:
- inspects dataset shape and structure  
- checks for missing values  
- confirms that only **public cases (N = 1,000)** contain reimbursement labels  
- verifies that cleaned variables remain consistent after Phase 1 wrangling  

---

## 2. Engineering Derived Features  
**Notebook:**  
[02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb)

To capture efficiency, balance, and nonlinear travel behavior, the following features are engineered:

| Feature | Formula | Purpose |
|--------|---------|---------|
| `cost_per_day` | receipts ÷ days | Daily spending intensity |
| `cost_per_mile` | receipts ÷ miles | Travel efficiency per mile |
| `miles_per_day` | miles ÷ days | Daily travel intensity |
| `cost_ratio` | cost_per_day ÷ cost_per_mile | Balance between time- and distance-based costs |

These features are:
- protected against division-by-zero  
- cleaned of NaN and infinite values  
- validated using descriptive statistics  

---

## 3. Outlier Detection & Feature Distribution Analysis  
**Notebook:**  
[02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb)

Using an IQR-based approach, the notebook evaluates:
- trip duration  
- miles traveled  
- total receipts  
- engineered efficiency features  

**Summary findings:**
- No extreme anomalies in core input features  
- Engineered cost metrics show expected right-skewed behavior  
- Feature distributions are stable and modeling-ready  

---

## 4. Preparing the Modeling Dataset  
**Notebook:**  
[02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb)

Seven numeric predictors are selected:

- `trip_duration_days`  
- `miles_traveled`  
- `total_receipts_amount`  
- `cost_per_day`  
- `cost_per_mile`  
- `miles_per_day`  
- `cost_ratio`  

**Target variable:** `reimbursement`

A deterministic **75 / 25 train–test split** is applied:
- first 750 public cases → training  
- remaining 250 public cases → testing  

---

## 5. Baseline Model Development  
**Notebook:**  
[02_Feature_Engineering_and_Baseline_Model.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/02_Feature_Engineering_and_Baseline_Model.ipynb)

Four baseline regression models are evaluated:

### 5.1 Linear Regression
- Serves as the baseline comparator  
- Achieves **R² ≈ 0.784**  
- Confirms strong linear influence of receipts, mileage, and duration  

### 5.2 Ridge Regression
- Introduces L2 regularization  
- Produces performance similar to Linear Regression  
- Demonstrates coefficient stability  

### 5.3 Lasso Regression
- Introduces L1 regularization  
- Retains all predictors (no feature elimination)  

### 5.4 Polynomial Regression (Degree 2)
- Introduces second-order terms and feature interactions  
- Achieves the **highest baseline performance (R² ≈ 0.892)**  
- Demonstrates that nonlinear interactions are required to approximate legacy behavior  

---

## 6. Feature Correlation & Driver Analysis  
**Notebook:**  
[Feature Correlation and Visualization.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Feature%20Correlation%20and%20Visualization.ipynb)

Key findings:
- `total_receipts_amount` is the strongest predictor  
- `miles_traveled` and `trip_duration_days` are meaningful contributors  
- Engineered features exhibit low linear correlation but support nonlinear modeling  

---

## 7. Baseline Performance Evaluation  
**Notebook:**  
[Performance Summary.ipynb](https://github.com/MHaynes33/BlackBox/blob/main/Notebooks/Performance%20Summary.ipynb)

Baseline models are evaluated using:
- MAE  
- RMSE  
- R²  

Results confirm:
- Polynomial regression outperforms linear baselines  
- Linear models capture global trends but miss nonlinear structure  

---

## 8. Phase 2 Outputs

**Generated Dataset:**  
[phase2_features_baseline_models.csv](https://github.com/MHaynes33/BlackBox/blob/main/data/phase2_features_baseline_models.csv)

This enriched dataset serves as the modeling input for Phase 3.

---

## 9. Phase 2 Summary

Phase 2 successfully:
- engineered features aligned with cost, duration, and efficiency patterns  
- validated engineered features through distribution and outlier checks  
- established baseline regression models  
- demonstrated that nonlinear interactions are required to approximate legacy behavior  
- produced the enriched dataset used in Phase 3  

By the end of Phase 2, the project had a validated feature set, baseline performance benchmarks, and a modeling-ready dataset to support advanced nonlinear and ensemble modeling in Phase 3.


## 5. Key Findings


## 6. Final Notes

~ To get a more detailed overview of the project check our presentation: 

~ For all other questions reach out to any of the members listed above
