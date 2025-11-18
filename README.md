

# ACME Corp – Legacy Reimbursement System

## I. Team Members and Roles

-   Ayushi Bohra : Technical Lead / ML Engineer

    \~ Leads feature engineering, modeling approach, and code structure. Ensures model reproducibility and pipeline design.

-   Mike Haynes : Business Analyst

    \~ Interprets interview insights and PRD context, aligns findings to business logic, writes narrative justification for modeling decisions.

-   Colyn Martin: Documentation & Communication Lead

    \~ Owns formatting of reports, slide design, final written outputs, and ensures clarity of deliverables.

-   Matthew Fernald: QA / Testing / Data Wrangler

    \~ Validates dataset integrity, performs spot-checks on model outputs, tests edge cases, and verifies correctness before submission.

## II. Project Overview

This project aim to replicate ACME Corporation’s legacy reimbursement logic by using machine learning techniques to uncover hidden business logic patterns. By analyzing historical reimbursement data and reverse-engineering the system’s behavior, our team seeks to build predictive models that both accurately match the legacy system’s outputs and provide clear and easy to understand insights into how those decisions are made.

## III. Repository Structure (working on this)

### Branches (are we deleting the other branches)

Main

BABranch

CVM-Documentation

FernMt-patch-1

### Main Branch Structure and Details (Update as we go!!)

## Repository Structure (in case we delete the other branches)

The repository is organized into directories for notebooks, datasets, Quarto reports, presentation materials, and source code. The structure is:

``` text
.
├── .idea/                            # ????
│   ├── inspectionProfiles/
│   ├── profiles_settings.xml
│   ├── BlackBox.iml
│   ├── misc.xml
│   ├── modules.xml
│   └── vcs.xml
│
├── Notebooks/                        # Jupyter notebooks for all analytical work
│   ├── 01_EDA_Reimbursement.ipynb
│   ├── 02_Feature_Engineering_and_Baseline_Model.ipynb
│   ├── Feature_Correlation_and_Visualization.ipynb
│   ├── week1_data_cleaning.ipynb
│   └── .gitkeep
│
├── data/                             # Provided datasets (public + private) and dataset we worked witha long the way?
│   ├── public_cases.json
│   ├── private_cases.json
│   └── .gitkeep
│
├── presentation/                     # Presentation materials and location of the final report
│   └── .gitkeep
│
├── reports/                          # Quarto documentation (QMD + rendered HTML)
│   ├── Business_logic_summary.qmd
│   ├── EDA_with_Baseline_Model.qmd
│   ├── EDA_with_Baseline_Model.html
│   ├── Feature_Definitions_and_Rationale.qmd
│   ├── Feature_Definitions_and_Rationale.html
│   ├── Phase1_summary.qmd
│   └── .gitkeep
│
├── src/                              # Source code, scripts, supporting docs (gotta work on this)
│   ├── INTERVIEWS.md                 # Requirement-gathering or interview notes
│   ├── PRD.md                        # Product Requirements Document
│   ├── README.md                     # Source-level documentation
│   ├── eval.sh                       # Evaluation runner script
│   ├── generate_results.sh           # Automated results generation script
│   ├── run.sh.template               # Template for execution pipeline
│   └── .gitignore
│
├── LICENSE                           # Project license
└── README.md                         # Main repository documentation/guide
```

## IV. How to Run the Necessary Code for Replication

This section explains how to access the Jupyter notebooks, install dependencies, and execute the code required to reproduce all analyses, figures, models, and results for the ACME reimbursement prediction project.

### 1. Accessing the Notebooks

All code used in this project is contained in Jupyter notebooks stored on the `main` branch of the GitHub repository.

1.  Navigate to the repository on GitHub.\
2.  Make sure you are on the **`main`** branch.\
3.  Open the **`Notebooks`** directory.

### 2. Contents of the `Notebooks` directory

### 3. Guide to Understanding and Recreating Jupyter Notebook Contents

## V. Data Description

### What we were given to work with?

`public_case.json`

Access here: https://github.com/MHaynes33/BlackBox/blob/main/data/public_cases.json

`private_cases.json`

Access here: https://github.com/MHaynes33/BlackBox/blob/main/data/private_cases.json

### Dataset compiled during our process?

## VI. Project Approach

### Phase 1

**What Happened During Phase 1**

During Phase 1, our objective was to understand the behavior of ACME Corp’s legacy reimbursement system using the provided 1,000 historical reimbursement examples (`public_cases.json`) and stakeholder interview narratives.

The goal at this stage was not to redesign or improve the system, but to form a clear behavioral understanding that we will later replicate using machine learning.

**Data Reviewed During Phase 1**

The dataset that we reviewed amounted to 6,000 records (1,000 public with reimbursement labels and 5,000 unlabeled private cases). We examined the historical reimbursement dataset, which includes:

-   **Trip Duration (days)**
-   **Total Mileage**
-   **Total Receipts**
-   **Reimbursement Amount** (system output)

A data cleaning notebook was created to: - Remove formatting inconsistencies - Validate numeric ranges - Prepare features for further modeling

*Result:* A clean, analysis-ready dataset.

**What patterns were observed during Phase 1?**

| Factor | Observation | Interpretation |
|------------------------|------------------------|------------------------|
| **Trip Length** | Reimbursement is **more generous** around **4–6 days**, declines for \>7 days trips | Suggests a **sweet spot and long-trip penalty** |
| **Mileage** | Value-per-mile decreases after \~100 miles/day | Indicates a **non-linear mileage adjustment curve** |
| **Receipts** | Higher receipts do **not** consistently produce higher reimbursement | Suggests **diminishing returns and upper/lower spend penalties** |
| **Non-linear Behavior** | Adjustments change together, not independently | Legacy system likely uses **multiple interacting rules** |
| **Rounding Artifacts** | Some reimbursements show small irregularities | Suggests **bugs/features that must be preserved** |

**Phase 1 Outcomes**

-   We confirmed that the legacy system is **not linear**.

-   We identified meaningful candidate **derived features** for modeling:

    -   `miles_per_day`
    -   `spend_per_day`
    -   `log_miles`
    -   `efficiency` indicators (balanced trip behavior)

-   We created a consolidated **Business Logic Summary** aligning interview anecdotes with statistical evidence.

Access here: https://github.com/MHaynes33/BlackBox/blob/main/reports/Business_logic_summary.qmd

-   We also created an EDA...

Access here: https://github.com/MHaynes33/BlackBox/blob/main/reports/EDA%20with%20Baseline%20Model.html

## work on this!!

### Phase 2 (waiting on phase 2 summary)

### Phase 3

### Phase 4

## VII. Results Summary

## VIII. Presentation Link/Location:
