**ACME Reimbursement Engine – Reverse Engineering Project** (Work in Progress)

Machine Learning Project | ECU Computer Science

Using statistical analysis, business logic synthesis, and machine learning, we reverse-engineered ACME Corporation’s 60-year-old legacy reimbursement engine — a system whose internal logic is unknown but still used daily.

Our goal: Replicate the reimbursement behavior with high accuracy and interpretability, using only historical data and stakeholder interviews.

-Project Overview--

ACME Corporation maintains a legacy reimbursement engine built decades ago. No documentation exists describing the true business logic, but the company relies on its outputs to compensate employees for travel. A new modernized system is in development — however, discrepancies between the legacy and modern outputs have raised concerns.

Our team was tasked with:

Investigating how the legacy system computes reimbursement

Identifying hidden rules, nonlinear behaviors, and business constraints

Engineering features to approximate the system’s logic

Building and evaluating baseline predictive models

Preparing for advanced nonlinear modeling (Phase 3)

This project simulates a full ML consulting engagement, integrating data science, domain reasoning, model development, and communication.

👥 Team Members & Roles

Ayushi Bohra – Technical Lead / ML Engineer
Colyn Martin – Documentation & Communication Lead
Mike Haynes – Business Analyst
Matthew Fernald – Quality Analyst / Tester / Data Wrangler

.
├── .idea/
│   ├── inspectionProfiles/
│   │   └── profiles_settings.xml
│   ├── .gitignore
│   ├── BlackBox.iml
│   ├── misc.xml
│   ├── modules.xml
│   └── vcs.xml
│
├── Notebooks/
│   ├── .gitkeep
│   ├── 01_EDA_Reimbursement (3).ipynb
│   ├── 02_Feature_Engineering_and_Baseline_Model.ipynb
│   ├── Feature Correlation and Visualization.ipynb
│   ├── Model Development & Integration.ipynb
│   ├── Model Evaluation Checklist.ipynb
│   ├── Model Interpretability & Feature Impact - Phase 4.ipynb
│   ├── Performance Summary.ipynb
│   ├── Phase3_Performance.ipynb
│   ├── _Ensemble Learning (2).ipynb
│   └── week1_data_cleaning.ipynb
│
├── data/
│   ├── .gitkeep
│   ├── TeamProjectInstructions.qmd
│   ├── combined_clean.csv
│   ├── phase2_features_baseline_models.csv
│   ├── private_cases.json
│   ├── private_clean.csv
│   ├── public_cases.json
│   └── public_clean.csv
│
├── presentation/
│   ├── .gitkeep
│   ├── Phase 1 Presentation Markdown html and QMD.html
│   ├── Phase 1 Presentation Markdown html and QMD.qmd
│   ├── Phase 2 Presentation Markdown html and QMD v2.html
│   ├── Phase 2 Presentation Markdown html and QMD v2.qmd
│   └── phase1-2-of-presentation-pdf-.pdf  :contentReference[oaicite:0]{index=0}
│
├── reports/
│   ├── .gitkeep
│   ├── Business_logic_summary.html
│   ├── Business_logic_summary.qmd
│   ├── EDA & Baseline Model Info.html
│   ├── EDA & Baseline Model Info.qmd
│   ├── Feature Definitions and Rationales Table Updated .html
│   ├── Feature Definitions and Rationales Table Updated .qmd
│   ├── Phase1_summary.qmd
│   ├── project_Summary.pdf
│   ├── project_Summary.qmd
│   ├── typst-show.typ
│   └── typst-template.typ
│
├── src/
│   ├── .gitkeep
│   ├── INTERVIEWS.md
│   ├── PRD.md
│   ├── README.md
│   ├── eval.sh
│   ├── final_model.pkl
│   ├── generate_results.sh
│   ├── predict.py
│   └── run.sh.template
│
├── .gitignore
├── LICENSE
├── README.md
└── deleteme.txt
Phase 1
Phase 2
Phase 4
Phase 4
How to run the model

Key Resources
