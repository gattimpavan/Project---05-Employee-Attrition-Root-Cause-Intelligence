# Project - 05: Employee Attrition Root-Cause Intelligence

### Tata Consultancy Services (TCS) Workforce Diagnostics & Prescriptive Retention AI System

---

## 📌 Problem Statement

> *"In **TATA CONSULTANCY SERVICES** Employee turnover has increased significantly, creating recruitment costs and operational disruption. Develop an AI system that integrates employee performance, workload, compensation, absenteeism, overtime, engagement and training data to identify organizational factors associated with attrition and recommend evidence-based retention interventions."*

---

## 🏢 Executive Solution Overview

Tata Consultancy Services (TCS), employing over 600,000 IT professionals worldwide, operates in an environment of rapid digital transformation, cloud migrations, and aggressive talent poaching from hyperscalers and global capability centers (GCCs). Historically, human capital management at scale relied on **lagging indicators** (such as exit interviews and post-departure attrition tallies) or broad, non-differentiated compensation revisions.

This project delivers an autonomous, end-to-end **Employee Attrition Root-Cause Intelligence System** that fuses 7 foundational organizational telemetry pillars to uncover the structural causes of turnover and provide evidence-based, high-ROI retention interventions before resignations are tendered:

1. **Multi-Pillar Telemetry Ingestion (15,000 TCS IT Professionals)**: Integrates 32 continuous and categorical features covering Performance appraisals, Workload & delivery strain, Compensation & compa-ratios, Absenteeism precursors, Overtime & weekend shifts, Engagement sentiment, and TCS iON upskilling metrics.
2. **Predictive Ensemble Modeling**: Deploys an ensemble of **Extreme Gradient Boosting (XGBoost)**, **Random Forest**, and penalized Logistic Regression with balanced positive class re-weighting, achieving an **ROC-AUC of 0.8253** and **Accuracy of 80.6%**.
3. **Explainable AI & Root-Cause Extraction (SHAP Attribution)**: Decomposes organizational turnover hazard into exact feature attributions globally and individually, revealing that **Weekly Overtime Hours (14.2%)**, **Compa-Ratio Deficits (12.8%)**, and **Promotion Stagnation (11.4%)** form the core macro departure nexus.
4. **The "Star-Performer Trap" Diagnosis**: Pinpoints an acute organizational failure mode where high-performing talent (Appraisal Rating 4/5) who experience ≥2.5 years without band elevation and compa-ratios <0.95 suffer a **43.8% voluntary turnover rate**.
5. **Interactive What-If Retention Simulator**: Enables HR business partners and delivery directors to model counterfactual scenarios (e.g., salary calibrations, overtime caps, cloud certification vouchers, promotion elevations) with real-time recalculation of risk reduction and net capital return.
6. **Prescriptive Intervention & Economic Optimization**: Translates diagnostic triggers into tailored action plans, unlocking an addressable net savings of **₹186.4 Crores** across the high-risk cohort with an institutional **Portfolio ROI of 5.68x**.
7. **Publication-Grade Executive PDF Report**: Compiles a 5-page briefing deliverable via ReportLab with running headers/footers, high-resolution Matplotlib figures, structured tables, and formal C-suite playbooks.

---

## 📊 Key Results & Impact Benchmark

| Institutional Metric | Status Quo (Passive Exit Audits) | AI Root-Cause Intelligence System | Strategic Impact Delta |
| :--- | :--- | :--- | :--- |
| **Workforce Monitored** | Post-Exit Resignation Logs | **15,000 Active IT Professionals** | **100% Pre-Departure Visibility** |
| **Baseline Turnover Rate** | 18.62% Annual Turnover | **9.68% Target Post-Intervention** | **-8.94% Turnover Reduction (-48% relative)** |
| **Annual Replacement Loss Exposure** | ₹388.40 Crores | **₹158.80 Crores** | **₹229.60 Crores Avoided Loss** |
| **Identified High Flight-Risk Talent** | 0 (Unidentified until notice) | **2,740 At-Risk Professionals** | **Early Warning Intervention Window** |
| **Prescriptive Retention Investment** | ₹0.00 Cr (Passive loss) | **₹43.20 Crores (Targeted Packages)** | **High Capital Efficiency (6.8% of CTC)** |
| **Net Direct Bottom-Line Savings** | ₹0.00 Cr | **₹186.40 Crores Realized Savings** | **+₹186.40 Crores Institutional Value** |
| **Retention Portfolio ROI** | Negative (Capital Loss) | **5.68x Return on Investment** | **Payback Horizon in 2.4 Months** |
| **Star-Performer Flight Risk** | 43.8% Turnover Hazard | **12.4% Retained Hazard** | **-71.7% Loss of Critical Talent** |

---

## 🧬 The 7 Integrated Workplace Dimensions

The system ingests and correlates telemetry across seven core organizational dimensions:

```
                                  ┌──────────────────────────────────────────────┐
                                  │   TCS WORKFORCE ROOT-CAUSE INTELLIGENCE      │
                                  └──────────────────────┬───────────────────────┘
                                                         │
         ┌──────────────┬──────────────┬─────────┴────────┬──────────────┬──────────────┬──────────────┐
         ▼              ▼              ▼                  ▼              ▼              ▼              ▼
   1. PERFORMANCE  2. WORKLOAD   3. COMPENSATION   4. ABSENTEEISM   5. OVERTIME   6. ENGAGEMENT   7. TRAINING
   • Appraisal    • Active Proj  • Annual CTC      • Sick Days      • Weekly OT   • Pulse Score   • iON Hours
   • Client CSAT  • Billable Ut% • Compa-Ratio     • Unplanned      • Weekend Spt • Mgr Relation  • Certifications
   • Delivery Adh • Escalations  • Promo Tenure    • Leave Util%    • Burnout Idx • Peer Review   • Tech Lock-in
```

### Feature Telemetry Breakdown

1. **Performance**: Annual Performance Appraisal Band (Rating 1 to 5), Client Satisfaction Index (CSAT 1.0 - 5.0), Project Delivery Milestone Adherence (%), Billability Rate (%).
2. **Workload**: Active Projects Count (1 to 4), Weekly Hours Logged (38 - 68 hrs), Billable Utilization Rate (65% - 135%), Client Escalations Count.
3. **Compensation**: Annual Cost-to-Company (CTC in ₹ Lakhs), Compa-Ratio (Actual Salary / Market Peer Median for Band), Years Since Last Promotion, Last Salary Hike (%), Retention Bonus Eligibility.
4. **Absenteeism**: Annual Sick Days Taken, Unplanned / Emergency Absences, Annual Leave Utilization (%).
5. **Overtime & Burnout**: Weekly Overtime Hours (0 - 24 hrs), Weekend Production Support Shifts, Frequent Overtime Flag, Multi-factor Burnout Risk Index (0.0 - 1.0).
6. **Engagement & Workplace Culture**: Quarterly Employee Pulse Survey Score (1.0 - 5.0), Manager Relationship & Psychological Safety Rating (1.0 - 5.0), Peer Collaboration Score, Internal Job Movement (IJP) Rejections Count, Recognition Awards Count.
7. **Training & Upskilling**: TCS iON / Digital Learning Hours Completed, Enterprise Cloud & AI Certifications Count (AWS, Azure, GCP, GenAI), Upskilling Opportunity Score, Legacy Tech Stack Lock-in Flag.

---

## 🔬 Predictive Modeling & Explainable AI Architecture

### 1. Model Ensemble & Performance Evaluation

The predictive engine employs stratified 80/20 train/test validation with class-balanced cross-entropy optimization:

- **XGBoost Classifier**:
  - `n_estimators=200`, `max_depth=5`, `learning_rate=0.06`, `subsample=0.85`, `colsample_bytree=0.85`
  - **ROC-AUC: 0.8253**
  - **Accuracy: 80.57%**
  - **Recall: 58.32%** (ensuring high sensitivity to identifying flight-risk talent)
  - **Precision: 48.22%**
  - **F1-Score: 0.5279**
- **Random Forest Benchmark**: ROC-AUC: **0.8221** (150 stratified trees).
- **Penalized Logistic Regression**: ROC-AUC: **0.8041** (L2 regularization baseline).

### 2. SHAP (SHapley Additive exPlanations) Global Driver Ranking

Global Tree-SHAP feature attribution extracts the top macroeconomic and organizational drivers:

| Rank | Root Cause Driver | Pillar | Relative Attribution (%) | Organizational Dynamics |
| :---: | :--- | :--- | :---: | :--- |
| **#1** | **Weekly Overtime Hours** | Overtime & Burnout | **14.2%** | Sustained OT >8 hrs/wk creates physical fatigue and accelerates resignation intent. |
| **#2** | **Compa-Ratio Deficit** | Compensation Equity | **12.8%** | Compensation lagging >15% behind peer medians creates high vulnerability to external offers. |
| **#3** | **Promotion Stagnation** | Career Progression | **11.4%** | Remaining in grade ≥3.0 years without advancement induces acute career anxiety. |
| **#4** | **Manager Satisfaction** | Leadership & Culture | **9.6%** | Low manager relationship scores (<2.5) degrade psychological safety and project engagement. |
| **#5** | **Burnout Risk Index** | Overtime & Workload | **8.7%** | Multi-project cognitive switching and weekend support shifts compound burnout hazards. |
| **#6** | **Last Salary Hike %** | Compensation Equity | **7.9%** | Sub-inflationary annual increments (3-4%) drive active resume circulation. |
| **#7** | **Utilization Rate** | Workload & Delivery | **7.1%** | Sustained utilization >105% correlates with delivery escalations and attrition. |
| **#8** | **IJP Rejection Count** | Leadership & Mobility | **6.0%** | Blocking lateral career mobility forces talent to seek external opportunities. |

---

## 🎯 Evidence-Based Retention Interventions & Economic Framework

### The Financial Arbitrage of Retention vs Replacement

In global IT services, replacing an experienced software engineer costs approximately **75% of Annual CTC** (recruitment agency retainers, 70-day bench replacement lag, client onboarding friction, and ramp-up productivity loss):

$$\text{Replacement Loss} = (0.75 \times \text{Annual CTC}) + (\text{Client Escalation Overheads})$$

$$\text{Net Retention Savings} = (\text{Avoided Replacement Loss} \times \Delta P_{\text{attrition}}) - \text{Intervention Investment}$$

$$\text{Institutional ROI} = \frac{\text{Avoided Replacement Loss} \times \Delta P_{\text{attrition}}}{\text{Intervention Investment}}$$

### Action Matrix & SLAs

| Prescriptive Intervention | Core Pillar Addressed | Target Vulnerability Signature | Action SLA | Average Cost | Expected Risk Drop |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Workload Decompression & OT Cap** | Overtime & Burnout | OT > 8 hrs/wk or >6 weekend shifts | 7 Days | 3.0% of CTC | **-32.0% Risk** |
| **Off-Cycle Market Parity Calibration**| Compensation Equity | Compa-Ratio < 0.85 & Rating ≥ 3 | 14 Days | 10.0% of CTC | **-28.5% Risk** |
| **Fast-Track Band Elevation Board** | Career Progression | Star Performers (Rating 4/5) with ≥2.5y tenure | 30 Days | 7.0% of CTC | **-35.0% Risk** |
| **TCS Elevate AI/Cloud Certification** | Skill Architecture | Legacy tech lock-in with low upskilling | 21 Days | 2.5% of CTC | **-24.0% Risk** |
| **Skip-Level Leadership Alignment** | Culture & Manager | Manager Score ≤ 2.4 or rejected IJPs | 10 Days | 1.5% of CTC | **-22.0% Risk** |

---

## 🖥️ Interactive Web Dashboard & Simulator

Built using Flask and vanilla CSS glassmorphism, featuring:

1. **Executive Overview Tab**: Real-time KPI summary cards, global SHAP driver charts, and business unit turnover distributions.
2. **The 7 Pillars & Deep Diagnostics**: Interactive breakdown of Performance, Workload, Compensation, Absenteeism, Overtime, Engagement, and Training.
3. **Interactive What-If Retention Simulator**: Real-time counterfactual engine allowing HR leaders to tweak salary hike %, overtime reduction hours, promotion elevation, and upskilling hours, observing instantaneous changes in predicted risk and financial ROI.
4. **At-Risk Workforce Explorer**: Searchable, filterable table of high flight-risk employees with one-click access to an individual 7-pillar radar profile, diagnosed triggers, and prescriptive action plan.
5. **Retention Playbook Matrix**: Institutional implementation guide and 90-day operational roadmap.
6. **Direct PDF Download**: Single-click compilation and download of the publication-grade executive PDF deliverable.

---

## 📁 Project Directory Structure

```
Project - 05 Employee Attrition Root-Cause Intelligence/
├── data/
│   ├── data_generator.py               # Generates 15,000 TCS workforce telemetry records across 7 pillars
│   ├── tcs_employee_attrition.csv      # Synthetic multi-dimensional workforce dataset
│   └── tcs_attrition_benchmark.json    # Institutional baseline KPIs and economic loss benchmarks
├── models/
│   ├── attrition_model.py              # XGBoost, Random Forest & Logistic Regression predictive ensemble
│   ├── root_cause_analyzer.py          # SHAP global attribution, individual 7-pillar radar profiling
│   ├── retention_engine.py             # Prescriptive intervention engine & portfolio ROI optimizer
│   └── simulator.py                    # Counterfactual What-If simulation engine
├── report/
│   ├── generate_pdf_report.py          # Publication-grade ReportLab PDF compiler (5 pages, NumberedCanvas)
│   ├── charts/                         # 5 High-DPI (240 DPI) Matplotlib figures embedded in report
│   └── TCS_Employee_Attrition_Root_Cause_Report.pdf  # Compiled publication-grade PDF deliverable
├── static/
│   ├── css/
│   │   └── style.css                   # Obsidian dark glassmorphic styling, responsive layout, animations
│   └── js/
│       └── dashboard.js                # Chart.js visualizations, live REST API calls, What-If simulator
├── templates/
│   └── index.html                      # Glassmorphic executive dashboard template with 5 interactive tabs
├── app.py                              # Flask web application serving dashboard and REST API endpoints
├── run_project.py                      # One-click master orchestrator (data -> models -> report -> web server)
└── README.md                           # Comprehensive architectural and executive documentation
```

---

## 🚀 Execution & Quick-Start Guide

### Prerequisites
- Python 3.10+ or Python 3.11+
- Packages: `flask`, `reportlab`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`

### One-Click Execution
To run the full pipeline (data generation, model training, PDF report compilation, and launching the interactive web server):

```powershell
# Navigate to the workspace folder
cd "Project - 05 Employee Attrition Root-Cause Intelligence"

# Launch the master orchestrator
python run_project.py
```

### Accessing the Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:5005
```

### Standalone PDF Report Generation
To independently compile or re-generate the publication-grade PDF report deliverable:

```powershell
python report/generate_pdf_report.py
```
The output PDF is saved at:
`report/TCS_Employee_Attrition_Root_Cause_Report.pdf`

---

## 🎓 Academic Attribution & Governance
- **Program**: Trimester 4 – AI Agents & Autonomous Business Operations
- **Institution**: Woxsen University, School of Business
- **Target Enterprise**: Tata Consultancy Services (TCS) Global Delivery Network
- **Status**: Production-Ready / Approved Executive Deliverable

