"""
One-Click Master Orchestrator for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Workforce Diagnostics & Prescriptive Retention AI System.

Executes the complete end-to-end pipeline:
1. Ingests or generates 15,000 multi-dimensional TCS workforce telemetry records.
2. Trains an ensemble predictive framework (XGBoost, Random Forest, Logistic Baseline).
3. Executes SHAP-based organizational root cause extraction across 7 workplace pillars.
4. Compiles the publication-grade executive PDF report with high-DPI Matplotlib visualizations.
5. Launches the interactive obsidian dark glassmorphic web dashboard.
"""

import os
import sys
import webbrowser
import time

# Ensure local packages are importable
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.data_generator import generate_tcs_attrition_data
from models.attrition_model import TCSAttritionPredictor
from models.root_cause_analyzer import RootCauseAnalyzer
from models.retention_engine import RetentionEngine
from models.simulator import RetentionSimulator
from report.generate_pdf_report import generate_pdf

def main():
    print("=" * 85)
    print(" PROJECT - 05: EMPLOYEE ATTRITION ROOT-CAUSE INTELLIGENCE")
    print(" Tata Consultancy Services (TCS) Workforce Diagnostics & Prescriptive Retention AI")
    print("=" * 85)

    data_dir = os.path.join(base_dir, "data")
    csv_path = os.path.join(data_dir, "tcs_employee_attrition.csv")
    report_path = os.path.join(base_dir, "report", "TCS_Employee_Attrition_Root_Cause_Report.pdf")

    # Step 1: Data Generation / Verification
    print("\n[Step 1/4] Checking and ingesting TCS workforce telemetry dataset...")
    if not os.path.exists(csv_path):
        df = generate_tcs_attrition_data(data_dir, n_samples=15000)
    else:
        import pandas as pd
        df = pd.read_csv(csv_path)
    churn_rate = (df['attrition'].mean() * 100)
    print(f"  ✓ Ingested {len(df):,} workforce records across 7 pillars. Baseline Attrition Rate: {churn_rate:.2f}%.")

    # Step 2: Predictive Ensemble Modeling
    print("\n[Step 2/4] Training XGBoost, Random Forest & Logistic Regression ensemble...")
    predictor = TCSAttritionPredictor(csv_path)
    metrics = predictor.train_models()
    print(f"  ✓ Production Model: XGBoost Classifier")
    print(f"  ✓ Performance Metrics: ROC-AUC={metrics['roc_auc']:.4f} | Accuracy={metrics['accuracy']*100:.1f}% | Recall={metrics['recall']*100:.1f}% | F1={metrics['f1_score']:.4f}")

    # Step 3: Root Cause Diagnostics & Retention Engine
    print("\n[Step 3/4] Extracting SHAP organizational root causes & prescriptive retention ROI...")
    analyzer = RootCauseAnalyzer(predictor, predictor.df)
    retention_engine = RetentionEngine(analyzer, predictor.df)
    top_drivers = analyzer.get_top_global_drivers(3)
    p_roi = retention_engine.calculate_macro_portfolio_roi()
    print(f"  ✓ Top 3 Macro Drivers: {', '.join([d['label'] for d in top_drivers])}")
    print(f"  ✓ High-Risk Talent Cohort: {p_roi['total_at_risk_headcount']:,} employees")
    print(f"  ✓ Addressable Turnover Exposure: ₹{p_roi['total_replacement_exposure_crores']} Cr")
    print(f"  ✓ Net Realized Savings: ₹{p_roi['net_capital_saved_crores']} Cr ({p_roi['portfolio_roi_multiplier']})")

    # Step 4: Publication-Grade PDF Report Compilation
    print("\n[Step 4/4] Compiling publication-grade executive PDF briefing report...")
    generate_pdf(report_path, predictor, analyzer, retention_engine)
    print(f"  ✓ PDF Deliverable generated successfully at: {report_path}")

    # Launch Web Dashboard
    print("\n" + "=" * 85)
    print(" ALL PIPELINES READY: Launching Interactive Executive Web Dashboard")
    print(" Server Endpoint: http://127.0.0.1:5005")
    print("=" * 85)

    from app import app
    port = int(os.environ.get("PORT", 5005))
    app.run(host="127.0.0.1", port=port, debug=False)

if __name__ == "__main__":
    main()

