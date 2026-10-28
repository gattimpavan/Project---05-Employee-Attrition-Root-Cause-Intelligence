"""
Flask Web Application for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Workforce Analytics & Prescriptive Retention AI.

Serves the interactive executive dashboard, REST API endpoints for real-time diagnostics,
counterfactual What-If retention simulations, employee risk profiling, and PDF report delivery.
"""

import os
import sys
import json
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file

# Add project root to path
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from data.data_generator import generate_tcs_attrition_data
from models.attrition_model import TCSAttritionPredictor
from models.root_cause_analyzer import RootCauseAnalyzer
from models.retention_engine import RetentionEngine
from models.simulator import RetentionSimulator

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize Paths
DATA_DIR = os.path.join(base_dir, "data")
CSV_PATH = os.path.join(DATA_DIR, "tcs_employee_attrition.csv")
REPORT_PATH = os.path.join(base_dir, "report", "TCS_Employee_Attrition_Root_Cause_Report.pdf")

# Ensure dataset exists
if not os.path.exists(CSV_PATH):
    print("Generating TCS workforce telemetry dataset (15,000 records)...")
    df_raw = generate_tcs_attrition_data(DATA_DIR, n_samples=15000)
else:
    df_raw = pd.read_csv(CSV_PATH)

print("Training TCS Attrition Predictive Ensemble & Building Preprocessors...")
predictor = TCSAttritionPredictor(CSV_PATH)
metrics = predictor.train_models()

print("Initializing Root-Cause Intelligence, Prescriptive Engine, and Simulator...")
analyzer = RootCauseAnalyzer(predictor, predictor.df)
retention_engine = RetentionEngine(analyzer, predictor.df)
simulator = RetentionSimulator(predictor, analyzer)
print("TCS Attrition Intelligence System loaded successfully!")

@app.route("/")
def index():
    """Serves the main interactive TCS Attrition Root-Cause Intelligence Dashboard."""
    return render_template("index.html")

@app.route("/api/summary")
def api_summary():
    """Returns high-level workforce metrics, model accuracy, and macro attrition statistics."""
    portfolio_roi = retention_engine.calculate_macro_portfolio_roi()
    total_records = len(predictor.df)
    total_churn = int(predictor.df["attrition"].sum())
    churn_rate = round(float(predictor.df["attrition"].mean() * 100), 2)
    total_loss_cr = round(float(predictor.df[predictor.df["attrition"] == 1]["replacement_cost_lakhs"].sum() / 100.0), 2)

    return jsonify({
        "status": "success",
        "organization": "Tata Consultancy Services (TCS)",
        "project_title": "Project - 05: Employee Attrition Root-Cause Intelligence",
        "total_headcount": total_records,
        "observed_attrition_count": total_churn,
        "attrition_rate_pct": churn_rate,
        "total_replacement_loss_cr": total_loss_cr,
        "model_performance": {
            "roc_auc": metrics["roc_auc"],
            "accuracy_pct": round(metrics["accuracy"] * 100, 1),
            "precision_pct": round(metrics["precision"] * 100, 1),
            "recall_pct": round(metrics["recall"] * 100, 1),
            "f1_score": metrics["f1_score"]
        },
        "portfolio_roi": portfolio_roi
    })

@app.route("/api/root_causes")
def api_root_causes():
    """Returns global feature importances, top macro drivers, and pillar definitions."""
    top_drivers = analyzer.get_top_global_drivers(top_n=10)
    return jsonify({
        "status": "success",
        "top_drivers": top_drivers,
        "pillars": analyzer.PILLAR_DEFINITIONS
    })

@app.route("/api/department_analysis")
def api_department_analysis():
    """Returns aggregated root-cause and attrition metrics by Business Unit and Band."""
    bu_data = analyzer.get_department_root_cause_analysis()
    band_data = analyzer.get_band_vulnerability_analysis()
    return jsonify({
        "status": "success",
        "business_units": bu_data,
        "designation_bands": band_data
    })

@app.route("/api/employees_at_risk")
def api_employees_at_risk():
    """Returns paginated or filtered list of high flight-risk employees for targeting."""
    bu_filter = request.args.get("bu", "All")
    band_filter = request.args.get("band", "All")
    limit = int(request.args.get("limit", 50))

    df_filtered = predictor.df.copy()
    if bu_filter != "All":
        df_filtered = df_filtered[df_filtered["business_unit"] == bu_filter]
    if band_filter != "All":
        df_filtered = df_filtered[df_filtered["designation_band"] == band_filter]

    # Focus on at-risk employees
    df_sorted = df_filtered.sort_values(by="model_attrition_risk", ascending=False).head(limit)

    records = []
    for _, row in df_sorted.iterrows():
        records.append({
            "emp_id": row["emp_id"],
            "name": row["name"],
            "business_unit": row["business_unit"],
            "designation_band": row["designation_band"],
            "location": row["location"],
            "annual_ctc_lakhs": row["annual_ctc_lakhs"],
            "compa_ratio": row["compa_ratio"],
            "weekly_overtime_hours": row["weekly_overtime_hours"],
            "years_since_last_promotion": row["years_since_last_promotion"],
            "performance_rating": int(row["performance_rating"]),
            "attrition_risk_prob": float(row["model_attrition_risk"]),
            "risk_category": str(row["risk_category"]),
            "replacement_cost_lakhs": float(row["replacement_cost_lakhs"])
        })

    return jsonify({
        "status": "success",
        "total_matching": len(records),
        "employees": records
    })

@app.route("/api/employee/<emp_id>")
def api_employee_detail(emp_id):
    """Provides individual diagnosis, 7-pillar radar breakdown, and prescriptive retention plan."""
    diag = analyzer.diagnose_employee(emp_id)
    if "error" in diag:
        return jsonify({"status": "error", "message": diag["error"]}), 404

    prescription = retention_engine.prescribe_for_employee(emp_id)
    return jsonify({
        "status": "success",
        "diagnosis": diag,
        "prescription": prescription
    })

@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """Performs real-time What-If counterfactual retention simulation on an employee."""
    data = request.get_json() or {}
    emp_id = data.get("emp_id")
    salary_hike_delta = float(data.get("salary_hike_delta_pct", 0.0))
    overtime_reduction = float(data.get("overtime_reduction_hours", 0.0))
    grant_promotion = bool(data.get("grant_promotion", False))
    add_training = float(data.get("add_training_hours", 0.0))
    manager_score = float(data.get("improve_manager_score_to", 4.2)) if data.get("improve_manager", False) else None

    if not emp_id:
        # Default to highest risk employee
        emp_id = predictor.df.sort_values(by="model_attrition_risk", ascending=False).iloc[0]["emp_id"]

    sim_res = simulator.simulate_employee_intervention(
        emp_id=emp_id,
        salary_hike_delta_pct=salary_hike_delta,
        overtime_reduction_hours=overtime_reduction,
        grant_promotion=grant_promotion,
        add_training_hours=add_training,
        improve_manager_score_to=manager_score
    )

    if "error" in sim_res:
        return jsonify({"status": "error", "message": sim_res["error"]}), 400

    return jsonify({
        "status": "success",
        "simulation": sim_res
    })

@app.route("/download_report")
def download_report():
    """Downloads the generated publication-grade PDF report."""
    if not os.path.exists(REPORT_PATH):
        # Trigger generation if not already built
        print("Report not found on disk. Compiling PDF report...")
        from report.generate_pdf_report import generate_pdf
        generate_pdf(REPORT_PATH, predictor, analyzer, retention_engine)

    return send_file(
        REPORT_PATH,
        as_attachment=True,
        download_name="TCS_Employee_Attrition_Root_Cause_Report.pdf",
        mimetype="application/pdf"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    print(f"Launching TCS Attrition Intelligence Web Server on http://127.0.0.1:{port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

