"""
Interactive What-If Simulation Engine for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Retention Scenario Modeling.

Allows HR business partners and delivery managers to simulate tactical workplace interventions
(e.g., salary adjustments, overtime restrictions, promotion elevation, training sponsorship)
and observe instantaneous changes in predicted attrition hazard and institutional ROI.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any

class RetentionSimulator:
    """Simulates sensitivity of employee attrition hazard to managerial and organizational interventions."""

    def __init__(self, predictor, root_cause_analyzer):
        self.predictor = predictor
        self.analyzer = root_cause_analyzer

    def simulate_employee_intervention(
        self,
        emp_id: str,
        salary_hike_delta_pct: float = 0.0,
        overtime_reduction_hours: float = 0.0,
        grant_promotion: bool = False,
        add_training_hours: float = 0.0,
        improve_manager_score_to: float = None
    ) -> Dict[str, Any]:
        """Evaluates how specific counterfactual modifications alter an employee's turnover hazard."""
        row_matches = self.analyzer.df[self.analyzer.df["emp_id"] == emp_id]
        if row_matches.empty:
            return {"error": f"Employee {emp_id} not found."}

        baseline_record = row_matches.iloc[0].to_dict()
        sim_record = baseline_record.copy()

        # 1. Baseline Risk
        base_prob, base_cat = self.predictor.predict_employee(baseline_record)

        # 2. Apply Counterfactual Interventions
        # Compensation adjustment
        if salary_hike_delta_pct > 0.0:
            sim_record["last_salary_hike_pct"] = min(25.0, sim_record["last_salary_hike_pct"] + salary_hike_delta_pct)
            # Higher salary directly improves annual CTC and compa-ratio
            salary_multiplier = 1.0 + (salary_hike_delta_pct / 100.0)
            sim_record["annual_ctc_lakhs"] = round(sim_record["annual_ctc_lakhs"] * salary_multiplier, 2)
            sim_record["compa_ratio"] = round(sim_record["compa_ratio"] * salary_multiplier, 3)

        # Overtime & Burnout reduction
        if overtime_reduction_hours > 0.0:
            sim_record["weekly_overtime_hours"] = max(0.0, sim_record["weekly_overtime_hours"] - overtime_reduction_hours)
            sim_record["weekly_hours_logged"] = max(40.0, sim_record["weekly_hours_logged"] - overtime_reduction_hours)
            sim_record["utilization_rate"] = max(75.0, sim_record["utilization_rate"] - (overtime_reduction_hours / 45.0 * 100))
            sim_record["weekend_support_shifts"] = max(0, int(sim_record["weekend_support_shifts"] * 0.4))
            sim_record["frequent_overtime"] = int(sim_record["weekly_overtime_hours"] >= 7.5)
            # Recompute burnout score
            sim_record["burnout_risk_score"] = round(
                max(0.0, sim_record["burnout_risk_score"] - (overtime_reduction_hours / 24.0 * 0.45)), 3
            )

        # Promotion reset
        if grant_promotion:
            sim_record["years_since_last_promotion"] = 0.1
            # Promotions also elevate base CTC and sentiment
            sim_record["compa_ratio"] = round(max(sim_record["compa_ratio"], 1.02), 3)
            sim_record["pulse_survey_score"] = min(5.0, sim_record["pulse_survey_score"] + 0.8)

        # Training & Upskilling
        if add_training_hours > 0.0:
            sim_record["training_hours_completed"] += add_training_hours
            sim_record["certifications_count"] += int(add_training_hours // 25.0)
            sim_record["upskilling_opportunity_score"] = min(5.0, sim_record["upskilling_opportunity_score"] + 0.7)
            sim_record["legacy_tech_lockin"] = 0 # Freed from legacy trap

        # Managerial Support
        if improve_manager_score_to is not None:
            sim_record["manager_satisfaction_score"] = max(sim_record["manager_satisfaction_score"], improve_manager_score_to)
            sim_record["pulse_survey_score"] = min(5.0, sim_record["pulse_survey_score"] + 0.6)

        # 3. Predict Simulated Risk
        sim_prob, sim_cat = self.predictor.predict_employee(sim_record)

        # 4. Financial Cost & Savings Evaluation
        # Incremental cost of interventions
        intervention_cost_lakhs = 0.0
        if salary_hike_delta_pct > 0.0:
            intervention_cost_lakhs += round(baseline_record["annual_ctc_lakhs"] * (salary_hike_delta_pct / 100.0), 2)
        if overtime_reduction_hours > 0.0:
            intervention_cost_lakhs += 0.45 # Resource rebalancing buffer
        if grant_promotion:
            intervention_cost_lakhs += round(baseline_record["annual_ctc_lakhs"] * 0.10, 2)
        if add_training_hours > 0.0:
            intervention_cost_lakhs += round(0.25 + (add_training_hours / 100.0) * 0.20, 2)
        if improve_manager_score_to is not None:
            intervention_cost_lakhs += 0.15 # Leadership coaching session

        replacement_cost = baseline_record["replacement_cost_lakhs"]
        risk_reduction_abs = round(base_prob - sim_prob, 4)
        gross_savings = round(replacement_cost * max(0.0, risk_reduction_abs), 2)
        net_savings = round(gross_savings - intervention_cost_lakhs, 2)
        roi_ratio = round(gross_savings / max(0.05, intervention_cost_lakhs), 1)

        return {
            "emp_id": emp_id,
            "employee_name": baseline_record["name"],
            "baseline": {
                "probability": base_prob,
                "risk_category": base_cat,
                "weekly_overtime_hours": baseline_record["weekly_overtime_hours"],
                "compa_ratio": baseline_record["compa_ratio"],
                "years_since_last_promotion": baseline_record["years_since_last_promotion"],
                "training_hours_completed": baseline_record["training_hours_completed"]
            },
            "simulated": {
                "probability": sim_prob,
                "risk_category": sim_cat,
                "weekly_overtime_hours": sim_record["weekly_overtime_hours"],
                "compa_ratio": sim_record["compa_ratio"],
                "years_since_last_promotion": sim_record["years_since_last_promotion"],
                "training_hours_completed": sim_record["training_hours_completed"]
            },
            "impact_delta": {
                "risk_reduction_absolute_pct": round(risk_reduction_abs * 100, 1),
                "relative_risk_drop_pct": round((risk_reduction_abs / max(0.01, base_prob)) * 100, 1),
                "category_shift": f"{base_cat} -> {sim_cat}"
            },
            "economics": {
                "intervention_cost_lakhs": round(intervention_cost_lakhs, 2),
                "replacement_cost_avoided_lakhs": gross_savings,
                "net_realized_savings_lakhs": net_savings,
                "simulated_roi_multiplier": f"{roi_ratio}x"
            }
        }

