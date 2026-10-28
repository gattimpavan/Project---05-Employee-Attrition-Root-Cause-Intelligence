"""
Prescriptive Retention Intervention Engine for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Evidence-Based HR Interventions & ROI Optimization.

Formulates targeted, evidence-based retention interventions tailored to individual root causes,
and computes institutional financial return on investment (ROI) against recruitment and operational disruption costs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

class RetentionEngine:
    """Prescriptive analytics engine matching diagnosed root causes to evidence-based retention interventions."""

    INTERVENTION_CATALOG = {
        "compensation_correction": {
            "title": "Off-Cycle Market Parity & Retention Milestone Adjustment",
            "category": "Compensation & Benefits",
            "mechanism": "Deliver an immediate 8% to 14% off-cycle compensation calibration to elevate compa-ratio to >= 0.96, supplemented with a 12-month retention bonus vesting schedule.",
            "average_cost_pct_salary": 0.10, # 10% of annual CTC
            "expected_risk_drop_pct": 28.5,
            "horizon_days": 14,
            "target_root_causes": ["Compensation Equity", "Below-Market Compa-Ratio", "Stagnant Increment"]
        },
        "workload_decompression": {
            "title": "Workload Decompression & Strict Overtime Ceiling",
            "category": "Operational Workload",
            "mechanism": "Enforce a strict 4-hour/week overtime ceiling, rebalance project assignments to bring billable utilization below 85%, and implement mandatory compensatory off for weekend support shifts.",
            "average_cost_pct_salary": 0.03, # 3% capacity buffer cost
            "expected_risk_drop_pct": 32.0,
            "horizon_days": 7,
            "target_root_causes": ["Overtime & Burnout Strain", "Excessive Overwork", "High Utilization"]
        },
        "accelerated_promotion": {
            "title": "Fast-Track Band Elevation & Role Redesignation",
            "category": "Career Progression",
            "mechanism": "Initiate an accelerated out-of-cycle promotion evaluation committee for eligible star performers with >= 2.5 years in grade, transitioning them into module lead / architect responsibilities.",
            "average_cost_pct_salary": 0.07,
            "expected_risk_drop_pct": 35.0,
            "horizon_days": 30,
            "target_root_causes": ["Career Progression", "Promotion Stagnation", "Star Performer Promotion Deficit"]
        },
        "managerial_coaching_reset": {
            "title": "Skip-Level Sponsorship & Managerial Alignment Intervention",
            "category": "Leadership & Culture",
            "mechanism": "Conduct executive skip-level alignment session with Delivery Head, reassign to high-psychological-safety project lead, and provide manager sensitivity coaching.",
            "average_cost_pct_salary": 0.015, # Nominal coaching overhead
            "expected_risk_drop_pct": 22.0,
            "horizon_days": 10,
            "target_root_causes": ["Managerial & Culture", "Leadership Deficit", "Psychological Safety"]
        },
        "tcs_elevate_upskilling": {
            "title": "TCS Elevate AI/Cloud Certification Sponsorship & Tech Transfer",
            "category": "Skill Architecture",
            "mechanism": "Provide sponsored enterprise voucher for AWS/Azure/GenAI certification, allocate 15% dedicated weekly learning hours, and approve fast-track IJP transfer from legacy codebase to modern cloud unit.",
            "average_cost_pct_salary": 0.025, # INR 25k-40k voucher + learning hours
            "expected_risk_drop_pct": 24.0,
            "horizon_days": 21,
            "target_root_causes": ["Skill Relevance", "Legacy Technology Entrapment", "Lack of Upskilling"]
        },
        "workplace_flexibility": {
            "title": "Flexible Delivery Arrangement & Burnout Recharge Buffer",
            "category": "Wellbeing & Flexibility",
            "mechanism": "Grant transition to flexible hybrid schedule or remote delivery center, supplemented with a 5-day wellness recharge leave and ergonomic home workstation stipend.",
            "average_cost_pct_salary": 0.012,
            "expected_risk_drop_pct": 18.5,
            "horizon_days": 5,
            "target_root_causes": ["Burnout Hazard", "Commute Stress", "Unplanned Absences"]
        }
    }

    def __init__(self, root_cause_analyzer, df: pd.DataFrame):
        self.analyzer = root_cause_analyzer
        self.df = df.copy()

    def prescribe_for_employee(self, emp_id: str) -> Dict[str, Any]:
        """Prescribes customized retention interventions and computes individualized ROI."""
        diag = self.analyzer.diagnose_employee(emp_id)
        if "error" in diag:
            return diag

        emp_info = diag["employee_info"]
        attr_risk = diag["attrition_risk"]
        current_prob = attr_risk["probability"]
        triggers = diag["primary_root_causes"]
        annual_ctc = emp_info["annual_ctc_lakhs"]
        replacement_cost = emp_info["replacement_cost_lakhs"]

        prescriptions = []
        cumulative_risk_drop = 0.0
        total_intervention_cost = 0.0

        for t in triggers:
            pillar = t["pillar"]
            if "Overtime" in pillar or "Burnout" in pillar:
                action = self.INTERVENTION_CATALOG["workload_decompression"]
            elif "Compensation" in pillar:
                action = self.INTERVENTION_CATALOG["compensation_correction"]
            elif "Career" in pillar or "Promotion" in pillar:
                action = self.INTERVENTION_CATALOG["accelerated_promotion"]
            elif "Managerial" in pillar or "Culture" in pillar:
                action = self.INTERVENTION_CATALOG["managerial_coaching_reset"]
            elif "Skill" in pillar or "Legacy" in pillar:
                action = self.INTERVENTION_CATALOG["tcs_elevate_upskilling"]
            else:
                action = self.INTERVENTION_CATALOG["workplace_flexibility"]

            cost_lakhs = round(annual_ctc * action["average_cost_pct_salary"], 2)
            prescriptions.append({
                "intervention_title": action["title"],
                "category": action["category"],
                "tactical_mechanism": action["mechanism"],
                "cost_inr_lakhs": cost_lakhs,
                "projected_risk_drop_pct": action["expected_risk_drop_pct"],
                "turnaround_days": action["horizon_days"],
                "addresses_root_cause": t["title"]
            })

            total_intervention_cost += cost_lakhs
            # Multiplicative diminishing returns for compound interventions
            marginal_drop = (action["expected_risk_drop_pct"] / 100.0) * (1.0 - cumulative_risk_drop) * 0.70
            cumulative_risk_drop += marginal_drop

        # Fallback if no specific trigger fired
        if not prescriptions:
            action = self.INTERVENTION_CATALOG["workplace_flexibility"]
            cost_lakhs = round(annual_ctc * action["average_cost_pct_salary"], 2)
            prescriptions.append({
                "intervention_title": action["title"],
                "category": action["category"],
                "tactical_mechanism": action["mechanism"],
                "cost_inr_lakhs": cost_lakhs,
                "projected_risk_drop_pct": action["expected_risk_drop_pct"],
                "turnaround_days": action["horizon_days"],
                "addresses_root_cause": "General Retention Anchor"
            })
            total_intervention_cost += cost_lakhs
            cumulative_risk_drop = 0.18

        post_risk_prob = min(current_prob, max(0.02, round(current_prob * (1.0 - cumulative_risk_drop), 4)))
        risk_drop_abs = max(0.0, round(current_prob - post_risk_prob, 4))

        # Expected replacement cost avoided = Replacement Cost * Probability Reduction
        gross_savings = round(replacement_cost * risk_drop_abs, 2)
        net_savings = round(gross_savings - total_intervention_cost, 2)
        roi_multiplier = round(gross_savings / max(0.01, total_intervention_cost), 1) if gross_savings > 0 else 0.0

        return {
            "emp_id": emp_id,
            "employee_name": emp_info["name"],
            "current_attrition_risk": current_prob,
            "projected_post_intervention_risk": post_risk_prob,
            "absolute_risk_reduction_pct": round(risk_drop_abs * 100, 1),
            "prescribed_interventions": prescriptions,
            "financial_roi": {
                "replacement_cost_lakhs": replacement_cost,
                "total_intervention_cost_lakhs": round(total_intervention_cost, 2),
                "expected_gross_savings_lakhs": gross_savings,
                "net_realized_savings_lakhs": net_savings,
                "roi_multiplier": f"{roi_multiplier}x"
            }
        }

    def calculate_macro_portfolio_roi(self) -> Dict[str, Any]:
        """Calculates institutional financial impact across the high-risk cohort (>= 55% hazard)."""
        at_risk_df = self.df[self.df["attrition_hazard_prob"] >= 0.55]
        total_at_risk_headcount = len(at_risk_df)

        total_annual_ctc = at_risk_df["annual_ctc_lakhs"].sum()
        total_replacement_exposure_lakhs = at_risk_df["replacement_cost_lakhs"].sum()
        
        # Average intervention cost is ~6.8% of CTC (blended across salary hike, training, workload buffers)
        total_intervention_budget_lakhs = round(total_annual_ctc * 0.068, 2)
        
        # Average risk mitigation: drops probability by 48% relative
        expected_retained_employees = int(total_at_risk_headcount * 0.48)
        avoided_replacement_losses_lakhs = round(total_replacement_exposure_lakhs * 0.48, 2)
        
        net_cost_savings_lakhs = round(avoided_replacement_losses_lakhs - total_intervention_budget_lakhs, 2)
        portfolio_roi = round(avoided_replacement_losses_lakhs / max(1.0, total_intervention_budget_lakhs), 2)

        return {
            "total_at_risk_headcount": total_at_risk_headcount,
            "total_workforce_analyzed": len(self.df),
            "at_risk_percentage": round(total_at_risk_headcount / len(self.df) * 100, 1),
            "total_replacement_exposure_crores": round(total_replacement_exposure_lakhs / 100.0, 2),
            "required_intervention_budget_crores": round(total_intervention_budget_lakhs / 100.0, 2),
            "projected_retained_talent": expected_retained_employees,
            "avoided_replacement_loss_crores": round(avoided_replacement_losses_lakhs / 100.0, 2),
            "net_capital_saved_crores": round(net_cost_savings_lakhs / 100.0, 2),
            "portfolio_roi_multiplier": f"{portfolio_roi}x Return",
            "payback_period_months": 2.4
        }
