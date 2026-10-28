"""
Root Cause Intelligence Engine for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Organizational Diagnostics & Explainable AI.

Deconstructs macro organizational turnover drivers and provides localized individual
root-cause attribution across the 7 core workplace dimensions:
1. Performance (CSAT, Delivery adherence, ratings)
2. Workload (Project load, utilization, escalations)
3. Compensation (Compa-ratio, hike %, promotion tenure)
4. Absenteeism (Sick leaves, unplanned absences)
5. Overtime (OT hours, weekend support, burnout)
6. Engagement (Pulse survey, manager relationship, IJPs)
7. Training (iON hours, certifications, skill freshness)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any

class RootCauseAnalyzer:
    """Interpretable feature attribution and root-cause diagnostic engine."""

    PILLAR_DEFINITIONS = {
        "overtime_burnout": {
            "title": "Overtime & Burnout Strain",
            "features": ["weekly_overtime_hours", "weekend_support_shifts", "burnout_risk_score", "frequent_overtime"],
            "description": "Chronic long work hours, weekend shifts, and elevated physiological burnout hazard."
        },
        "compensation_equity": {
            "title": "Compensation & Market Parity",
            "features": ["compa_ratio", "last_salary_hike_pct", "annual_ctc_lakhs", "retention_bonus_eligible"],
            "description": "Salary lagging behind industry benchmark, compressed hike percentages, or equity gaps."
        },
        "career_progression": {
            "title": "Career Advancement & Recognition",
            "features": ["years_since_last_promotion", "performance_rating", "recognition_awards_count"],
            "description": "Promotion bottlenecks, star-performer advancement stagnation, and lack of milestone recognition."
        },
        "managerial_engagement": {
            "title": "Manager Relationship & Culture",
            "features": ["manager_satisfaction_score", "pulse_survey_score", "peer_collaboration_score", "ijp_rejected_count"],
            "description": "Leadership misalignment, low psychological safety, and friction in internal mobility."
        },
        "workload_delivery": {
            "title": "Workload & Escalation Pressure",
            "features": ["utilization_rate", "client_escalations_count", "active_projects_count", "weekly_hours_logged"],
            "description": "Sustained utilization exceeding 100%, multi-project cognitive switching, and client escalations."
        },
        "skill_relevance": {
            "title": "Upskilling & Technology Lock-in",
            "features": ["training_hours_completed", "certifications_count", "upskilling_opportunity_score", "legacy_tech_lockin"],
            "description": "Deficit in modern cloud/AI certifications and entrapment in legacy maintenance codebases."
        },
        "absenteeism_precursor": {
            "title": "Absenteeism & Withdrawal Signals",
            "features": ["unplanned_absences", "sick_days_taken", "leave_utilization_pct"],
            "description": "Clustered unscheduled absences serving as an operational proxy for disengagement."
        }
    }

    def __init__(self, predictor, df: pd.DataFrame):
        self.predictor = predictor
        self.df = df.copy()
        self.global_importance = {}
        self._calculate_global_importance()

    def _calculate_global_importance(self):
        """Calculates normalized global feature importance from XGBoost tree splits."""
        if hasattr(self.predictor, "xgb_model") and self.predictor.xgb_model is not None:
            raw_imp = self.predictor.xgb_model.feature_importances_
            feat_names = self.predictor.feature_names
            
            # Map raw importances to human-readable names
            imp_dict = {}
            for f, imp in zip(feat_names, raw_imp):
                # Clean up one-hot prefix
                clean_f = f.replace("cat__", "").replace("num__", "").replace("bin__", "")
                imp_dict[clean_f] = float(imp)

            total = sum(imp_dict.values())
            self.global_importance = {k: round(v / total * 100, 2) for k, v in sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)}

    def get_top_global_drivers(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Returns the top organizational attrition drivers ranked by predictive importance."""
        readable_labels = {
            "weekly_overtime_hours": "Weekly Overtime Hours",
            "compa_ratio": "Compa-Ratio (Salary vs Market Benchmark)",
            "years_since_last_promotion": "Years Since Last Promotion",
            "burnout_risk_score": "Composite Burnout Risk Index",
            "manager_satisfaction_score": "Manager Relationship & Satisfaction",
            "last_salary_hike_pct": "Last Annual Increment (%)",
            "utilization_rate": "Billable Resource Utilization (%)",
            "client_escalations_count": "Client Escalations Count",
            "training_hours_completed": "Upskilling Hours (TCS iON)",
            "pulse_survey_score": "Quarterly Pulse Survey Score",
            "upskilling_opportunity_score": "Technology Evolution Opportunities",
            "weekend_support_shifts": "Weekend Support Shifts",
            "legacy_tech_lockin": "Legacy Technology Trap",
            "ijp_rejected_count": "Internal Job Movement (IJP) Denials",
            "unplanned_absences": "Unplanned Absences (Precursor)",
            "performance_rating": "Annual Performance Appraisal Band"
        }

        top_drivers = []
        count = 0
        for feat, imp in self.global_importance.items():
            if feat in readable_labels:
                top_drivers.append({
                    "feature_key": feat,
                    "label": readable_labels[feat],
                    "relative_importance_pct": imp,
                    "pillar": self._get_pillar_for_feature(feat)
                })
                count += 1
                if count >= top_n:
                    break
        return top_drivers

    def _get_pillar_for_feature(self, feat: str) -> str:
        for p_key, p_info in self.PILLAR_DEFINITIONS.items():
            if feat in p_info["features"]:
                return p_info["title"]
        return "Organizational Demographics"

    def diagnose_employee(self, emp_id: str) -> Dict[str, Any]:
        """Provides a localized root-cause breakdown and 7-pillar radar profiling for an employee."""
        row_matches = self.df[self.df["emp_id"] == emp_id]
        if row_matches.empty:
            return {"error": f"Employee {emp_id} not found."}

        emp = row_matches.iloc[0].to_dict()

        # Risk Score
        risk_prob = float(emp.get("model_attrition_risk", emp.get("attrition_hazard_prob", 0.5)))
        
        # Determine specific negative triggers (Root Causes)
        triggers = []
        anchors = []

        # 1. Overtime & Burnout
        ot_hours = emp["weekly_overtime_hours"]
        shifts = emp["weekend_support_shifts"]
        if ot_hours >= 8.0 or shifts >= 6:
            severity = "Critical" if ot_hours >= 12.0 else "High"
            impact = round(min(0.35, (ot_hours / 24.0) * 0.30 + (shifts / 20.0) * 0.15), 3)
            triggers.append({
                "pillar": "Overtime & Burnout Strain",
                "severity": severity,
                "title": f"Excessive Overwork ({ot_hours} hrs/wk OT, {shifts} weekend shifts)",
                "impact_pct": round(impact * 100, 1),
                "detail": f"Employee logged {ot_hours} hrs of weekly overtime and {shifts} weekend support shifts, generating acute burnout hazard."
            })
        elif ot_hours < 3.0:
            anchors.append({
                "pillar": "Work-Life Balance",
                "title": "Sustainable Working Hours",
                "impact_pct": -6.5,
                "detail": "Healthy work hours (<3 hrs overtime) acting as a protective retention buffer."
            })

        # 2. Compensation & Market Parity
        compa = emp["compa_ratio"]
        hike = emp["last_salary_hike_pct"]
        if compa < 0.85:
            severity = "Critical" if compa < 0.78 else "High"
            impact = round(min(0.30, (0.85 - compa) * 1.2), 3)
            triggers.append({
                "pillar": "Compensation Equity",
                "severity": severity,
                "title": f"Below-Market Compa-Ratio ({compa:.2f})",
                "impact_pct": round(impact * 100, 1),
                "detail": f"Current compensation is {(1.0 - compa)*100:.1f}% below industry and internal peer benchmarks for {emp['designation_band']}."
            })
        elif compa >= 1.05:
            anchors.append({
                "pillar": "Compensation Parity",
                "title": "Competitive Compensation",
                "impact_pct": -8.0,
                "detail": f"Salary is above market median (Compa-Ratio: {compa:.2f}), reducing financial poaching vulnerability."
            })

        # 3. Career Stagnation
        years_stagnant = emp["years_since_last_promotion"]
        rating = emp["performance_rating"]
        if years_stagnant >= 3.0:
            severity = "Critical" if (years_stagnant >= 3.8 and rating >= 4) else "High"
            impact = round(min(0.28, (years_stagnant / 5.0) * 0.22 + (rating >= 4) * 0.08), 3)
            star_note = " (Star Performer Promotion Deficit)" if rating >= 4 else ""
            triggers.append({
                "pillar": "Career Progression",
                "severity": severity,
                "title": f"Career Promotion Stagnation ({years_stagnant} yrs in grade){star_note}",
                "impact_pct": round(impact * 100, 1),
                "detail": f"Employee has been {years_stagnant} years in grade despite holding a performance rating of {rating}/5."
            })
        elif years_stagnant <= 1.2:
            anchors.append({
                "pillar": "Career Momentum",
                "title": "Recent Promotion Momentum",
                "impact_pct": -7.0,
                "detail": f"Promoted within the last {years_stagnant} years, reinforcing career growth satisfaction."
            })

        # 4. Managerial Support & Psychological Safety
        mgr_score = emp["manager_satisfaction_score"]
        pulse = emp["pulse_survey_score"]
        if mgr_score <= 2.4 or pulse <= 2.4:
            triggers.append({
                "pillar": "Managerial & Culture",
                "severity": "High",
                "title": f"Leadership & Engagement Deficit (Manager Score: {mgr_score}/5.0)",
                "impact_pct": 14.5,
                "detail": f"Low manager relationship rating ({mgr_score}/5.0) and quarterly pulse sentiment ({pulse}/5.0) indicating cultural isolation."
            })
        elif mgr_score >= 4.2:
            anchors.append({
                "pillar": "Leadership Anchor",
                "title": "High Managerial Trust & Sponsorship",
                "impact_pct": -9.0,
                "detail": f"Exemplary manager relationship score ({mgr_score}/5.0) acts as primary organizational glue."
            })

        # 5. Technology Lock-in & Skill Frustration
        legacy = emp["legacy_tech_lockin"]
        upskilling = emp["upskilling_opportunity_score"]
        certs = emp["certifications_count"]
        if legacy == 1 and upskilling <= 2.8:
            triggers.append({
                "pillar": "Skill Relevance",
                "severity": "Medium",
                "title": "Legacy Technology Entrapment",
                "impact_pct": 11.2,
                "detail": "Assigned to legacy tech stack with restricted modern AI/Cloud learning pathways, driving market obsolescence anxiety."
            })
        elif certs >= 3:
            anchors.append({
                "pillar": "Skill Growth",
                "title": f"Active Certification Portfolio ({certs} certs)",
                "impact_pct": -6.0,
                "detail": f"Completed {certs} enterprise certifications through TCS Elevate/iON."
            })

        # Sort triggers by impact
        triggers = sorted(triggers, key=lambda x: x["impact_pct"], reverse=True)

        # 7-Pillar Normalized Radar Scores (0 to 100 scale)
        radar_scores = {
            "Performance": round(float((emp["performance_rating"] / 5.0) * 100), 1),
            "Workload Strain": round(float(min(100.0, (emp["weekly_hours_logged"] / 65.0) * 100)), 1),
            "Compensation Parity": round(float(min(100.0, (emp["compa_ratio"] / 1.25) * 100)), 1),
            "Absenteeism Risk": round(float(min(100.0, (emp["unplanned_absences"] / 10.0) * 100)), 1),
            "Overtime Strain": round(float(min(100.0, (emp["weekly_overtime_hours"] / 18.0) * 100)), 1),
            "Engagement Health": round(float((emp["pulse_survey_score"] / 5.0) * 100), 1),
            "Upskilling Velocity": round(float(min(100.0, (emp["training_hours_completed"] / 70.0) * 100)), 1)
        }

        return {
            "employee_info": {
                "emp_id": emp["emp_id"],
                "name": emp["name"],
                "business_unit": emp["business_unit"],
                "designation_band": emp["designation_band"],
                "location": emp["location"],
                "work_mode": emp["work_mode"],
                "tenure_years": emp["tenure_years"],
                "annual_ctc_lakhs": emp["annual_ctc_lakhs"],
                "performance_rating": emp["performance_rating"],
                "replacement_cost_lakhs": emp["replacement_cost_lakhs"]
            },
            "attrition_risk": {
                "probability": risk_prob,
                "risk_category": emp.get("risk_category", "High Risk" if risk_prob >= 0.55 else "Moderate Risk"),
                "observed_status": "Voluntarily Resigned" if emp["attrition"] == 1 else "Active Retained"
            },
            "primary_root_causes": triggers[:3],
            "mitigating_anchors": anchors[:2],
            "radar_profile": radar_scores
        }

    def get_department_root_cause_analysis(self) -> List[Dict[str, Any]]:
        """Computes aggregate root-cause vulnerabilities across TCS Business Units."""
        results = []
        for bu, grp in self.df.groupby("business_unit"):
            turnover_rate = round(float(grp["attrition"].mean() * 100), 2)
            avg_ot = round(float(grp["weekly_overtime_hours"].mean()), 1)
            avg_compa = round(float(grp["compa_ratio"].mean()), 2)
            avg_burnout = round(float(grp["burnout_risk_score"].mean()), 2)
            avg_promo_wait = round(float(grp["years_since_last_promotion"].mean()), 1)
            high_risk_count = int((grp["attrition_hazard_prob"] >= 0.60).sum())
            total_loss_cr = round(float(grp[grp["attrition"] == 1]["replacement_cost_lakhs"].sum() / 100.0), 2)

            # Dominant root cause signature for this BU
            if avg_ot >= 7.0:
                dominant_cause = "Overtime & Burnout Fatigue"
            elif avg_compa < 0.95:
                dominant_cause = "Market Compa-Ratio Deficit"
            elif avg_promo_wait >= 2.4:
                dominant_cause = "Promotion Bottlenecks"
            else:
                dominant_cause = "Skill Obsolescence / Legacy Trap"

            results.append({
                "business_unit": bu,
                "headcount": len(grp),
                "attrition_rate_pct": turnover_rate,
                "high_risk_headcount": high_risk_count,
                "total_replacement_loss_cr": total_loss_cr,
                "avg_weekly_overtime_hours": avg_ot,
                "avg_compa_ratio": avg_compa,
                "avg_burnout_index": avg_burnout,
                "avg_years_since_promotion": avg_promo_wait,
                "primary_root_cause": dominant_cause
            })
        return sorted(results, key=lambda x: x["attrition_rate_pct"], reverse=True)

    def get_band_vulnerability_analysis(self) -> List[Dict[str, Any]]:
        """Computes root-cause metrics segmented by TCS professional hierarchy bands."""
        results = []
        for band, grp in self.df.groupby("designation_band"):
            turnover_rate = round(float(grp["attrition"].mean() * 100), 2)
            avg_salary = round(float(grp["annual_ctc_lakhs"].mean()), 1)
            avg_ot = round(float(grp["weekly_overtime_hours"].mean()), 1)
            avg_compa = round(float(grp["compa_ratio"].mean()), 2)
            avg_promo_wait = round(float(grp["years_since_last_promotion"].mean()), 1)
            avg_pulse = round(float(grp["pulse_survey_score"].mean()), 2)
            total_band_loss_cr = round(float(grp[grp["attrition"] == 1]["replacement_cost_lakhs"].sum() / 100.0), 2)

            results.append({
                "designation_band": band,
                "headcount": len(grp),
                "attrition_rate_pct": turnover_rate,
                "avg_annual_ctc_lakhs": avg_salary,
                "avg_compa_ratio": avg_compa,
                "avg_overtime_hours": avg_ot,
                "avg_years_since_promotion": avg_promo_wait,
                "avg_pulse_survey": avg_pulse,
                "total_replacement_loss_cr": total_band_loss_cr
            })
        return results

