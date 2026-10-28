"""
Data Generator for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Workforce Telemetry & Turnover Dataset Generator.

Generates 15,000 realistic employee records spanning:
- Performance (ratings, CSAT, delivery milestone adherence, billability)
- Workload (active projects, weekly logged hours, client escalations, utilization)
- Compensation (annual CTC, compa-ratio, promotion tenure, last hike, retention bonus)
- Absenteeism (sick days, unplanned absences, leave utilization rate)
- Overtime (weekly OT hours, weekend support shifts, frequent OT flag, burnout score)
- Engagement (pulse survey, manager relationship, peer feedback, rejected IJPs, awards)
- Training & Skills (learning hours, certifications, upskilling score, legacy lock-in)
"""

import os
import json
import numpy as np
import pandas as pd

def generate_tcs_attrition_data(data_dir: str, n_samples: int = 15000, random_seed: int = 42) -> pd.DataFrame:
    """Generates synthetic workforce telemetry for TCS with realistic organizational root causes."""
    np.random.seed(random_seed)
    os.makedirs(data_dir, exist_ok=True)

    # 1. Organizational Metadata
    business_units = [
        "BFSI (Banking & Financial Services)",
        "Retail, CPG & Travel",
        "Life Sciences & Healthcare",
        "Manufacturing & Utilities",
        "HiTech & Professional Services",
        "Communications, Media & Telecom",
        "Cloud, AI & Digital Transformation"
    ]
    bu_weights = [0.32, 0.17, 0.14, 0.12, 0.10, 0.08, 0.07]

    bands = [
        "Assistant System Engineer (ASE)",
        "System Engineer (SE)",
        "IT Analyst (ITA)",
        "Assistant Consultant (AST)",
        "Associate Consultant (CON)",
        "Delivery Manager / Project Lead (DM)"
    ]
    band_weights = [0.24, 0.31, 0.23, 0.12, 0.07, 0.03]

    locations = ["Chennai", "Bengaluru", "Hyderabad", "Pune", "Mumbai", "Kolkata", "Delhi-NCR"]
    loc_weights = [0.28, 0.24, 0.20, 0.11, 0.09, 0.04, 0.04]

    work_modes = ["Hybrid (3 Days Office)", "Client Onsite Delivery", "Remote Delivery Center"]
    work_mode_weights = [0.65, 0.22, 0.13]

    first_names = [
        "Aarav", "Aditi", "Ananya", "Arjun", "Deepak", "Divya", "Gautam", "Ishaan",
        "Kavya", "Manish", "Neha", "Nikhil", "Pooja", "Pranav", "Priyanka", "Rahul",
        "Riya", "Rohan", "Siddharth", "Sneha", "Surya", "Tanvi", "Varun", "Vikas"
    ]
    last_names = [
        "Sharma", "Verma", "Patel", "Reddy", "Nair", "Iyer", "Rao", "Chatterjee",
        "Mukherjee", "Singh", "Gupta", "Deshmukh", "Kulkarni", "Mehta", "Joshi", "Das"
    ]

    emp_ids = [f"TCS-{100000 + i}" for i in range(n_samples)]
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_samples)]
    gender = np.random.choice(["Female", "Male", "Non-Binary"], size=n_samples, p=[0.38, 0.60, 0.02])
    bu = np.random.choice(business_units, size=n_samples, p=bu_weights)
    band = np.random.choice(bands, size=n_samples, p=band_weights)
    location = np.random.choice(locations, size=n_samples, p=loc_weights)
    work_mode = np.random.choice(work_modes, size=n_samples, p=work_mode_weights)

    # 2. Tenure & Age mapped to Band
    band_tenure_map = {
        "Assistant System Engineer (ASE)": (0.5, 2.0, 22, 25),
        "System Engineer (SE)": (1.5, 4.0, 24, 28),
        "IT Analyst (ITA)": (3.5, 7.5, 27, 33),
        "Assistant Consultant (AST)": (6.5, 11.0, 31, 38),
        "Associate Consultant (CON)": (9.0, 15.0, 35, 44),
        "Delivery Manager / Project Lead (DM)": (12.0, 20.0, 38, 52)
    }

    tenure_years = []
    age = []
    annual_ctc_lakhs = []
    base_salaries = {
        "Assistant System Engineer (ASE)": (3.6, 4.5),
        "System Engineer (SE)": (4.8, 7.2),
        "IT Analyst (ITA)": (8.5, 13.5),
        "Assistant Consultant (AST)": (14.0, 20.0),
        "Associate Consultant (CON)": (21.0, 29.0),
        "Delivery Manager / Project Lead (DM)": (30.0, 42.0)
    }

    for b in band:
        t_min, t_max, a_min, a_max = band_tenure_map[b]
        t = np.round(np.random.uniform(t_min, t_max), 1)
        a = int(np.random.randint(a_min, a_max + 1))
        tenure_years.append(t)
        age.append(a)

        s_min, s_max = base_salaries[b]
        # Introduce compa-ratio variation
        sal = np.round(np.random.uniform(s_min, s_max), 2)
        annual_ctc_lakhs.append(sal)

    tenure_years = np.array(tenure_years)
    age = np.array(age)
    annual_ctc_lakhs = np.array(annual_ctc_lakhs)

    # 3. Workload & Overtime Data
    weekly_hours_logged = np.round(np.random.normal(45.5, 5.5, n_samples).clip(38.0, 68.0), 1)
    weekly_overtime_hours = np.round(np.maximum(0.0, weekly_hours_logged - 45.0 + np.random.normal(0, 2.5, n_samples)).clip(0.0, 24.0), 1)
    weekend_support_shifts = np.random.poisson(lam=weekly_overtime_hours * 0.55).clip(0, 26)
    frequent_overtime = (weekly_overtime_hours >= 7.5).astype(int)
    
    active_projects_count = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.48, 0.38, 0.11, 0.03])
    utilization_rate = np.round((weekly_hours_logged / 45.0 * 100.0 + np.random.normal(0, 4.0, n_samples)).clip(65.0, 135.0), 1)
    
    # Client escalations correlated with high utilization and fatigue
    escalation_prob = (utilization_rate > 105.0) * 0.35 + 0.08
    client_escalations_count = np.random.binomial(n=4, p=np.clip(escalation_prob, 0.05, 0.65))

    # Burnout index (0.0 to 1.0 composite)
    burnout_risk_score = np.round(
        (0.40 * (weekly_overtime_hours / 24.0) +
         0.25 * (weekend_support_shifts / 26.0) +
         0.20 * (np.clip(utilization_rate - 85.0, 0, 50) / 50.0) +
         0.15 * (client_escalations_count / 4.0)).clip(0.0, 1.0), 3
    )

    # 4. Compensation Metrics
    # Compa-ratio: actual / median
    median_ctc_map = {
        "Assistant System Engineer (ASE)": 4.0,
        "System Engineer (SE)": 6.0,
        "IT Analyst (ITA)": 11.0,
        "Assistant Consultant (AST)": 17.0,
        "Associate Consultant (CON)": 25.0,
        "Delivery Manager / Project Lead (DM)": 36.0
    }
    medians = np.array([median_ctc_map[b] for b in band])
    compa_ratio = np.round((annual_ctc_lakhs / medians).clip(0.68, 1.38), 3)

    # Years since last promotion
    years_since_last_promotion = np.round(np.random.exponential(scale=1.8, size=n_samples).clip(0.1, 6.2), 1)
    
    # Last salary hike percentage (TCS standard increments: 3% to 15%)
    last_salary_hike_pct = np.round(np.random.normal(7.2, 3.4, n_samples).clip(1.5, 18.0), 1)
    
    # Retention bonus flag (primarily offered to high-demand skills or senior architects)
    retention_bonus_eligible = np.random.binomial(n=1, p=np.where(compa_ratio < 0.85, 0.12, 0.28))

    # 5. Performance Metrics
    # Performance rating 1 to 5 (TCS bands: 5 is Band A top 10%, 4 is Band B 30%, 3 is Band C 50%, 1-2 bottom 10%)
    perf_probs = [0.03, 0.07, 0.52, 0.28, 0.10]
    performance_rating = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=perf_probs)

    client_csat_score = np.round((performance_rating * 0.7 + np.random.normal(1.3, 0.4, n_samples)).clip(1.5, 5.0), 2)
    project_delivery_adherence = np.round((70.0 + performance_rating * 5.5 + np.random.normal(0, 4.0, n_samples)).clip(60.0, 100.0), 1)
    billability_rate = np.round(np.random.normal(88.0, 10.0, n_samples).clip(45.0, 100.0), 1)

    # 6. Absenteeism
    # Overworked/burned-out employees experience higher unplanned absences and sick days
    sick_days_taken = np.random.poisson(lam=4.0 + burnout_risk_score * 7.0).clip(0, 24)
    unplanned_absences = np.random.poisson(lam=1.5 + burnout_risk_score * 4.5).clip(0, 16)
    leave_utilization_pct = np.round(np.random.normal(74.0, 14.0, n_samples).clip(25.0, 100.0), 1)

    # 7. Engagement & Workplace Sentiment
    # Manager relationship score (1.0 to 5.0)
    manager_satisfaction_score = np.round(np.random.normal(3.6, 0.85, n_samples).clip(1.0, 5.0), 2)
    peer_collaboration_score = np.round(np.random.normal(3.8, 0.75, n_samples).clip(1.0, 5.0), 2)
    
    # Pulse survey (quarterly engagement score)
    pulse_survey_score = np.round(
        (0.35 * manager_satisfaction_score + 
         0.25 * peer_collaboration_score + 
         0.20 * (5.0 - burnout_risk_score * 4.0) +
         0.20 * (compa_ratio * 4.0)).clip(1.0, 5.0), 2
    )

    # Internal Job Movement (IJP) rejected
    ijp_rejected_count = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.72, 0.18, 0.07, 0.03])
    recognition_awards_count = np.random.poisson(lam=performance_rating * 0.35).clip(0, 6)

    # 8. Training & Upskilling (TCS Elevate / iON Ecosystem)
    training_hours_completed = np.round(np.random.exponential(scale=28.0, size=n_samples).clip(4.0, 130.0), 1)
    certifications_count = np.random.poisson(lam=training_hours_completed / 32.0).clip(0, 7)
    legacy_tech_lockin = np.random.binomial(n=1, p=np.where(bu == "BFSI (Banking & Financial Services)", 0.36, 0.20))
    upskilling_opportunity_score = np.round(
        (np.random.normal(3.4, 0.8, n_samples) - legacy_tech_lockin * 0.9 + (certifications_count > 1) * 0.6).clip(1.0, 5.0), 2
    )

    # 9. Ground-Truth Attrition Synthesis (Realistic IT Industry Attrition Dynamics)
    # Latent Log-Odds of Attrition
    # Key Root Causes modeled:
    # 1. Overtime Burnout (long hours, weekend shifts)
    # 2. Compensation gap (low compa-ratio, low hike)
    # 3. Career stagnation (years without promotion, high rating but no advancement)
    # 4. Manager & Culture friction (low manager satisfaction, rejected IJPs)
    # 5. Skill stagnation (legacy lock-in, low upskilling opportunities)
    # 6. Mitigating anchors: high training, fair compa-ratio, retention bonuses, awards

    log_odds = (
        -2.50 # Base rate anchor (~16-18% attrition)
        # Burnout and Overwork Drivers
        + 1.85 * (weekly_overtime_hours >= 8.0) * ((weekly_overtime_hours - 8.0) / 10.0)
        + 0.95 * (weekend_support_shifts >= 8)
        + 0.85 * (burnout_risk_score > 0.60)
        + 0.70 * (client_escalations_count >= 2)
        # Compensation & Growth Drivers
        + 1.65 * (compa_ratio < 0.82)
        + 1.10 * ((compa_ratio >= 0.82) & (compa_ratio < 0.92))
        + 1.45 * (years_since_last_promotion >= 3.2)
        + 0.90 * (last_salary_hike_pct < 4.5)
        # Star Performer Trap (Rating 4/5 with Stalled Promotion & Low Compa)
        + 1.40 * ((performance_rating >= 4) & (years_since_last_promotion >= 2.5) & (compa_ratio < 0.95))
        # Managerial & Cultural Drivers
        + 1.25 * (manager_satisfaction_score <= 2.2)
        + 0.95 * (ijp_rejected_count >= 1)
        + 0.75 * (pulse_survey_score <= 2.4)
        # Absenteeism Correlation
        + 0.65 * (unplanned_absences >= 6)
        # Skill Obsolescence & Technology Frustration
        + 0.85 * (legacy_tech_lockin == 1) * (upskilling_opportunity_score < 2.8)
        # Retention Anchors (Protective Factors)
        - 1.10 * (compa_ratio >= 1.08)
        - 0.90 * (retention_bonus_eligible == 1)
        - 0.80 * (training_hours_completed >= 50.0)
        - 0.75 * (certifications_count >= 3)
        - 0.65 * (manager_satisfaction_score >= 4.2)
        - 0.50 * (recognition_awards_count >= 2)
    )

    attrition_probs = 1.0 / (1.0 + np.exp(-log_odds))
    attrition = (np.random.uniform(0, 1, n_samples) < attrition_probs).astype(int)

    # Calculate Replacement Cost per employee (in INR Lakhs)
    # Standard IT Industry formula: 75% of Annual CTC + escalation overhead
    replacement_cost_lakhs = np.round(annual_ctc_lakhs * 0.75 + (client_escalations_count * 0.40), 2)

    df = pd.DataFrame({
        "emp_id": emp_ids,
        "name": names,
        "gender": gender,
        "age": age,
        "tenure_years": tenure_years,
        "business_unit": bu,
        "designation_band": band,
        "location": location,
        "work_mode": work_mode,
        # Performance
        "performance_rating": performance_rating,
        "client_csat_score": client_csat_score,
        "project_delivery_adherence": project_delivery_adherence,
        "billability_rate": billability_rate,
        # Workload
        "active_projects_count": active_projects_count,
        "weekly_hours_logged": weekly_hours_logged,
        "weekly_overtime_hours": weekly_overtime_hours,
        "weekend_support_shifts": weekend_support_shifts,
        "frequent_overtime": frequent_overtime,
        "utilization_rate": utilization_rate,
        "client_escalations_count": client_escalations_count,
        "burnout_risk_score": burnout_risk_score,
        # Compensation
        "annual_ctc_lakhs": annual_ctc_lakhs,
        "compa_ratio": compa_ratio,
        "years_since_last_promotion": years_since_last_promotion,
        "last_salary_hike_pct": last_salary_hike_pct,
        "retention_bonus_eligible": retention_bonus_eligible,
        # Absenteeism
        "sick_days_taken": sick_days_taken,
        "unplanned_absences": unplanned_absences,
        "leave_utilization_pct": leave_utilization_pct,
        # Engagement
        "pulse_survey_score": pulse_survey_score,
        "manager_satisfaction_score": manager_satisfaction_score,
        "peer_collaboration_score": peer_collaboration_score,
        "ijp_rejected_count": ijp_rejected_count,
        "recognition_awards_count": recognition_awards_count,
        # Training & Upskilling
        "training_hours_completed": training_hours_completed,
        "certifications_count": certifications_count,
        "legacy_tech_lockin": legacy_tech_lockin,
        "upskilling_opportunity_score": upskilling_opportunity_score,
        # Target & Economics
        "attrition_hazard_prob": np.round(attrition_probs, 4),
        "attrition": attrition,
        "replacement_cost_lakhs": replacement_cost_lakhs
    })

    csv_path = os.path.join(data_dir, "tcs_employee_attrition.csv")
    df.to_csv(csv_path, index=False)

    # Create Baseline Metrics JSON
    baseline_stats = {
        "organization": "Tata Consultancy Services (TCS)",
        "total_workforce_records": len(df),
        "total_attrition_count": int(df["attrition"].sum()),
        "overall_attrition_rate_pct": round(float(df["attrition"].mean() * 100), 2),
        "average_tenure_years": round(float(df["tenure_years"].mean()), 2),
        "total_annual_replacement_loss_cr": round(float(df[df["attrition"] == 1]["replacement_cost_lakhs"].sum() / 100.0), 2),
        "high_risk_headcount": int((df["attrition_hazard_prob"] >= 0.60).sum()),
        "average_weekly_hours": round(float(df["weekly_hours_logged"].mean()), 1),
        "overtime_headcount_pct": round(float(df["frequent_overtime"].mean() * 100), 1),
        "underpaid_below_market_pct": round(float((df["compa_ratio"] < 0.85).mean() * 100), 1),
        "promotion_stagnant_pct": round(float((df["years_since_last_promotion"] >= 3.0).mean() * 100), 1),
        "burnout_at_risk_pct": round(float((df["burnout_risk_score"] >= 0.50).mean() * 100), 1)
    }

    benchmark_path = os.path.join(data_dir, "tcs_attrition_benchmark.json")
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(baseline_stats, f, indent=4)

    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_tcs_attrition_data(current_dir, n_samples=15000)
    print(f"Generated {len(df):,} TCS records with attrition rate: {df['attrition'].mean()*100:.2f}%")

