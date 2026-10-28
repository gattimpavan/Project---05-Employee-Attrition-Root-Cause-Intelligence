/**
 * Project - 05: Employee Attrition Root-Cause Intelligence
 * Tata Consultancy Services (TCS) Workforce Analytics & Retention AI
 * Frontend Orchestration, Dynamic Chart.js Visualizations & Real-time Simulator
 */

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    loadSummaryKPIs();
    loadRootCauses();
    loadDepartmentAnalysis();
    loadEmployeesAtRisk();
    setupSimulatorEvents();
});

// Global state
let currentSelectedEmpId = "TCS-100000";
let radarChartInstance = null;
let globalDriversChartInstance = null;
let buChartInstance = null;
let bandChartInstance = null;
let burnoutChartInstance = null;
let compaPromoChartInstance = null;

// Tab Switching
function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            const targetEl = document.getElementById(target);
            if (targetEl) targetEl.classList.add("active");
        });
    });
}

// 1. Load Executive Summary KPIs
async function loadSummaryKPIs() {
    try {
        const res = await fetch("/api/summary");
        const data = await res.json();
        if (data.status === "success") {
            document.getElementById("kpi-headcount").textContent = data.total_headcount.toLocaleString();
            document.getElementById("kpi-attrition-rate").textContent = `${data.attrition_rate_pct}%`;
            document.getElementById("kpi-loss-cr").textContent = `₹${data.total_replacement_loss_cr} Cr`;
            
            const pRoi = data.portfolio_roi;
            document.getElementById("kpi-at-risk").textContent = pRoi.total_at_risk_headcount.toLocaleString();
            document.getElementById("kpi-savings-cr").textContent = `₹${pRoi.net_capital_saved_crores} Cr`;
            document.getElementById("kpi-portfolio-roi").textContent = pRoi.portfolio_roi_multiplier;

            // Model badge
            document.getElementById("model-auc-badge").textContent = `XGBoost ROC-AUC: ${data.model_performance.roc_auc}`;
            document.getElementById("model-acc-badge").textContent = `Accuracy: ${data.model_performance.accuracy_pct}%`;
            document.getElementById("model-recall-badge").textContent = `Recall: ${data.model_performance.recall_pct}%`;
        }
    } catch (err) {
        console.error("Failed to load summary KPIs:", err);
    }
}

// 2. Load Root Causes and Global Drivers Chart
async function loadRootCauses() {
    try {
        const res = await fetch("/api/root_causes");
        const data = await res.json();
        if (data.status === "success") {
            const drivers = data.top_drivers;
            renderGlobalDriversChart(drivers);
            renderPillarsCatalog(data.pillars);
        }
    } catch (err) {
        console.error("Failed to load root causes:", err);
    }
}

function renderGlobalDriversChart(drivers) {
    const ctx = document.getElementById("globalDriversChart");
    if (!ctx) return;

    const labels = drivers.map(d => d.label);
    const values = drivers.map(d => d.relative_importance_pct);
    const backgroundColors = [
        "#ef4444", "#f97316", "#f59e0b", "#38bdf8", "#818cf8",
        "#a855f7", "#ec4899", "#10b981", "#14b8a6", "#64748b"
    ];

    if (globalDriversChartInstance) globalDriversChartInstance.destroy();

    globalDriversChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Predictive Feature Attribution (%)",
                data: values,
                backgroundColor: backgroundColors,
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` Relative Impact: ${ctx.parsed.x}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: "rgba(255,255,255,0.06)" },
                    ticks: { color: "#94a3b8", callback: v => `${v}%` }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: "#f8fafc", font: { size: 11 } }
                }
            }
        }
    });
}

function renderPillarsCatalog(pillars) {
    const container = document.getElementById("pillarsGrid");
    if (!container) return;

    let html = "";
    for (const [key, p] of Object.entries(pillars)) {
        html += `
            <div class="kpi-card" style="border-left: 3px solid var(--accent-cyan);">
                <div class="kpi-label">${p.title}</div>
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 8px;">${p.description}</div>
                <div style="font-size: 0.75rem; color: var(--accent-cyan); display: flex; flex-wrap: wrap; gap: 4px;">
                    ${p.features.map(f => `<span class="badge badge-cyan">${f}</span>`).join('')}
                </div>
            </div>
        `;
    }
    container.innerHTML = html;
}

// 3. Load Department & Band Analysis
async function loadDepartmentAnalysis() {
    try {
        const res = await fetch("/api/department_analysis");
        const data = await res.json();
        if (data.status === "success") {
            renderBUChart(data.business_units);
            renderBandChart(data.designation_bands);
            renderBUTable(data.business_units);
            renderBandTable(data.designation_bands);
        }
    } catch (err) {
        console.error("Failed to load department analysis:", err);
    }
}

function renderBUChart(buList) {
    const ctx = document.getElementById("buAttritionChart");
    if (!ctx) return;

    const labels = buList.map(b => b.business_unit.split(" (")[0]);
    const rates = buList.map(b => b.attrition_rate_pct);
    const losses = buList.map(b => b.total_replacement_loss_cr);

    if (buChartInstance) buChartInstance.destroy();

    buChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Attrition Rate (%)",
                    data: rates,
                    backgroundColor: "rgba(239, 68, 68, 0.75)",
                    borderRadius: 6,
                    yAxisID: "y"
                },
                {
                    label: "Replacement Loss (₹ Cr)",
                    data: losses,
                    backgroundColor: "rgba(56, 189, 248, 0.75)",
                    borderRadius: 6,
                    yAxisID: "y1"
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#cbd5e1" } }
            },
            scales: {
                x: { ticks: { color: "#94a3b8", maxRotation: 25 } },
                y: {
                    type: "linear",
                    position: "left",
                    ticks: { color: "#f87171", callback: v => `${v}%` },
                    grid: { color: "rgba(255,255,255,0.06)" }
                },
                y1: {
                    type: "linear",
                    position: "right",
                    ticks: { color: "#38bdf8", callback: v => `₹${v}Cr` },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderBandChart(bandList) {
    const ctx = document.getElementById("bandAttritionChart");
    if (!ctx) return;

    const labels = bandList.map(b => b.designation_band.replace("Assistant System Engineer", "ASE").replace("System Engineer", "SE").replace("IT Analyst", "ITA").replace("Assistant Consultant", "AST").replace("Associate Consultant", "CON").replace("Delivery Manager / Project Lead", "DM"));
    const rates = bandList.map(b => b.attrition_rate_pct);
    const compa = bandList.map(b => b.avg_compa_ratio * 100);

    if (bandChartInstance) bandChartInstance.destroy();

    bandChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "Turnover Hazard (%)",
                    data: rates,
                    borderColor: "#ef4444",
                    backgroundColor: "rgba(239, 68, 68, 0.15)",
                    fill: true,
                    tension: 0.35,
                    pointRadius: 5
                },
                {
                    label: "Compa-Ratio Benchmark Index (%)",
                    data: compa,
                    borderColor: "#10b981",
                    borderDash: [5, 5],
                    tension: 0.35,
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { labels: { color: "#cbd5e1" } } },
            scales: {
                x: { ticks: { color: "#94a3b8" } },
                y: {
                    ticks: { color: "#94a3b8" },
                    grid: { color: "rgba(255,255,255,0.06)" }
                }
            }
        }
    });
}

function renderBUTable(buList) {
    const tbody = document.getElementById("buTableBody");
    if (!tbody) return;

    let html = "";
    buList.forEach(b => {
        html += `
            <tr>
                <td style="font-weight: 600;">${b.business_unit}</td>
                <td>${b.headcount.toLocaleString()}</td>
                <td><span class="badge ${b.attrition_rate_pct >= 20 ? 'badge-crimson' : 'badge-amber'}">${b.attrition_rate_pct}%</span></td>
                <td>${b.high_risk_headcount.toLocaleString()}</td>
                <td>₹${b.total_replacement_loss_cr} Cr</td>
                <td>${b.avg_weekly_overtime_hours} hrs/wk</td>
                <td>${b.avg_compa_ratio}</td>
                <td><span class="badge badge-cyan">${b.primary_root_cause}</span></td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}

function renderBandTable(bandList) {
    const tbody = document.getElementById("bandTableBody");
    if (!tbody) return;

    let html = "";
    bandList.forEach(b => {
        html += `
            <tr>
                <td style="font-weight: 600;">${b.designation_band}</td>
                <td>${b.headcount.toLocaleString()}</td>
                <td><span class="badge ${b.attrition_rate_pct >= 20 ? 'badge-crimson' : 'badge-amber'}">${b.attrition_rate_pct}%</span></td>
                <td>₹${b.avg_annual_ctc_lakhs} L</td>
                <td>${b.avg_compa_ratio}</td>
                <td>${b.avg_overtime_hours} hrs/wk</td>
                <td>${b.avg_years_since_promotion} yrs</td>
                <td>₹${b.total_replacement_loss_cr} Cr</td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}

// 4. Load High-Risk Employees
async function loadEmployeesAtRisk() {
    const buFilter = document.getElementById("filterBU") ? document.getElementById("filterBU").value : "All";
    const bandFilter = document.getElementById("filterBand") ? document.getElementById("filterBand").value : "All";

    try {
        const res = await fetch(`/api/employees_at_risk?bu=${encodeURIComponent(buFilter)}&band=${encodeURIComponent(bandFilter)}&limit=30`);
        const data = await res.json();
        if (data.status === "success") {
            renderEmployeeTable(data.employees);
            if (data.employees.length > 0 && !currentSelectedEmpId) {
                currentSelectedEmpId = data.employees[0].emp_id;
            }
        }
    } catch (err) {
        console.error("Failed to load employees at risk:", err);
    }
}

function renderEmployeeTable(employees) {
    const tbody = document.getElementById("employeeTableBody");
    if (!tbody) return;

    let html = "";
    employees.forEach(emp => {
        const riskPct = Math.round(emp.attrition_risk_prob * 100);
        let badgeClass = "badge-crimson";
        if (riskPct < 55) badgeClass = "badge-amber";
        if (riskPct < 25) badgeClass = "badge-green";

        html += `
            <tr>
                <td style="font-weight: 700; color: var(--accent-cyan); cursor: pointer;" onclick="inspectEmployee('${emp.emp_id}')">${emp.emp_id}</td>
                <td style="font-weight: 600;">${emp.name}</td>
                <td style="font-size: 0.8rem; color: var(--text-muted);">${emp.business_unit.split(" (")[0]}</td>
                <td>${emp.designation_band.split(" (")[0]}</td>
                <td>₹${emp.annual_ctc_lakhs} L</td>
                <td>${emp.compa_ratio}</td>
                <td>${emp.weekly_overtime_hours} hrs</td>
                <td>${emp.years_since_last_promotion} yrs</td>
                <td><span class="badge ${badgeClass}">${riskPct}% Flight Risk</span></td>
                <td>
                    <button class="btn btn-secondary" style="padding: 5px 12px; font-size: 0.75rem;" onclick="inspectEmployee('${emp.emp_id}')">
                        Dossier & Plan
                    </button>
                    <button class="btn btn-emerald" style="padding: 5px 10px; font-size: 0.75rem;" onclick="loadEmployeeIntoSimulator('${emp.emp_id}')">
                        Simulate
                    </button>
                </td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}

// 5. Inspect Employee Modal
async function inspectEmployee(empId) {
    currentSelectedEmpId = empId;
    try {
        const res = await fetch(`/api/employee/${empId}`);
        const data = await res.json();
        if (data.status === "success") {
            const diag = data.diagnosis;
            const pres = data.prescription;

            document.getElementById("modalEmpName").textContent = `${diag.employee_info.name} (${diag.employee_info.emp_id})`;
            document.getElementById("modalEmpMeta").textContent = `${diag.employee_info.designation_band} | ${diag.employee_info.business_unit} | CTC: ₹${diag.employee_info.annual_ctc_lakhs} LPA`;

            const probPct = Math.round(diag.attrition_risk.probability * 100);
            document.getElementById("modalRiskBadge").textContent = `${probPct}% Flight Hazard (${diag.attrition_risk.risk_category})`;
            document.getElementById("modalRiskBadge").className = `badge ${probPct >= 55 ? 'badge-crimson' : 'badge-amber'}`;

            // Render 7-Pillar Radar Chart
            renderRadarChart(diag.radar_profile);

            // Render Root Causes
            let triggersHtml = "";
            diag.primary_root_causes.forEach(t => {
                triggersHtml += `
                    <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: var(--radius-sm); padding: 10px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 700; color: #f87171; font-size: 0.85rem;">${t.title}</span>
                            <span class="badge badge-crimson">+${t.impact_pct}% Hazard</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #cbd5e1;">${t.detail}</div>
                    </div>
                `;
            });
            document.getElementById("modalTriggers").innerHTML = triggersHtml || "<p style='color: var(--text-muted);'>No acute operational triggers diagnosed.</p>";

            // Render Prescriptions
            let presHtml = "";
            pres.prescribed_interventions.forEach(p => {
                presHtml += `
                    <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: var(--radius-sm); padding: 12px 14px; margin-bottom: 8px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                            <span style="font-weight: 700; color: #34d399; font-size: 0.9rem;">${p.intervention_title}</span>
                            <span class="badge badge-green">-${p.projected_risk_drop_pct}% Risk</span>
                        </div>
                        <div style="font-size: 0.8rem; color: #e2e8f0; margin-bottom: 6px;">${p.tactical_mechanism}</div>
                        <div style="font-size: 0.75rem; color: var(--text-muted); display: flex; gap: 14px;">
                            <span>Investment: <b style="color: #fff;">₹${p.cost_inr_lakhs} L</b></span>
                            <span>Target SLA: <b style="color: #fff;">${p.turnaround_days} Days</b></span>
                            <span>Addresses: <b style="color: var(--accent-cyan);">${p.addresses_root_cause}</b></span>
                        </div>
                    </div>
                `;
            });
            document.getElementById("modalPrescriptions").innerHTML = presHtml;

            // ROI summary
            document.getElementById("modalGrossSavings").textContent = `₹${pres.financial_roi.expected_gross_savings_lakhs} L`;
            document.getElementById("modalInterventionCost").textContent = `₹${pres.financial_roi.total_intervention_cost_lakhs} L`;
            document.getElementById("modalNetSavings").textContent = `₹${pres.financial_roi.net_realized_savings_lakhs} L`;
            document.getElementById("modalRoiMultiplier").textContent = pres.financial_roi.roi_multiplier;

            // Open Modal
            document.getElementById("employeeModal").classList.add("active");
        }
    } catch (err) {
        console.error("Failed to inspect employee:", err);
    }
}

function closeModal() {
    document.getElementById("employeeModal").classList.remove("active");
}

function renderRadarChart(profile) {
    const ctx = document.getElementById("employeeRadarChart");
    if (!ctx) return;

    const labels = Object.keys(profile);
    const values = Object.values(profile);

    if (radarChartInstance) radarChartInstance.destroy();

    radarChartInstance = new Chart(ctx, {
        type: "radar",
        data: {
            labels: labels,
            datasets: [{
                label: "Employee Operational Strain Profile",
                data: values,
                backgroundColor: "rgba(56, 189, 248, 0.25)",
                borderColor: "#38bdf8",
                pointBackgroundColor: "#38bdf8",
                pointBorderColor: "#fff",
                pointHoverBackgroundColor: "#fff",
                pointHoverBorderColor: "#38bdf8"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: "rgba(255,255,255,0.1)" },
                    grid: { color: "rgba(255,255,255,0.08)" },
                    pointLabels: { color: "#cbd5e1", font: { size: 10 } },
                    ticks: { display: false, min: 0, max: 100 }
                }
            },
            plugins: { legend: { display: false } }
        }
    });
}

// 6. What-If Simulator Orchestration
function setupSimulatorEvents() {
    const simHike = document.getElementById("simHike");
    const simHikeVal = document.getElementById("simHikeVal");
    const simOT = document.getElementById("simOT");
    const simOTVal = document.getElementById("simOTVal");
    const simTraining = document.getElementById("simTraining");
    const simTrainingVal = document.getElementById("simTrainingVal");
    const simPromo = document.getElementById("simPromo");
    const simManager = document.getElementById("simManager");

    if (simHike) {
        simHike.addEventListener("input", (e) => {
            simHikeVal.textContent = `+${e.target.value}%`;
            triggerSimulation();
        });
    }
    if (simOT) {
        simOT.addEventListener("input", (e) => {
            simOTVal.textContent = `-${e.target.value} hrs`;
            triggerSimulation();
        });
    }
    if (simTraining) {
        simTraining.addEventListener("input", (e) => {
            simTrainingVal.textContent = `+${e.target.value} hrs`;
            triggerSimulation();
        });
    }
    if (simPromo) simPromo.addEventListener("change", triggerSimulation);
    if (simManager) simManager.addEventListener("change", triggerSimulation);

    // Initial simulation
    triggerSimulation();
}

function loadEmployeeIntoSimulator(empId) {
    currentSelectedEmpId = empId;
    // Switch to tab 3
    const simTabBtn = document.querySelector('[data-tab="tab-simulator"]');
    if (simTabBtn) simTabBtn.click();
    triggerSimulation();
}

async function triggerSimulation() {
    const empId = currentSelectedEmpId || "TCS-100000";
    const hike = parseFloat(document.getElementById("simHike") ? document.getElementById("simHike").value : 0);
    const ot = parseFloat(document.getElementById("simOT") ? document.getElementById("simOT").value : 0);
    const promo = document.getElementById("simPromo") ? document.getElementById("simPromo").checked : false;
    const training = parseFloat(document.getElementById("simTraining") ? document.getElementById("simTraining").value : 0);
    const mgr = document.getElementById("simManager") ? document.getElementById("simManager").checked : false;

    try {
        const res = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                emp_id: empId,
                salary_hike_delta_pct: hike,
                overtime_reduction_hours: ot,
                grant_promotion: promo,
                add_training_hours: training,
                improve_manager: mgr
            })
        });
        const data = await res.json();
        if (data.status === "success") {
            renderSimulationOutput(data.simulation);
        }
    } catch (err) {
        console.error("Simulation failed:", err);
    }
}

function renderSimulationOutput(sim) {
    document.getElementById("simEmpHeader").textContent = `Simulating Retention Package for: ${sim.employee_name} (${sim.emp_id})`;

    const baseProb = Math.round(sim.baseline.probability * 100);
    const simProb = Math.round(sim.simulated.probability * 100);

    document.getElementById("simBaseGauge").textContent = `${baseProb}%`;
    document.getElementById("simBaseCat").textContent = sim.baseline.risk_category;

    document.getElementById("simPostGauge").textContent = `${simProb}%`;
    document.getElementById("simPostCat").textContent = sim.simulated.risk_category;

    document.getElementById("simDeltaDrop").textContent = `-${sim.impact_delta.risk_reduction_absolute_pct}% Absolute Risk Reduction (${sim.impact_delta.relative_risk_drop_pct}% Relative Lift)`;

    // Financial outputs
    document.getElementById("simCostVal").textContent = `₹${sim.economics.intervention_cost_lakhs} L`;
    document.getElementById("simAvoidedVal").textContent = `₹${sim.economics.replacement_cost_avoided_lakhs} L`;
    document.getElementById("simNetSavingsVal").textContent = `₹${sim.economics.net_realized_savings_lakhs} L`;
    document.getElementById("simRoiVal").textContent = sim.economics.simulated_roi_multiplier;
}

