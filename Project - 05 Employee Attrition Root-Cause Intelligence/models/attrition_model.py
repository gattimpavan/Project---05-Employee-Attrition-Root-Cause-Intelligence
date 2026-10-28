"""
Predictive Attrition Modeling Engine for Project - 05: Employee Attrition Root-Cause Intelligence
Tata Consultancy Services (TCS) Workforce Analytics & Retention Intelligence.

Builds an ensemble predictive framework:
- Extreme Gradient Boosting (XGBoost)
- Random Forest Classifier
- Penalized Logistic Regression Baseline
Evaluates multi-metric performance (ROC-AUC, Precision, Recall, F1, Confusion Matrix).
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, roc_curve
)
import xgboost as xgb

class TCSAttritionPredictor:
    """End-to-end predictive pipeline for employee turnover hazard estimation."""

    NUMERICAL_FEATURES = [
        "tenure_years", "age", "performance_rating", "client_csat_score",
        "project_delivery_adherence", "billability_rate", "active_projects_count",
        "weekly_hours_logged", "weekly_overtime_hours", "weekend_support_shifts",
        "utilization_rate", "client_escalations_count", "burnout_risk_score",
        "annual_ctc_lakhs", "compa_ratio", "years_since_last_promotion",
        "last_salary_hike_pct", "sick_days_taken", "unplanned_absences",
        "leave_utilization_pct", "pulse_survey_score", "manager_satisfaction_score",
        "peer_collaboration_score", "ijp_rejected_count", "recognition_awards_count",
        "training_hours_completed", "certifications_count", "upskilling_opportunity_score"
    ]

    CATEGORICAL_FEATURES = [
        "business_unit", "designation_band", "location", "work_mode"
    ]

    BINARY_FEATURES = [
        "frequent_overtime", "retention_bonus_eligible", "legacy_tech_lockin"
    ]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.preprocessor = None
        self.xgb_model = None
        self.rf_model = None
        self.lr_model = None
        self.feature_names = []
        self.metrics = {}
        self.is_trained = False

    def build_preprocessor(self):
        """Builds column transformer for scaling and one-hot encoding."""
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.NUMERICAL_FEATURES),
                ("bin", "passthrough", self.BINARY_FEATURES),
                ("cat", cat_transformer, self.CATEGORICAL_FEATURES)
            ]
        )
        return self.preprocessor

    def train_models(self) -> Dict[str, Any]:
        """Trains XGBoost, Random Forest, and Logistic Regression models, evaluating metrics."""
        features = self.NUMERICAL_FEATURES + self.BINARY_FEATURES + self.CATEGORICAL_FEATURES
        X = self.df[features]
        y = self.df["attrition"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        self.build_preprocessor()
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Reconstruct transformed feature names
        cat_features_encoded = self.preprocessor.named_transformers_["cat"].get_feature_names_out(self.CATEGORICAL_FEATURES)
        self.feature_names = self.NUMERICAL_FEATURES + self.BINARY_FEATURES + list(cat_features_encoded)

        # Calculate class weight ratio for imbalanced churn
        pos_weight = (len(y_train) - sum(y_train)) / max(1, sum(y_train))

        # 1. XGBoost
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.06,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=pos_weight * 0.75, # Balanced positive weighting
            eval_metric="logloss",
            random_state=42
        )
        self.xgb_model.fit(X_train_proc, y_train)

        # 2. Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=6,
            class_weight="balanced",
            random_state=42
        )
        self.rf_model.fit(X_train_proc, y_train)

        # 3. Logistic Regression Baseline
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_model.fit(X_train_proc, y_train)

        # Evaluation on Test Set for XGBoost (Primary Production Model)
        y_pred = self.xgb_model.predict(X_test_proc)
        y_prob = self.xgb_model.predict_proba(X_test_proc)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "confusion_matrix": {
                "true_negative": int(cm[0, 0]),
                "false_positive": int(cm[0, 1]),
                "false_negative": int(cm[1, 0]),
                "true_positive": int(cm[1, 1])
            },
            "roc_curve": {
                "fpr": [round(x, 4) for x in fpr[::max(1, len(fpr)//50)]],
                "tpr": [round(x, 4) for x in tpr[::max(1, len(tpr)//50)]]
            },
            "rf_roc_auc": round(float(roc_auc_score(y_test, self.rf_model.predict_proba(X_test_proc)[:, 1])), 4),
            "lr_roc_auc": round(float(roc_auc_score(y_test, self.lr_model.predict_proba(X_test_proc)[:, 1])), 4)
        }

        # Predict hazard probability for entire dataset and append
        X_all_proc = self.preprocessor.transform(X)
        self.df["model_attrition_risk"] = np.round(self.xgb_model.predict_proba(X_all_proc)[:, 1], 4)
        self.df["risk_category"] = pd.cut(
            self.df["model_attrition_risk"],
            bins=[-0.01, 0.25, 0.55, 0.80, 1.0],
            labels=["Low Risk", "Moderate Risk", "High Risk", "Critical Flight Hazard"]
        )

        self.is_trained = True
        return self.metrics

    def predict_employee(self, employee_dict: Dict[str, Any]) -> Tuple[float, str]:
        """Predicts attrition risk probability for a single employee record."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before predicting.")
        
        sample_df = pd.DataFrame([employee_dict])
        features = self.NUMERICAL_FEATURES + self.BINARY_FEATURES + self.CATEGORICAL_FEATURES
        sample_proc = self.preprocessor.transform(sample_df[features])
        prob = float(self.xgb_model.predict_proba(sample_proc)[0, 1])
        
        if prob >= 0.80:
            category = "Critical Flight Hazard"
        elif prob >= 0.55:
            category = "High Risk"
        elif prob >= 0.25:
            category = "Moderate Risk"
        else:
            category = "Low Risk"
            
        return round(prob, 4), category

    def save_checkpoint(self, checkpoint_path: str):
        """Serializes predictor state."""
        with open(checkpoint_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_checkpoint(cls, checkpoint_path: str):
        """Loads serialized predictor."""
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_csv = os.path.join(base_dir, "data", "tcs_employee_attrition.csv")
    predictor = TCSAttritionPredictor(data_csv)
    metrics = predictor.train_models()
    print("XGBoost Training Metrics:", metrics)
