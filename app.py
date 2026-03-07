"""
Evalix Credit Risk Assessment — FastAPI Backend
Serves predictions from the trained best_model.pkl,
applies the same feature engineering used in training,
and returns SHAP explanations alongside the decision.
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── project imports ──────────────────────────────────────────────
from src.feature_engineering import (
    engineer_single_row,
    ENGINEERED_FEATURE_NAMES,
)

# ── paths ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATED_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_calibrated.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "outputs", "scaler.pkl")

# ── load artefacts once at startup ───────────────────────────────
# Use base model (calibrated model over-compresses mid-range probabilities)
model = joblib.load(MODEL_PATH)
print(f"✓ Loaded base model from {MODEL_PATH}")

scaler = joblib.load(SCALER_PATH)

# Feature order used during training (raw + engineered)
RAW_FEATURES = [
    "age", "income", "assets", "credit_score",
    "debt_to_income_ratio", "existing_loan", "criminal_record",
]
ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURE_NAMES

# Optional: load SHAP explainer if available
shap_explainer = None
try:
    import shap
    
    # Handle calibrated models - extract base estimator for SHAP
    if hasattr(model, 'calibrated_classifiers_'):
        base_estimator = model.calibrated_classifiers_[0].estimator
        print(f"Detected CalibratedClassifierCV. Using base estimator: {type(base_estimator).__name__}")
        shap_explainer = shap.TreeExplainer(base_estimator)
    else:
        shap_explainer = shap.TreeExplainer(model)
    
    print("SHAP TreeExplainer loaded ✓")
except Exception as exc:
    print(f"SHAP explainer not available ({exc}). Predictions will still work.")

# ── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Evalix Credit Risk API",
    version="1.0.0",
    description="Predict loan approval with SHAP explanations",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Human-friendly names for the frontend
FEATURE_DISPLAY = {
    "age": "Age",
    "income": "Annual Income",
    "assets": "Total Assets",
    "credit_score": "Credit Score",
    "debt_to_income_ratio": "Debt-to-Income Ratio",
    "existing_loan": "Existing Loan",
    "criminal_record": "Criminal Record",
    "credit_tier": "Credit Tier",
    "high_debt": "High Debt Flag",
    "asset_income_ratio": "Asset-to-Income Ratio",
    "age_group": "Age Group",
    "income_per_age": "Earning Efficiency",
    "payment_capacity": "Monthly Payment Capacity",
    "credit_utilization_proxy": "Credit Utilization",
}


# ── request / response schemas ───────────────────────────────────
class LoanApplication(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Applicant age")
    income: float = Field(..., ge=0, description="Annual income")
    assets: float = Field(..., ge=0, description="Total assets")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score (300-850)")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="DTI ratio (0–1)")
    existing_loan: int = Field(..., ge=0, le=1, description="Has existing loan (0/1)")
    criminal_record: int = Field(..., ge=0, le=1, description="Has criminal record (0/1)")


class PredictionResponse(BaseModel):
    decision: str
    probability: float
    risk_level: str
    confidence: float
    top_risk_factors: list
    top_protective_factors: list
    improvement_suggestions: list


# ── helpers ──────────────────────────────────────────────────────
def _decision_from_probability(prob: float) -> str:
    """Map probability to decision using dynamic thresholds.
    
    Thresholds tuned for the base Random Forest model:
    - High confidence approval: prob >= 0.50
    - Borderline (needs review): 0.25 <= prob < 0.50
    - High confidence rejection: prob < 0.25
    """
    if prob >= 0.50:
        return "Approved"
    elif prob >= 0.25:
        return "Review Needed"
    else:
        return "Rejected"


def _risk_level(prob: float) -> str:
    """Map probability to risk level, aligned with decision thresholds.
    
    Approved zone (>=0.50): Low
    Upper review zone (0.35-0.50): Medium  
    Lower review zone (0.25-0.35): Elevated
    Rejection zone (0.10-0.25): High
    Hard rejection (<0.10): Very High
    """
    if prob >= 0.50:
        return "Low"
    elif prob >= 0.35:
        return "Medium"
    elif prob >= 0.25:
        return "Elevated"
    elif prob >= 0.10:
        return "High"
    else:
        return "Very High"


def _confidence_score(prob: float) -> float:
    """Calculate confidence score that reflects uncertainty in borderline cases.
    
    Confidence is highest at extremes (0 or 1), lowest at 0.5.
    """
    # Distance from 0.5 (uncertainty point)
    distance_from_uncertain = abs(prob - 0.5)
    # Map to 0-100 scale: distance=0.5 → confidence=100, distance=0 → confidence=0
    confidence = (distance_from_uncertain / 0.5) * 100
    return round(confidence, 1)


def _improvement_suggestions(row: dict, prob: float) -> list:
    """Generate actionable suggestions based on the applicant profile."""
    tips = []
    if row["credit_score"] < 670:
        tips.append("Improve your credit score above 670 to reach the 'Good' tier.")
    if row["debt_to_income_ratio"] > 0.40:
        tips.append("Reduce your debt-to-income ratio below 40%.")
    if row["existing_loan"] == 1:
        tips.append("Pay off existing loans before applying.")
    if row["criminal_record"] == 1:
        tips.append("A clean record significantly boosts approval odds.")
    if row["assets"] < row["income"] * 0.5:
        tips.append("Increase your asset base relative to income.")
    if prob < 0.50 and not tips:
        tips.append("Consider applying with a co-signer or collateral.")
    if not tips:
        tips.append("Your profile looks strong — maintain current financials.")
    return tips


# ── endpoints ────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": type(model).__name__,
        "shap_available": shap_explainer is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    try:
        raw = application.model_dump()

        # Feature engineering (same logic as training)
        engineered = engineer_single_row(raw)

        # Build feature vector in the correct order
        feature_vals = [engineered[f] for f in ALL_FEATURES]
        X = pd.DataFrame([feature_vals], columns=ALL_FEATURES)

        # Scale
        X_scaled = pd.DataFrame(
            scaler.transform(X), columns=ALL_FEATURES
        )

        # Predict
        prob = float(model.predict_proba(X_scaled)[0][1])
        decision = _decision_from_probability(prob)
        risk = _risk_level(prob)
        confidence = _confidence_score(prob)

        # SHAP explanations
        risk_factors = []
        protective_factors = []

        if shap_explainer is not None:
            shap_vals = shap_explainer.shap_values(X_scaled)
            # For binary classifiers shap_values may return a list [class0, class1] or multi-dim array
            if isinstance(shap_vals, list):
                # List of arrays - use positive class
                sv = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
            else:
                # NumPy array - handle various shapes
                shape = shap_vals.shape
                if len(shape) == 3 and shape[2] == 2:
                    # Shape: (samples, features, classes) - select positive class
                    sv = shap_vals[0, :, 1]
                elif len(shape) == 2:
                    # Shape: (samples, features) or (features, classes)
                    sv = shap_vals[0] if shape[0] == 1 else shap_vals[:, 1] if shape[1] == 2 else shap_vals[0]
                else:
                    # Shape: (features,) - single sample
                    sv = shap_vals
            
            # Convert to 1D array if needed
            sv = np.array(sv).flatten()[:len(ALL_FEATURES)]

            pairs = sorted(
                zip(ALL_FEATURES, sv), key=lambda x: abs(x[1]), reverse=True
            )
            for feat, val in pairs:
                entry = {
                    "feature": FEATURE_DISPLAY.get(feat, feat),
                    "impact": round(float(val), 4),
                }
                if val < 0:
                    risk_factors.append(entry)
                else:
                    protective_factors.append(entry)

            risk_factors = risk_factors[:5]
            protective_factors = protective_factors[:5]
        else:
            # Fallback: use feature importance if available
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                pairs = sorted(
                    zip(ALL_FEATURES, imp), key=lambda x: x[1], reverse=True
                )[:5]
                for feat, val in pairs:
                    risk_factors.append({
                        "feature": FEATURE_DISPLAY.get(feat, feat),
                        "impact": round(float(val), 4),
                    })

        suggestions = _improvement_suggestions(raw, prob)

        return PredictionResponse(
            decision=decision,
            probability=round(prob, 4),
            risk_level=risk,
            confidence=confidence,
            top_risk_factors=risk_factors,
            top_protective_factors=protective_factors,
            improvement_suggestions=suggestions,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
