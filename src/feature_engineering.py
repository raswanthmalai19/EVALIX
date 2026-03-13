import pandas as pd
import numpy as np


def credit_tier(score):
    """Map credit score to industry-standard tier (0-4)."""
    if score >= 800:
        return 4  # Excellent
    if score >= 740:
        return 3  # Very Good
    if score >= 670:
        return 2  # Good
    if score >= 580:
        return 1  # Fair
    return 0  # Poor


ENGINEERED_FEATURE_NAMES = [
    "credit_tier",
    "high_debt",
    "asset_income_ratio",
    "age_group",
    "income_per_age",
    "payment_capacity",
    "credit_utilization_proxy",
]


def create_features(df):
    """Add all engineered features and return (df, new_feature_list).
    
    Removed problematic features:
    - financial_health (redundant linear combination of existing features)
    - risk_flags (simple counter, no new information)
    
    Added domain-specific features:
    - payment_capacity: monthly disposable income after debt payments
    - credit_utilization_proxy: estimated credit usage relative to income
    """
    out = df.copy()

    # 1. Credit tier (industry-standard bucketing)
    out["credit_tier"] = out["credit_score"].apply(credit_tier)

    # 2. High debt flag (DTI > 40%)
    out["high_debt"] = (out["debt_to_income_ratio"] > 0.40).astype(int)

    # 3. Asset-to-income ratio (wealth accumulation indicator)
    out["asset_income_ratio"] = out["assets"] / (out["income"] + 1)

    # 4. Age group (life stage bucketing)
    out["age_group"] = pd.cut(
        out["age"], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]
    ).astype(int)

    # 5. Income per age (earning efficiency)
    out["income_per_age"] = out["income"] / (out["age"] + 1)

    # 6. Payment capacity (monthly disposable income after debt)
    # Estimate: monthly income - (monthly income * DTI) = disposable income
    out["payment_capacity"] = (out["income"] * (1 - out["debt_to_income_ratio"])) / 12

    # 7. Credit utilization proxy (existing loan relative to potential credit line)
    # Assumption: credit line ~= 50% of annual income
    out["credit_utilization_proxy"] = out["existing_loan"] / (out["income"] * 0.5 + 1)

    print(f"Created {len(ENGINEERED_FEATURE_NAMES)} engineered features")
    print(f"  Removed: financial_health (redundant), risk_flags (no new info)")
    print(f"  Added: payment_capacity, credit_utilization_proxy")
    return out, ENGINEERED_FEATURE_NAMES


def engineer_single_row(row: dict) -> dict:
    """Apply the same feature engineering to a single input dictionary.

    Used by the backend API at inference time.
    """
    out = dict(row)
    
    # 1. Credit tier
    out["credit_tier"] = credit_tier(out["credit_score"])
    
    # 2. High debt flag
    out["high_debt"] = int(out["debt_to_income_ratio"] > 0.40)
    
    # 3. Asset-to-income ratio
    out["asset_income_ratio"] = out["assets"] / (out["income"] + 1)

    # 4. Age group
    age = out["age"]
    if age <= 30:
        out["age_group"] = 0
    elif age <= 45:
        out["age_group"] = 1
    elif age <= 60:
        out["age_group"] = 2
    else:
        out["age_group"] = 3

    # 5. Income per age
    out["income_per_age"] = out["income"] / (out["age"] + 1)
    
    # 6. Payment capacity
    out["payment_capacity"] = (out["income"] * (1 - out["debt_to_income_ratio"])) / 12
    
    # 7. Credit utilization proxy
    out["credit_utilization_proxy"] = out["existing_loan"] / (out["income"] * 0.5 + 1)
    
    return out