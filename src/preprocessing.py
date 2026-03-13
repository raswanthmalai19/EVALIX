import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
import joblib
import os
import warnings


def load_data(filepath="../data/Loan_Prediction_Realistic.csv"):
    """Load the realistic loan prediction dataset with noise and complexity."""
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Using realistic dataset with injected noise and label uncertainty")
    return df


def handle_missing_values(df):
    """Fill missing values with column medians (more robust to outliers)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} NaNs with median={median_val:.2f}")
    return df


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Stratified train-test split + StandardScaler."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print(f"Training set: {X_train_scaled.shape[0]} samples")
    print(f"Test set:     {X_test_scaled.shape[0]} samples")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def apply_smote(X_train, y_train, random_state=42):
    """Apply BorderlineSMOTE to balance minority class (focuses on decision boundary)."""
    print(f"Before SMOTE: {(y_train == 0).sum()} rejected, {(y_train == 1).sum()} approved")
    
    # Use BorderlineSMOTE instead of vanilla SMOTE for better generalization
    # It focuses on borderline samples near the decision boundary
    smote = BorderlineSMOTE(random_state=random_state, k_neighbors=7)
    
    try:
        X_bal, y_bal = smote.fit_resample(X_train, y_train)
        print(f"After  BorderlineSMOTE: {(y_bal == 0).sum()} rejected, {(y_bal == 1).sum()} approved")
    except ValueError as e:
        warnings.warn(f"BorderlineSMOTE failed: {e}. Falling back to no oversampling.")
        return X_train, y_train
    
    return X_bal, y_bal


def save_preprocessed(X_train, X_test, y_train, y_test, scaler,
                       X_original, y_original, output_dir="../outputs"):
    """Persist all preprocessed artefacts."""
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False, header=["loan_approved"])
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False, header=["loan_approved"])
    X_original.to_csv(os.path.join(output_dir, "X_original.csv"), index=False)
    y_original.to_csv(os.path.join(output_dir, "y_original.csv"), index=False, header=["loan_approved"])
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    print(f"All artefacts saved to {output_dir}/")


def load_preprocessed(output_dir="../outputs"):
    """Reload preprocessed data."""
    X_train = pd.read_csv(os.path.join(output_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(output_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(output_dir, "y_train.csv"))["loan_approved"]
    y_test = pd.read_csv(os.path.join(output_dir, "y_test.csv"))["loan_approved"]
    scaler = joblib.load(os.path.join(output_dir, "scaler.pkl"))
    print(f"Loaded  X_train={X_train.shape}  X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler