import os
import time
import joblib
import numpy as np
import warnings
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)


def train_model(model, X_train, y_train, model_name="Model"):
    """Train a sklearn-compatible model and report elapsed time."""
    print(f"Training {model_name} ...", end=" ", flush=True)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    print(f"done in {elapsed:.2f}s")
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Return metrics dict, y_pred, y_prob. Includes sanity checks for model behavior."""
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )
    
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1_Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }
    
    # Sanity checks
    _check_model_sanity(metrics, y_prob, model_name)
    
    return metrics, y_pred, y_prob


def _check_model_sanity(metrics, y_prob, model_name):
    """
    Flag suspicious model behavior that indicates deterministic patterns.
    
    Red flags:
    1. Accuracy > 95% (unrealistic for credit risk)
    2. Probability variance < 0.15 (extreme probabilities near 0 or 1)
    3. Perfect precision or recall = 1.0
    """
    issues = []
    
    # Check 1: Suspiciously high accuracy
    if metrics["Accuracy"] > 0.95:
        issues.append(f"Accuracy={metrics['Accuracy']:.4f} is unrealistically high (>95%)")
    
    # Check 2: Extreme probability predictions
    if y_prob is not None:
        prob_std = np.std(y_prob)
        if prob_std < 0.15:
            issues.append(f"Probability std={prob_std:.4f} is too low (<0.15) - model outputs extreme values")
        
        # Check proportion of extreme predictions (near 0 or 1)
        extreme_mask = (y_prob < 0.1) | (y_prob > 0.9)
        extreme_pct = extreme_mask.mean() * 100
        if extreme_pct > 80:
            issues.append(f"{extreme_pct:.1f}% of predictions are extreme (<0.1 or >0.9) - indicates if-else behavior")
    
    # Check 3: Perfect precision or recall
    if metrics["Precision"] >= 0.999:
        issues.append(f"Precision={metrics['Precision']:.4f} is suspiciously perfect")
    if metrics["Recall"] >= 0.999:
        issues.append(f"Recall={metrics['Recall']:.4f} is suspiciously perfect")
    
    # Emit warnings if issues found
    if issues:
        warning_msg = f"\n⚠️  SANITY CHECK WARNING for {model_name}:\n"
        for issue in issues:
            warning_msg += f"   - {issue}\n"
        warning_msg += "   → This suggests deterministic patterns or data leakage. Review data quality.\n"
        warnings.warn(warning_msg, UserWarning)
    else:
        print(f"  ✓ Sanity checks passed for {model_name}")


def save_model(model, model_name, model_dir="../models"):
    """Persist a trained model with joblib."""
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, path)
    print(f"Saved {path}")


def load_model(model_name, model_dir="../models"):
    """Load a persisted model."""
    path = os.path.join(model_dir, f"{model_name}.pkl")
    model = joblib.load(path)
    print(f"Loaded {path}")
    return model