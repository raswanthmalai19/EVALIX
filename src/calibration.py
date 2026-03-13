"""
Probability Calibration Module for Credit Risk Assessment
Ensures predicted probabilities reflect true likelihood of approval
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict
import joblib
import os

# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use('Agg')


def calibrate_model(base_model, X_train, y_train, method='isotonic', cv=5):
    """
    Apply probability calibration to a trained model.
    
    Args:
        base_model: Trained sklearn classifier
        X_train: Training features
        y_train: Training labels
        method: 'isotonic' or 'sigmoid' (isotonic recommended for tree models)
        cv: Number of CV folds
    
    Returns:
        Calibrated model
    """
    print(f"\nCalibrating model with method='{method}', cv={cv}")
    print(f"  Base model: {type(base_model).__name__}")
    
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method=method,
        cv=cv,
        n_jobs=-1
    )
    
    calibrated_model.fit(X_train, y_train)
    print(f"  ✓ Calibration complete")
    
    return calibrated_model


def evaluate_calibration(model, X_test, y_test, model_name="Model", output_dir="../outputs"):
    """
    Evaluate and plot calibration curve.
    
    A well-calibrated model should have a calibration curve close to the diagonal.
    If the curve is a step function, the model is outputting extreme probabilities.
    
    Args:
        model: Trained model (calibrated or base)
        X_test: Test features
        y_test: Test labels
        model_name: Name for plot title
        output_dir: Directory to save plot
    
    Returns:
        Dictionary with calibration metrics
    """
    print(f"\nEvaluating calibration for {model_name}")
    
    # Get predicted probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute calibration curve (bin predictions and compute actual frequencies)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, y_prob, n_bins=10, strategy='uniform'
    )
    
    # Compute calibration error (mean absolute deviation from diagonal)
    calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
    
    # Probability distribution statistics
    prob_mean = np.mean(y_prob)
    prob_std = np.std(y_prob)
    extreme_pct = ((y_prob < 0.1) | (y_prob > 0.9)).mean() * 100
    
    metrics = {
        "calibration_error": calibration_error,
        "prob_mean": prob_mean,
        "prob_std": prob_std,
        "extreme_prob_pct": extreme_pct,
    }
    
    print(f"  Calibration error: {calibration_error:.4f} (lower is better)")
    print(f"  Prob mean: {prob_mean:.3f}, std: {prob_std:.3f}")
    print(f"  Extreme predictions (<0.1 or >0.9): {extreme_pct:.1f}%")
    
    # Plot calibration curve
    _plot_calibration_curve(
        fraction_of_positives,
        mean_predicted_value,
        model_name,
        calibration_error,
        output_dir
    )
    
    return metrics


def _plot_calibration_curve(fraction_of_positives, mean_predicted_value,
                             model_name, cal_error, output_dir):
    """Generate and save calibration curve plot."""
    plt.figure(figsize=(8, 6))
    
    # Plot perfect calibration (diagonal)
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Plot model's calibration
    plt.plot(mean_predicted_value, fraction_of_positives, 's-',
             label=f'{model_name}\n(Error={cal_error:.3f})',
             linewidth=2, markersize=8)
    
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title(f'Calibration Curve: {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f"calibration_curve_{model_name.replace(' ', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Calibration curve saved to {filepath}")


def compare_calibration(base_model, calibrated_model, X_test, y_test, 
                        output_dir="../outputs"):
    """
    Compare base model vs calibrated model.
    
    Args:
        base_model: Original trained model
        calibrated_model: Calibrated version
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save plots
    
    Returns:
        Dictionary with comparison metrics
    """
    print("\n" + "="*60)
    print("CALIBRATION COMPARISON: Base vs Calibrated")
    print("="*60)
    
    # Evaluate both models
    base_metrics = evaluate_calibration(base_model, X_test, y_test, 
                                       "Base Model", output_dir)
    cal_metrics = evaluate_calibration(calibrated_model, X_test, y_test,
                                      "Calibrated Model", output_dir)
    
    # Compare
    print("\n" + "-"*60)
    print("COMPARISON SUMMARY:")
    print(f"  Calibration error:")
    print(f"    Base:       {base_metrics['calibration_error']:.4f}")
    print(f"    Calibrated: {cal_metrics['calibration_error']:.4f}")
    print(f"    Improvement: {(base_metrics['calibration_error'] - cal_metrics['calibration_error']):.4f}")
    
    print(f"\n  Probability std:")
    print(f"    Base:       {base_metrics['prob_std']:.4f}")
    print(f"    Calibrated: {cal_metrics['prob_std']:.4f}")
    
    print(f"\n  Extreme predictions (%):")
    print(f"    Base:       {base_metrics['extreme_prob_pct']:.1f}%")
    print(f"    Calibrated: {cal_metrics['extreme_prob_pct']:.1f}%")
    print("="*60 + "\n")
    
    return {"base": base_metrics, "calibrated": cal_metrics}


def save_calibrated_model(calibrated_model, model_name="best_model_calibrated",
                          model_dir="../models"):
    """Save calibrated model to disk."""
    os.makedirs(model_dir, exist_ok=True)
    filepath = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(calibrated_model, filepath)
    print(f"✓ Saved calibrated model to {filepath}")


def load_calibrated_model(model_name="best_model_calibrated", model_dir="../models"):
    """Load calibrated model from disk."""
    filepath = os.path.join(model_dir, f"{model_name}.pkl")
    model = joblib.load(filepath)
    print(f"✓ Loaded calibrated model from {filepath}")
    return model
