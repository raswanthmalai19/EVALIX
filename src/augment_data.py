"""
Data Augmentation Module for Credit Risk Assessment
Injects realistic noise and complexity to break deterministic patterns
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)


def add_feature_noise(df, config):
    """
    Add Gaussian noise to continuous features to simulate real-world measurement errors
    
    Args:
        df: DataFrame with original data
        config: Dictionary with noise parameters for each feature
    
    Returns:
        DataFrame with noisy features
    """
    df_noisy = df.copy()
    
    for feature, params in config.items():
        if feature in df_noisy.columns and params['noise_std'] > 0:
            noise = np.random.normal(0, params['noise_std'], len(df_noisy))
            df_noisy[feature] = df_noisy[feature] + noise
            
            # Apply bounds if specified
            if 'min_val' in params:
                df_noisy[feature] = df_noisy[feature].clip(lower=params['min_val'])
            if 'max_val' in params:
                df_noisy[feature] = df_noisy[feature].clip(upper=params['max_val'])
            
            # Round if specified
            if params.get('round_to'):
                df_noisy[feature] = df_noisy[feature].round(params['round_to'])
    
    return df_noisy


def add_label_noise(df, target_col='loan_approved', flip_rate=0.08):
    """
    Introduce label noise to simulate human judgment errors and edge cases
    
    Args:
        df: DataFrame with original labels
        target_col: Name of target column
        flip_rate: Proportion of labels to flip (default 8%)
    
    Returns:
        DataFrame with noisy labels, flip_mask for tracking
    """
    df_noisy = df.copy()
    n_samples = len(df_noisy)
    n_flips = int(n_samples * flip_rate)
    
    # Randomly select indices to flip
    flip_indices = np.random.choice(df_noisy.index, size=n_flips, replace=False)
    
    # Flip labels
    df_noisy.loc[flip_indices, target_col] = 1 - df_noisy.loc[flip_indices, target_col]
    
    # Create mask for tracking
    flip_mask = pd.Series(False, index=df_noisy.index)
    flip_mask.loc[flip_indices] = True
    
    return df_noisy, flip_mask


def add_feature_interactions(df):
    """
    Create non-linear patterns to break deterministic rules
    
    Strategy:
    - Young + high income → boost approval likelihood (adjust credit_score/DTI slightly)
    - Old + low debt → boost approval (reward financial stability)
    - High assets + borderline credit → slight boost (collateral effect)
    """
    df_interact = df.copy()
    
    # Pattern 1: Young high earners (age < 30, income > 80k) get slight credit boost
    young_high_earner = (df_interact['age'] < 30) & (df_interact['income'] > 80000)
    df_interact.loc[young_high_earner, 'credit_score'] += np.random.uniform(5, 15, young_high_earner.sum())
    
    # Pattern 2: Older low-debt individuals (age > 50, DTI < 0.25) get slight DTI reduction
    stable_senior = (df_interact['age'] > 50) & (df_interact['debt_to_income_ratio'] < 0.25)
    df_interact.loc[stable_senior, 'debt_to_income_ratio'] -= np.random.uniform(0.01, 0.03, stable_senior.sum())
    
    # Pattern 3: High assets (> 300k) with borderline credit (600-680) get credit boost
    high_assets_borderline = (df_interact['assets'] > 300000) & (df_interact['credit_score'].between(600, 680))
    df_interact.loc[high_assets_borderline, 'credit_score'] += np.random.uniform(10, 25, high_assets_borderline.sum())
    
    # Ensure constraints still hold
    df_interact['credit_score'] = df_interact['credit_score'].clip(300, 850)
    df_interact['debt_to_income_ratio'] = df_interact['debt_to_income_ratio'].clip(0, 1)
    
    return df_interact


def inject_outliers(df, outlier_rate=0.05):
    """
    Add outliers and edge cases to increase model difficulty
    
    Types of outliers:
    1. Excellent credit but denied (undisclosed risk factors)
    2. Poor credit but approved (strong collateral/guarantor)
    """
    df_outlier = df.copy()
    n_samples = len(df_outlier)
    n_outliers = int(n_samples * outlier_rate)
    
    # Half approved outliers (good credit but denied)
    n_good_denied = n_outliers // 2
    approved_mask = df_outlier['loan_approved'] == 1
    if approved_mask.sum() >= n_good_denied:
        good_denied_idx = np.random.choice(df_outlier[approved_mask].index, size=n_good_denied, replace=False)
        df_outlier.loc[good_denied_idx, 'loan_approved'] = 0
    
    # Half rejected outliers (poor credit but approved)
    n_poor_approved = n_outliers - n_good_denied
    rejected_mask = df_outlier['loan_approved'] == 0
    if rejected_mask.sum() >= n_poor_approved:
        poor_approved_idx = np.random.choice(df_outlier[rejected_mask].index, size=n_poor_approved, replace=False)
        df_outlier.loc[poor_approved_idx, 'loan_approved'] = 1
    
    return df_outlier


def augment_dataset(input_path, output_path, config=None):
    """
    Main augmentation pipeline
    
    Args:
        input_path: Path to original Loan_Prediction.csv
        output_path: Path to save augmented dataset
        config: Noise configuration (uses defaults if None)
    """
    print("=" * 60)
    print("CREDIT RISK DATA AUGMENTATION")
    print("=" * 60)
    
    # Load original data
    print(f"\n[1/6] Loading original data from: {input_path}")
    df_original = pd.read_csv(input_path)
    print(f"   ✓ Loaded {len(df_original)} samples, {len(df_original.columns)} features")
    
    # Default noise configuration
    if config is None:
        config = {
            'credit_score': {'noise_std': 20, 'min_val': 300, 'max_val': 850, 'round_to': 0},
            'debt_to_income_ratio': {'noise_std': 0.08, 'min_val': 0, 'max_val': 1, 'round_to': 2},
            'income': {'noise_std': 0.10, 'min_val': 0, 'round_to': 0},  # 10% std as fraction
            'assets': {'noise_std': 0.15, 'min_val': 0, 'round_to': 0},  # 15% std as fraction
            'age': {'noise_std': 2, 'min_val': 18, 'max_val': 100, 'round_to': 0}
        }
    
    # Apply percentage-based noise to income and assets
    print("\n[2/6] Adding feature noise (Gaussian)")
    df_aug = df_original.copy()
    
    # For income and assets, convert percentage to absolute
    for feature in ['income', 'assets']:
        if feature in df_aug.columns:
            noise_pct = config[feature]['noise_std']
            noise = np.random.normal(0, 1, len(df_aug)) * df_aug[feature] * noise_pct
            df_aug[feature] = df_aug[feature] + noise
            df_aug[feature] = df_aug[feature].clip(lower=config[feature]['min_val'])
            df_aug[feature] = df_aug[feature].round(0)
    
    # Apply absolute noise to other features
    for feature in ['credit_score', 'debt_to_income_ratio', 'age']:
        if feature in df_aug.columns:
            params = config[feature]
            noise = np.random.normal(0, params['noise_std'], len(df_aug))
            df_aug[feature] = df_aug[feature] + noise
            if 'min_val' in params:
                df_aug[feature] = df_aug[feature].clip(lower=params['min_val'])
            if 'max_val' in params:
                df_aug[feature] = df_aug[feature].clip(upper=params['max_val'])
            if params.get('round_to') is not None:
                df_aug[feature] = df_aug[feature].round(params['round_to'])
    
    print(f"   ✓ Added noise to 5 continuous features")
    
    # Add feature interactions
    print("\n[3/6] Creating non-linear feature interactions")
    df_aug = add_feature_interactions(df_aug)
    print("   ✓ Added 3 interaction patterns (age×income, age×debt, assets×credit)")
    
    # Add label noise
    print("\n[4/6] Introducing label uncertainty (8% flip rate)")
    df_aug, flip_mask = add_label_noise(df_aug, target_col='loan_approved', flip_rate=0.08)
    n_flipped = flip_mask.sum()
    print(f"   ✓ Flipped {n_flipped} labels ({n_flipped/len(df_aug)*100:.1f}%)")
    
    # Inject outliers
    print("\n[5/6] Injecting outliers and edge cases (5% of data)")
    df_aug = inject_outliers(df_aug, outlier_rate=0.05)
    print("   ✓ Added edge cases: excellent-but-denied & poor-but-approved")
    
    # Save augmented dataset
    print(f"\n[6/6] Saving augmented dataset to: {output_path}")
    df_aug.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df_aug)} samples")
    
    # Generate comparison report
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    
    print("\nFeature Statistics Comparison:")
    print("-" * 60)
    for col in ['credit_score', 'debt_to_income_ratio', 'income', 'assets', 'age']:
        if col in df_original.columns:
            orig_mean = df_original[col].mean()
            orig_std = df_original[col].std()
            aug_mean = df_aug[col].mean()
            aug_std = df_aug[col].std()
            
            print(f"\n{col.upper()}:")
            print(f"  Original: μ={orig_mean:.2f}, σ={orig_std:.2f}")
            print(f"  Augmented: μ={aug_mean:.2f}, σ={aug_std:.2f}")
            print(f"  Std change: {(aug_std - orig_std)/orig_std * 100:+.1f}%")
    
    # Class distribution
    print("\n" + "-" * 60)
    print("TARGET DISTRIBUTION:")
    orig_dist = df_original['loan_approved'].value_counts(normalize=True)
    aug_dist = df_aug['loan_approved'].value_counts(normalize=True)
    print(f"  Original - Rejected: {orig_dist[0]*100:.1f}%, Approved: {orig_dist[1]*100:.1f}%")
    print(f"  Augmented - Rejected: {aug_dist[0]*100:.1f}%, Approved: {aug_dist[1]*100:.1f}%")
    
    print("\n" + "=" * 60)
    print("✓ DATA AUGMENTATION COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Run EDA on {output_path}")
    print(f"  2. Update src/preprocessing.py to load Loan_Prediction_Realistic.csv")
    print(f"  3. Re-run all notebooks with realistic data")
    print()


if __name__ == "__main__":
    # Define paths
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "Loan_Prediction.csv"
    output_path = project_root / "data" / "Loan_Prediction_Realistic.csv"
    
    # Run augmentation
    augment_dataset(input_path, output_path)
