"""
Prepare Data for Model Training
Reads data_labeled.csv and creates train/test split files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def prepare_data(
    input_file='data/processed/data_labeled.csv',
    output_folder='processed_data',
    target_column='CRASH_TYPE',
    test_size=0.2,
    random_state=42
):
    """
    Load labeled data and create train/test split files
    
    Parameters:
    -----------
    input_file : str
        Path to your data_labeled.csv
    output_folder : str
        Folder to save processed files
    target_column : str
        Name of target variable column
    test_size : float
        Proportion for test set (default 20%)
    random_state : int
        Random seed for reproducibility
    """
    
    print("=" * 60)
    print("📂 PREPARING DATA FOR MODEL TRAINING")
    print("=" * 60)
    
    # Create output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"\n✓ Created folder: {output_folder}/")
    
    # Load data
    print(f"\n📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"✓ Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"\n📋 Columns found:")
    print(df.columns.tolist())
    
    # Check target column exists
    if target_column not in df.columns:
        print(f"\n❌ Error: Target column '{target_column}' not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Show target distribution
    print(f"\n🎯 Target Variable: '{target_column}'")
    print(f"Distribution:")
    print(df[target_column].value_counts())
    
    # Separate features and target
    # Drop LOCATION_CLUSTER if exists (as you mentioned)
    columns_to_drop = [target_column]
    if 'LOCATION_CLUSTER' in df.columns:
        columns_to_drop.append('LOCATION_CLUSTER')
        print(f"\n✓ Dropping 'LOCATION_CLUSTER' column")
    
    X = df.drop(columns=columns_to_drop)
    y = df[target_column]
    
    print(f"\n📊 Features shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Feature names
    feature_names = X.columns.tolist()
    print(f"\n🔢 Features ({len(feature_names)}):")
    for i, feat in enumerate(feature_names, 1):
        print(f"   {i}. {feat}")
    
    # Train-test split
    print(f"\n✂️ Splitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class distribution
    )
    
    print(f"✓ X_train: {X_train.shape}")
    print(f"✓ X_test: {X_test.shape}")
    print(f"✓ y_train: {y_train.shape}")
    print(f"✓ y_test: {y_test.shape}")
    
    # Check class distribution in splits
    print(f"\n📊 Class distribution in train set:")
    print(y_train.value_counts())
    print(f"\n📊 Class distribution in test set:")
    print(y_test.value_counts())
    
    # Save files
    print(f"\n💾 Saving files to {output_folder}/...")
    
    # Save X_train
    X_train.to_csv(f'{output_folder}/X_train_processed.csv', index=False)
    print(f"✓ Saved: {output_folder}/X_train_processed.csv")
    
    # Save X_test
    X_test.to_csv(f'{output_folder}/X_test_processed.csv', index=False)
    print(f"✓ Saved: {output_folder}/X_test_processed.csv")
    
    # Save y_train (as numpy array)
    np.save(f'{output_folder}/y_train_processed.npy', y_train.values)
    print(f"✓ Saved: {output_folder}/y_train_processed.npy")
    
    # Save y_test (as numpy array)
    np.save(f'{output_folder}/y_test_processed.npy', y_test.values)
    print(f"✓ Saved: {output_folder}/y_test_processed.npy")
    
    # Save feature names for reference
    with open(f'{output_folder}/feature_names.txt', 'w') as f:
        for feat in feature_names:
            f.write(f"{feat}\n")
    print(f"✓ Saved: {output_folder}/feature_names.txt")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 60)
    
    print(f"\n📁 Files created in '{output_folder}/':")
    for f in os.listdir(output_folder):
        size = os.path.getsize(f'{output_folder}/{f}') / 1024  # KB
        print(f"   • {f} ({size:.1f} KB)")
    
    print(f"\n🚀 Next step: Run 'python train_model.py'")
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    
    # ========================================
    # CONFIGURE THESE PATHS IF NEEDED
    # ========================================
    
    INPUT_FILE = 'data/processed/data_labeled.csv'  # Your input file
    OUTPUT_FOLDER = 'processed_data'                 # Output folder
    TARGET_COLUMN = 'CRASH_TYPE'                     # Target variable
    TEST_SIZE = 0.2                                  # 20% for testing
    
    # ========================================
    
    prepare_data(
        input_file=INPUT_FILE,
        output_folder=OUTPUT_FOLDER,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE
    )