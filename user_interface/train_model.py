"""
Train Model with Proper Class Imbalance Handling
Saves full pipeline + all metadata for Streamlit app
"""

import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, make_scorer
)
from sklearn.ensemble import ExtraTreesClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import lightgbm as lgb

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATH = "data/processed/data_labeled.csv"  # Your data file
MODEL_DIR = Path("model")
REPORTS_DIR = Path("reports")
RANDOM_STATE = 42
TARGET = "CRASH_TYPE"

# Create directories
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FEATURE CATEGORY MAPPINGS (for UI dropdowns)
# ============================================================================

FEATURE_MAPPINGS = {
    'TRAFFIC_CONTROL_DEVICE': ['NO CONTROLS', 'TRAFFIC SIGNAL', 'STOP SIGN/FLASHER', 'UNKNOWN', 'OTHER'],
    'DEVICE_CONDITION': ['NO CONTROLS', 'UNKNOWN', 'FUNCTIONING PROPERLY', 'OTHER', 
                         'NOT FUNCTIONING', 'FUNCTIONING IMPROPERLY'],
    'WEATHER_CONDITION': ['CLEAR', 'RAIN', 'UNKNOWN', 'CLOUDY', 'SNOW', 'FOG', 'OTHER',
                          'SEVERE CROSS WIND', 'HAIL', 'BLOWING SNOW', 'SANDSTORM'],
    'LIGHTING_CONDITION': ['DAYLIGHT', 'DARKNESS, LIGHTED ROAD', 'UNKNOWN', 'DARKNESS', 'DAWN', 'DUSK'],
    'FIRST_CRASH_TYPE': ['TURNING', 'REAR END', 'OTHER OBJECT', 'SIDESWIPE SAME DIRECTION', 'ANGLE',
                         'PARKED MOTOR VEHICLE', 'SIDESWIPE OPPOSITE DIRECTION', 'FIXED OBJECT',
                         'PEDALCYCLIST', 'REAR TO FRONT', 'PEDESTRIAN', 'HEAD ON', 'REAR TO SIDE',
                         'OTHER NONCOLLISION', 'REAR TO REAR', 'ANIMAL', 'OVERTURNED', 'TRAIN'],
    'TRAFFICWAY_TYPE': ['NOT DIVIDED', 'INTERSECTION', 'DIVIDED', 'PARKING LOT', 'ONE-WAY', 'FOUR WAY',
                        'UNKNOWN', 'ALLEY', 'CENTER TURN LANE', 'OTHER', 'RAMP', 'FIVE POINT, OR MORE',
                        'TRAFFIC ROUTE', 'ROUNDABOUT', 'DRIVEWAY'],
    'ALIGNMENT': ['STRAIGHT', 'CURVE'],
    'ROADWAY_SURFACE_COND': ['UNKNOWN', 'DRY', 'WET', 'ICE', 'OTHER', 'SNOW', 'SAND/MUD/DIRT'],
    'ROAD_DEFECT': ['NO DEFECTS', 'DEFECT', 'UNKNOWN'],
    'REPORT_TYPE': ['NOT ON SCENE (DESK REPORT)', 'ON SCENE', 'UNKNOWN', 'AMENDED'],
    'INTERSECTION_RELATED_I': ['UNKNOWN', 'Y', 'N'],
    'DAMAGE': ['OVER $1,500', '$501 - $1,500', '$500 OR LESS'],
    'PRIM_CONTRIBUTORY_CAUSE': ['BAD DRIVING SKILLS', 'NOT APPLICABLE', 'UNABLE TO DETERMINE',
                                 'DISREGARDING TRAFFIC RULES', 'OVERSPEEDING', 'WEATHER', 'DRINKING',
                                 'VEHICLE CONDITION', 'DISTRACTION', 'OBSTRUCTION',
                                 'PHYSICAL CONDITION OF DRIVER', 'RELATED TO BUS STOP', 'ROAD DEFECTS',
                                 'EVASIVE ACTION', 'PASSING STOPPED SCHOOL BUS', 'TEXTING',
                                 'BICYCLE/MOTORCYCLE ON RED'],
    'SEC_CONTRIBUTORY_CAUSE': ['UNABLE TO DETERMINE', 'NOT APPLICABLE', 'BAD DRIVING SKILLS',
                                'OVERSPEEDING', 'DISREGARDING TRAFFIC RULES', 'WEATHER', 'VEHICLE CONDITION',
                                'DISTRACTION', 'OBSTRUCTION', 'DRINKING', 'RELATED TO BUS STOP',
                                'ROAD DEFECTS', 'PHYSICAL CONDITION OF DRIVER', 'EVASIVE ACTION', 'TEXTING',
                                'BICYCLE/MOTORCYCLE ON RED', 'PASSING STOPPED SCHOOL BUS'],
    'CRASH_TIME_OF_DAY': ['Morning', 'Afternoon', 'Evening', 'Night'],
    'CRASH_DAY_OF_WEEK': {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 
                          5: 'Friday', 6: 'Saturday', 7: 'Sunday'},
    'CRASH_MONTH': {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                    7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
}


def train_and_save():
    """Train model with class balancing and save for Streamlit"""
    
    print("=" * 70)
    print("🚀 TRAINING MODEL WITH CLASS IMBALANCE HANDLING")
    print("=" * 70)
    
    # ==================== LOAD DATA ====================
    print(f"\n📂 Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Dataset shape: {df.shape}")
    
    # Check target
    assert TARGET in df.columns, f"Target '{TARGET}' not found!"
    
    # Separate features and target
    X = df.drop(columns=[TARGET])
    
    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(df[TARGET])
    
    print(f"\n🎯 Target classes: {le.classes_}")
    print(f"Class distribution:\n{pd.Series(y).value_counts()}")
    
    # Feature names
    numeric_features = X.columns.tolist()
    print(f"\n📊 Features ({len(numeric_features)}): {numeric_features}")
    
    # ==================== TRAIN/TEST SPLIT ====================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n✂️ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # ==================== PREPROCESSING ====================
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    preprocess = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)],
        remainder="drop"
    )
    
    # ==================== MODEL WITH SMOTE (Best for imbalance) ====================
    print("\n🔧 Building LightGBM + SMOTE pipeline...")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    
    # LightGBM with SMOTE for class balancing
    pipe_lgbm_smote = ImbPipeline([
        ("preprocess", preprocess),
        ("smote", smote),
        ("clf", lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',  # Additional balancing
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ))
    ])
    
    # Alternative: ExtraTrees with balanced weights (also good)
    pipe_extratrees = Pipeline([
        ("preprocess", preprocess),
        ("clf", ExtraTreesClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])
    
    # ==================== CROSS-VALIDATION ====================
    print("\n📊 Running cross-validation...")
    
    scoring = {
        "macro_f1": make_scorer(f1_score, average="macro"),
        "bal_acc": make_scorer(balanced_accuracy_score),
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    
    models = {
        "LightGBM+SMOTE": pipe_lgbm_smote,
        "ExtraTrees": pipe_extratrees
    }
    
    best_score = 0
    best_model_name = None
    best_model = None
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        
        macro_f1 = np.mean(cv_results["test_macro_f1"])
        bal_acc = np.mean(cv_results["test_bal_acc"])
        
        print(f"  ✓ {name}: Macro F1={macro_f1:.4f}, Balanced Acc={bal_acc:.4f}")
        
        if macro_f1 > best_score:
            best_score = macro_f1
            best_model_name = name
            best_model = model
    
    print(f"\n🏆 Best model: {best_model_name} (Macro F1: {best_score:.4f})")
    
    # ==================== TRAIN BEST MODEL ====================
    print(f"\n🎯 Training {best_model_name} on full training data...")
    best_model.fit(X_train, y_train)
    
    # ==================== EVALUATE ====================
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    test_bal_acc = balanced_accuracy_score(y_test, y_pred)
    test_macro_f1 = f1_score(y_test, y_pred, average="macro")
    
    print(f"\n📈 Test Results:")
    print(f"   Balanced Accuracy: {test_bal_acc:.4f}")
    print(f"   Macro F1-Score: {test_macro_f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    # Check prediction distribution
    print(f"\n📊 Prediction distribution on test set:")
    pred_counts = pd.Series(y_pred).value_counts()
    for idx, count in pred_counts.items():
        print(f"   {le.classes_[idx]}: {count} ({count/len(y_pred)*100:.1f}%)")
    
    # ==================== SAVE ARTIFACTS ====================
    print("\n💾 Saving model and metadata...")
    
    # 1. Save full pipeline
    joblib.dump(best_model, MODEL_DIR / "best_pipeline.joblib")
    print(f"✓ Saved: {MODEL_DIR}/best_pipeline.joblib")
    
    # 2. Save label encoder
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    print(f"✓ Saved: {MODEL_DIR}/label_encoder.joblib")
    
    # 3. Save feature names
    with open(MODEL_DIR / "feature_names.json", "w") as f:
        json.dump(numeric_features, f)
    print(f"✓ Saved: {MODEL_DIR}/feature_names.json")
    
    # 4. Save feature mappings (for UI)
    with open(MODEL_DIR / "feature_mappings.json", "w") as f:
        json.dump(FEATURE_MAPPINGS, f, indent=2)
    print(f"✓ Saved: {MODEL_DIR}/feature_mappings.json")
    
    # 5. Save metrics
    metrics = {
        "best_model": best_model_name,
        "test_balanced_accuracy": float(test_bal_acc),
        "test_macro_f1": float(test_macro_f1),
        "classes": le.classes_.tolist(),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test))
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved: {MODEL_DIR}/metrics.json")
    
    # 6. Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    np.save(MODEL_DIR / "confusion_matrix.npy", cm)
    print(f"✓ Saved: {MODEL_DIR}/confusion_matrix.npy")
    
    # 7. Save feature importance
    if hasattr(best_model.named_steps['clf'], 'feature_importances_'):
        importances = best_model.named_steps['clf'].feature_importances_
        feat_df = pd.DataFrame({
            "feature": numeric_features,
            "importance": importances
        }).sort_values("importance", ascending=False)
        feat_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)
        print(f"✓ Saved: {MODEL_DIR}/feature_importance.csv")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📁 All files saved to: {MODEL_DIR.absolute()}")
    print("\n🚀 Next: Run 'streamlit run app.py'")
    
    return best_model


if __name__ == "__main__":
    train_and_save()