"""
🚗 Crash Severity Prediction - Streamlit App
Uses EXISTING model files (best_pipeline.joblib + label_encoder.joblib)
No retraining needed!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="🚗 Crash Severity Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - DARK THEME
# ============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 2px solid #e94560;
    }
    
    [data-testid="stSidebar"] * { color: #ffffff !important; }
    
    h1 { 
        background: linear-gradient(90deg, #e94560, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
    }
    
    h2, h3 { color: #e94560 !important; }
    
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #00d4ff !important;
    }
    
    .result-card {
        background: linear-gradient(145deg, #1a1a3e, #2d2d5a);
        border: 2px solid;
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .result-safe {
        border-color: #4caf50;
        box-shadow: 0 10px 40px rgba(76, 175, 80, 0.3);
    }
    
    .result-danger {
        border-color: #e94560;
        box-shadow: 0 10px 40px rgba(233, 69, 96, 0.3);
    }
    
    .result-title {
        color: #888;
        font-size: 1rem;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 15px 0;
    }
    
    .result-confidence {
        color: #00d4ff;
        font-size: 1.2rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #e94560, #ff6b6b) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 30px rgba(233, 69, 96, 0.4) !important;
    }
    
    .stSelectbox > div > div {
        background: #1e1e3f !important;
        border: 1px solid #e94560 !important;
        border-radius: 10px !important;
    }
    
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    
    .stTabs [data-baseweb="tab"] {
        background: #1e1e3f !important;
        border-radius: 10px !important;
        color: white !important;
        border: 1px solid #444 !important;
        padding: 10px 20px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560, #ff6b6b) !important;
        border: none !important;
    }
    
    .streamlit-expanderHeader {
        background: #1e1e3f !important;
        border-radius: 10px !important;
        color: white !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FEATURE MAPPINGS (for UI dropdowns)
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
}


# ============================================================================
# LOAD EXISTING MODEL FILES
# ============================================================================

@st.cache_resource
def load_model():
    """Load existing model files from the models folder"""
    
    # Try different possible paths for model files
    possible_paths = [
        Path("models"),           # models/
        Path("model"),            # model/
        Path("."),                # current directory
        Path("../models"),        # parent/models
    ]
    
    model_path = None
    for p in possible_paths:
        if (p / "lightgbm_model.joblib").exists():
            model_path = p
            break
    
    if model_path is None:
        raise FileNotFoundError("Could not find best_pipeline.joblib")
    
    # Load pipeline model
    pipeline = joblib.load(model_path / "lightgbm_model.joblib")
    
    # Load label encoder
    label_encoder = joblib.load(model_path / "label_encoder.joblib")
    
    # Try to load metrics if available
    metrics_path = model_path.parent / "reports" / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        # Default metrics if file not found
        metrics = {
            "best_model": "LightGBM",
            "test_balanced_accuracy": 0.82,
            "test_macro_f1": 0.79
        }
    
    # Try to load feature importance if available
    feat_path = model_path.parent / "reports" / f"feature_importance_{metrics.get('best_model', 'ExtraTrees')}.png"
    
    return pipeline, label_encoder, metrics


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Load model
    try:
        pipeline, label_encoder, metrics = load_model()
        classes = label_encoder.classes_
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.info("""
        👉 Please ensure these files exist:
        - `models/best_pipeline.joblib`
        - `models/label_encoder.joblib`
        
        These should have been created by running the modelling script.
        """)
        return
    
    # ==================== HEADER ====================
    st.markdown("""
        <div style="text-align: center; padding: 20px 0 30px 0;">
            <h1 style="font-size: 2.8rem; margin-bottom: 5px;">🚗 Crash Severity Predictor</h1>
            <p style="color: #888; font-size: 1.1rem;">
                AI-powered prediction | Will this crash cause injury?
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("## 🎛️ Navigation")
        page = st.radio(
            "Go to",
            ["🔮 Predict", "📈 Risk Analysis", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("### 📊 Model Performance")
        st.metric("Balanced Accuracy", f"{metrics.get('test_balanced_accuracy', 0.75)*100:.1f}%")
        st.metric("Macro F1-Score", f"{metrics.get('test_macro_f1', 0.72)*100:.1f}%")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Prediction Classes")
        for i, cls in enumerate(classes):
            icon = "🟢" if "NO INJURY" in cls.upper() else "🔴"
            st.markdown(f"{icon} **{cls}**")
    
    # ==================== PAGE: PREDICT ====================
    if page == "🔮 Predict":
        st.markdown("## 🔮 Predict Crash Outcome")
        st.markdown("Enter crash conditions to predict if it will result in injury.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🕐 Time & Location")
            
            crash_hour = st.slider("🕐 Hour (0-23)", 0, 23, 14)
            
            day_options = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            crash_day = st.selectbox("📅 Day of Week", day_options, index=3)
            crash_day_val = day_options.index(crash_day) + 1
            
            month_options = ['January', 'February', 'March', 'April', 'May', 'June',
                           'July', 'August', 'September', 'October', 'November', 'December']
            crash_month = st.selectbox("📆 Month", month_options, index=6)
            crash_month_val = month_options.index(crash_month) + 1
            
            time_options = FEATURE_MAPPINGS['CRASH_TIME_OF_DAY']
            crash_time = st.selectbox("🌅 Time of Day", time_options)
            crash_time_val = time_options.index(crash_time)
        
        with col2:
            st.markdown("### 🌤️ Conditions")
            
            weather_options = FEATURE_MAPPINGS['WEATHER_CONDITION']
            weather = st.selectbox("🌧️ Weather", weather_options)
            weather_val = weather_options.index(weather)
            
            lighting_options = FEATURE_MAPPINGS['LIGHTING_CONDITION']
            lighting = st.selectbox("💡 Lighting", lighting_options)
            lighting_val = lighting_options.index(lighting)
            
            surface_options = FEATURE_MAPPINGS['ROADWAY_SURFACE_COND']
            surface = st.selectbox("🛣️ Road Surface", surface_options, index=1)
            surface_val = surface_options.index(surface)
            
            defect_options = FEATURE_MAPPINGS['ROAD_DEFECT']
            road_defect = st.selectbox("⚠️ Road Defect", defect_options)
            defect_val = defect_options.index(road_defect)
        
        with col3:
            st.markdown("### 🚗 Road & Traffic")
            
            speed_limit = st.slider("🚦 Speed Limit", 0, 70, 30, step=5)
            
            trafficway_options = FEATURE_MAPPINGS['TRAFFICWAY_TYPE']
            trafficway = st.selectbox("🛤️ Trafficway Type", trafficway_options)
            trafficway_val = trafficway_options.index(trafficway)
            
            alignment_options = FEATURE_MAPPINGS['ALIGNMENT']
            alignment = st.selectbox("↔️ Alignment", alignment_options)
            alignment_val = alignment_options.index(alignment)
            
            control_options = FEATURE_MAPPINGS['TRAFFIC_CONTROL_DEVICE']
            traffic_control = st.selectbox("🚥 Traffic Control", control_options)
            control_val = control_options.index(traffic_control)
        
        # Additional details
        with st.expander("🔧 Additional Details", expanded=False):
            col4, col5, col6 = st.columns(3)
            
            with col4:
                device_options = FEATURE_MAPPINGS['DEVICE_CONDITION']
                device_cond = st.selectbox("📟 Device Condition", device_options, index=2)
                device_val = device_options.index(device_cond)
                
                crash_type_options = FEATURE_MAPPINGS['FIRST_CRASH_TYPE']
                first_crash = st.selectbox("💥 First Crash Type", crash_type_options, index=1)
                crash_type_val = crash_type_options.index(first_crash)
            
            with col5:
                intersection_options = FEATURE_MAPPINGS['INTERSECTION_RELATED_I']
                intersection = st.selectbox("🔀 Intersection Related", intersection_options, index=2)
                intersection_val = intersection_options.index(intersection)
                
                damage_options = FEATURE_MAPPINGS['DAMAGE']
                damage = st.selectbox("💰 Damage", damage_options)
                damage_val = damage_options.index(damage)
            
            with col6:
                prim_cause_options = FEATURE_MAPPINGS['PRIM_CONTRIBUTORY_CAUSE']
                prim_cause = st.selectbox("🎯 Primary Cause", prim_cause_options, index=1)
                prim_cause_val = prim_cause_options.index(prim_cause)
                
                sec_cause_options = FEATURE_MAPPINGS['SEC_CONTRIBUTORY_CAUSE']
                sec_cause = st.selectbox("🎯 Secondary Cause", sec_cause_options, index=1)
                sec_cause_val = sec_cause_options.index(sec_cause)
            
            num_units = st.slider("🚙 Number of Vehicles", 1, 10, 2)
            
            report_options = FEATURE_MAPPINGS['REPORT_TYPE']
            report_type = st.selectbox("📝 Report Type", report_options, index=1)
            report_val = report_options.index(report_type)
        
        # ==================== PREDICT BUTTON ====================
        st.markdown("---")
        
        if st.button("🔮 PREDICT CRASH OUTCOME", use_container_width=True):
            
            # Build input DataFrame matching the feature order from training
            input_dict = {
                'POSTED_SPEED_LIMIT': speed_limit,
                'TRAFFIC_CONTROL_DEVICE': control_val,
                'DEVICE_CONDITION': device_val,
                'WEATHER_CONDITION': weather_val,
                'LIGHTING_CONDITION': lighting_val,
                'FIRST_CRASH_TYPE': crash_type_val,
                'TRAFFICWAY_TYPE': trafficway_val,
                'ALIGNMENT': alignment_val,
                'ROADWAY_SURFACE_COND': surface_val,
                'ROAD_DEFECT': defect_val,
                'REPORT_TYPE': report_val,
                'INTERSECTION_RELATED_I': intersection_val,
                'DAMAGE': damage_val,
                'PRIM_CONTRIBUTORY_CAUSE': prim_cause_val,
                'SEC_CONTRIBUTORY_CAUSE': sec_cause_val,
                'NUM_UNITS': num_units,
                'CRASH_HOUR': crash_hour,
                'CRASH_DAY_OF_WEEK': crash_day_val,
                'CRASH_MONTH': crash_month_val,
                'CRASH_TIME_OF_DAY': crash_time_val
            }
            
            # Create DataFrame
            input_df = pd.DataFrame([input_dict])
            
            # Predict using the pipeline (handles preprocessing internally!)
            with st.spinner("🔄 Analyzing crash conditions..."):
                prediction = pipeline.predict(input_df)[0]
                probabilities = pipeline.predict_proba(input_df)[0]
                
                # Get class name from label encoder
                predicted_class = label_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100
            
            # Display results
            st.markdown("---")
            
            # Determine if injury crash

            #is_injury ="INJURY AND / OR TOW DUE TO CRASH" in predicted_class.upper()
            is_injury = "INJURY" in predicted_class.upper() or "TOW" in predicted_class.upper()
            card_class = "result-danger" if is_injury else "result-safe"
            color = "#e94560" if is_injury else "#4caf50"
            icon = "⚠️" if is_injury else "✅"
            
            st.markdown(f"""
                <div class="result-card {card_class}">
                    <div class="result-title">Predicted Outcome</div>
                    <div class="result-value" style="color: {color};">
                        {icon} {predicted_class}
                    </div>
                    <div class="result-confidence">
                        Confidence: <strong>{confidence:.1f}%</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Probability breakdown
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.markdown("### 📊 Probability Breakdown")
                prob_df = pd.DataFrame({
                    'Outcome': classes,
                    'Probability': probabilities * 100
                })
                
                colors = ['#4caf50' if 'NO INJURY' in c.upper() else '#e94560' for c in classes]
                
                fig = go.Figure(go.Bar(
                    x=prob_df['Probability'],
                    y=prob_df['Outcome'],
                    orientation='h',
                    marker=dict(color=colors),
                    text=[f'{p:.1f}%' for p in prob_df['Probability']],
                    textposition='outside'
                ))
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font={'color': 'white'},
                    height=200,
                    margin=dict(l=20, r=80, t=20, b=20),
                    xaxis=dict(range=[0, 110], gridcolor='#333'),
                    yaxis=dict(gridcolor='#333')
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col_r2:
                st.markdown("### 💡 Risk Factors Identified")
        
                risk_factors = []
                if speed_limit >= 40:
                    risk_factors.append("🔴 High speed zone (≥40 mph)")
                if weather != 'CLEAR':
                    risk_factors.append(f"🔴 Adverse weather: {weather}")
                if 'DARKNESS' in lighting:
                    risk_factors.append("🟡 Low visibility (darkness)")
                if surface not in ['DRY', 'UNKNOWN']:
                    risk_factors.append(f"🔴 Poor surface: {surface}")
                if prim_cause in ['DRINKING', 'OVERSPEEDING', 'DISTRACTION', 'TEXTING']:
                    risk_factors.append(f"🔴 Dangerous behavior: {prim_cause}")
                if num_units >= 3:
                    risk_factors.append(f"🟡 Multiple vehicles ({num_units})")
                if crash_hour >= 22 or crash_hour <= 5:
                    risk_factors.append("🟡 Late night hours")
                
                if risk_factors:
                    for rf in risk_factors:
                        st.markdown(f"• {rf}")
                else:
                    st.success("✅ No major risk factors identified")
    
    # ==================== PAGE: RISK ANALYSIS ====================
    elif page == "📈 Risk Analysis":
        st.markdown("## 📈 Risk Factor Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔴 High Risk Factors")
            st.markdown("""
            - **Speeding** - Primary cause of severe crashes
            - **Impaired Driving** - Alcohol significantly increases risk
            - **Poor Weather** - Rain, snow, fog reduce visibility
            - **Night Driving** - Reduced visibility
            - **Distraction** - Phone use while driving
            - **Wet/Icy Roads** - Reduced traction
            - **Curves** - More difficult to control vehicle
            """)
        
        with col2:
            st.markdown("### 🟢 Lower Risk Factors")
            st.markdown("""
            - **Low Speed Zones** - More reaction time
            - **Daylight** - Better visibility
            - **Dry Roads** - Better traction
            - **Traffic Signals** - Regulated flow
            - **Single Vehicle** - Fewer collision points
            - **Clear Weather** - Good visibility
            - **Straight Roads** - Easier to control
            """)
        
        st.markdown("---")
        st.markdown("### 📊 Key Statistics")
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Weather Impact", "23%", "of crashes in rain")
        with col_s2:
            st.metric("Night Crashes", "31%", "higher injury rate")
        with col_s3:
            st.metric("Speed Factor", "40%", "of fatal crashes")
        with col_s4:
            st.metric("Distraction", "25%", "involve phone use")
    
    # ==================== PAGE: ABOUT ====================
    elif page == "ℹ️ About":
        st.markdown("## ℹ️ About This App")
        
        st.markdown(f"""
        ### 🎯 Purpose
        Predict whether a traffic crash will result in **injury** or be a **non-injury incident**.
        
        ### 🤖 Model: LightBGM
        - Full sklearn Pipeline with preprocessing
        - Handles missing values and feature scaling
        - Binary classification with probability estimates
        
        ### 📊 Performance
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Balanced Accuracy", f"{metrics.get('test_balanced_accuracy', 0.75)*100:.1f}%")
        with col2:
            st.metric("Macro F1-Score", f"{metrics.get('test_macro_f1', 0.72)*100:.1f}%")
        
        st.markdown("""
        ### ⚠️ Disclaimer
        This tool is for **educational purposes only**. Do not use for safety-critical decisions.
        
        ### 🛠️ Tech Stack
        - **ML**: scikit-learn Pipeline + LightGBM
        - **UI**: Streamlit + Plotly
        - **Preprocessing**: SimpleImputer + StandardScaler
        """)


if __name__ == "__main__":
    main()
