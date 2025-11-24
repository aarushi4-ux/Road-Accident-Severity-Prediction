# Road Accident Severity Prediction 

This repository contains our end-to-end **Road Accident Severity Prediction** project built as part of the Foundations of Data Science coursework. The goal of this project is to build a reliable machine learning pipeline that predicts the severity of a traffic crash and identifies high-risk areas using geospatial intelligence.

---

## 📂 Dataset

We use the **Chicago Crash Dataset**, which contains detailed reports of road accidents, including severity, environmental conditions, vehicle details, and locational information.

🔗 **Dataset link:** *[https://www.kaggle.com/datasets/nathaniellybrand/chicago-car-crash-dataset]*

---

## 🔄 Project Workflow

### **Data Preprocessing**
- Data cleaning and validation
- Handling missing values and outliers
- Feature engineering and encoding
- Train-test split with stratification
- **Output:** Clean, model-ready dataset

### **Model Training**
- **Algorithm:** LightGBM with SMOTE
- **Approach:** Binary classification (Injury vs. No Injury)
- **Pipeline:** Preprocessing + SMOTE + LightGBM classifier
- **Optimization:** Stratified K-fold cross-validation
- **Class Imbalance:** Addressed using SMOTE oversampling and balanced class weights

### **Model Explainability**
Understanding model predictions through:

**SHAP (SHapley Additive exPlanations):**
- Global feature importance
- Individual prediction explanations
- Feature interaction analysis
- Waterfall plots for decision transparency

**LIME (Local Interpretable Model-agnostic Explanations):**
- Local approximations of model behavior
- Instance-specific feature contributions
- Human-interpretable explanations

### **Geospatial Analysis**
Comprehensive spatial risk assessment pipeline:

**a) Crash Point Geometry**
- Validation of latitude/longitude coordinates
- Conversion to GeoDataFrame
- Export as GeoJSON for spatial operations

**b) Reverse Geocoding**
- Spatial join with postal area (POA) polygons
- Assignment of POA codes and names to crash points
- Geographical aggregation of crash data

**c) Hotspot Detection (DBSCAN)**
- Haversine-distance based clustering
- Identification of spatial crash clusters
- Noise point detection (outlier crashes)
- Cluster labeling and characterization

**d) Risk Score Computation**
Severity-weighted risk scoring system:
```
Risk Score = (3 × Killed + 2 × Serious + 1 × Moderate + 0.5 × Minor) / Total Crashes
```
Generated outputs:
- ZIP/Postcode risk tables
- Cluster risk tables
- Area-level risk rankings

**e) Interactive Visualizations**

**Choropleth Maps:**
- Color-coded risk intensity by region
- Smooth Viridis color gradient
- Interactive hover information
- Clear severity shading

**Hotspot Maps:**
- Multi-layer visualization with:
  - Raw crash points
  - DBSCAN clusters (size and color by severity)
  - Heatmap overlay
  - Postcode boundaries
  - Layer toggle controls

**Interactive Dashboard:**
- Real-time metric selection
- Configurable crash count filters
- Dynamic color scale adjustment
- Live Mapbox choropleth updates
- Comprehensive crash risk explorer

### **User Interface**
**Streamlit Web Application:**
- Interactive prediction interface
- Risk factor identification
- Feature importance visualization

---

## 📈 Model Performance

<!-- REPLACE WITH YOUR ACTUAL METRICS -->

| Metric | Value |
|--------|-------|
| **Balanced Accuracy** | 71.8% |
| **Macro F1-Score** | 80.6% |

---
## 👥 Contributors

This project was developed as part of a Foundations of Data Science group project.

| Name | GitHub | Contribution |
|------|--------|--------------|
| **Shreiya** | [@Shreiya-Muthuvelan](#) | Data preprocessing and Streamlit UI development |
| **Sanya** |[@sanya28wd](#) | Modelling using LightGBM |
| **Aarushi** | [@aarushi4-ux](#) | Model explainability (SHAP/LIME), documentation |
| **Chirudeva** | [@Tactical-Camell](#) | Geospatial analysis, hotspot detection, risk scoring |


