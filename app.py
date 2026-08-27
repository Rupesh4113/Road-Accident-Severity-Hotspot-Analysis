import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="Road Accident Severity & Hotspot Analyzer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🚗 Road Accident Severity & Hotspot Analyzer")
st.markdown("""
This interactive web dashboard allows you to predict the severity of traffic incidents using trained Machine Learning models (Random Forest and CatBoost), examine the spatial factors behind them, and explore geographical accident hotspots.
""")

# Load models and configurations
@st.cache_resource
def load_model_assets():
    assets = {}
    model_paths = {
        "uk_rf": "models/uk_random_forest.joblib",
        "uk_cb": "models/uk_catboost.joblib",
        "uk_scaler": "models/uk_scaler.joblib",
        "uk_features": "models/uk_features.joblib",
        "us_rf": "models/us_random_forest.joblib",
        "us_cb": "models/us_catboost.joblib",
        "us_scaler": "models/us_scaler.joblib",
        "us_features": "models/us_features.joblib",
    }
    for key, path in model_paths.items():
        if os.path.exists(path):
            assets[key] = joblib.load(path)
        else:
            assets[key] = None
    return assets

assets = load_model_assets()

# Verify assets are loaded
assets_loaded = all(v is not None for v in assets.values())
if not assets_loaded:
    st.warning("⚠️ Some model files are missing. Please run the model pipeline `python main.py` first to generate models and scalers.")

# Sidebar Configuration
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to Page:", ["Severity Predictor", "Interactive Hotspot Maps", "Model Insights & Diagnostics"])

# ----------------- PAGE 1: SEVERITY PREDICTOR -----------------
if app_mode == "Severity Predictor":
    st.header("⚡ Real-Time Severity Predictor")
    st.markdown("Select a dataset and input the accident details to calculate severity probabilities.")
    
    # Dataset selection
    dataset_choice = st.selectbox("Choose Target Region & Schema:", ["United Kingdom (STATS19 Schema)", "United States (Kaggle Schema)"])
    
    if dataset_choice == "United Kingdom (STATS19 Schema)":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Temporal & Scale Factors")
            speed_limit = st.selectbox("Speed Limit (mph):", [20, 30, 40, 50, 60, 70], index=1)
            num_vehicles = st.slider("Number of Vehicles Involved:", 1, 10, 2)
            num_casualties = st.slider("Number of Casualties:", 1, 10, 1)
            hour = st.slider("Hour of Day:", 0, 23, 12)
            month = st.slider("Month of Year:", 1, 12, 6)
            day_of_week_str = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            day_of_week = day_map[day_of_week_str]
            
        with col2:
            st.subheader("Environmental & Spatial Geometry")
            weather = st.selectbox("Weather Conditions:", [
                "Fine_No_Wind", "Raining_No_Wind", "Snowing_No_Wind", 
                "Fine_High_Wind", "Raining_High_Wind", "Snowing_High_Wind", 
                "Fog_or_Mist", "Other", "Unknown"
            ])
            light = st.selectbox("Light Conditions:", [
                "Daylight", "Darkness_Lights_Lit", "Darkness_Lights_Unlit", 
                "Darkness_No_Lighting", "Darkness_Lighting_Unknown"
            ])
            surface = st.selectbox("Road Surface Conditions:", [
                "Dry", "Wet_or_Damp", "Snow", "Frost_or_Ice", 
                "Flood_Deep", "Oil_or_Diesel", "Mud"
            ])
            road_type_desc = st.selectbox("Road Type:", [
                "Single_Carriageway", "Dual_Carriageway", "Roundabout", 
                "One_Way_Street", "Slip_Road", "Unknown"
            ])
            urban_rural = st.selectbox("Urban or Rural Area:", ["Urban", "Rural"])
            
            model_type = st.radio("Classification Model:", ["CatBoost (Recommended)", "Random Forest"])
            
        if st.button("🔮 Predict UK Incident Severity", type="primary"):
            if not assets["uk_rf"] or not assets["uk_cb"] or not assets["uk_scaler"] or not assets["uk_features"]:
                st.error("UK model assets are not loaded. Please run the training pipeline.")
            else:
                # 1. Build raw input row
                raw_data = {
                    "speed_limit": speed_limit,
                    "number_of_vehicles": num_vehicles,
                    "number_of_casualties": num_casualties,
                    "month": month,
                    "day_of_week": day_of_week,
                    "hour": hour
                }
                
                # 2. Scale numeric columns
                numeric_input = np.array([[speed_limit, num_vehicles, num_casualties]])
                scaled_numeric = assets["uk_scaler"].transform(numeric_input)[0]
                
                raw_data["speed_limit"] = scaled_numeric[0]
                raw_data["number_of_vehicles"] = scaled_numeric[1]
                raw_data["number_of_casualties"] = scaled_numeric[2]
                
                # 3. Handle one-hot encoding columns manually based on feature list
                features_list = assets["uk_features"]
                input_df = pd.DataFrame(columns=features_list)
                input_df.loc[0] = 0.0 # initialize with zeros
                
                # Map numeric inputs
                for k, v in raw_data.items():
                    if k in input_df.columns:
                        input_df.loc[0, k] = v
                        
                # Map categorical inputs
                cat_mappings = {
                    f"weather_{weather}": 1.0,
                    f"light_{light}": 1.0,
                    f"surface_{surface}": 1.0,
                    f"road_type_desc_{road_type_desc}": 1.0,
                    f"urban_rural_{urban_rural}": 1.0
                }
                for col_name, val in cat_mappings.items():
                    if col_name in input_df.columns:
                        input_df.loc[0, col_name] = val
                
                # Ensure correct column order
                input_df = input_df[features_list]
                
                # 4. Predict
                model = assets["uk_cb"] if model_type == "CatBoost (Recommended)" else assets["uk_rf"]
                probs = model.predict_proba(input_df)[0]
                pred_idx = np.argmax(probs)
                
                # Mapped back (0=Fatal, 1=Serious, 2=Slight)
                classes = ["Fatal (Severity 1)", "Serious (Severity 2)", "Slight (Severity 3)"]
                colors = ["danger", "warning", "info"]
                
                st.write("---")
                st.subheader("Prediction Results")
                st.markdown(f"**Predicted Class**: :{colors[pred_idx]}[{classes[pred_idx]}]")
                
                # Display bar charts
                prob_df = pd.DataFrame({
                    "Severity Class": ["Fatal", "Serious", "Slight"],
                    "Probability (%)": [p * 100 for p in probs]
                })
                st.bar_chart(prob_df, x="Severity Class", y="Probability (%)", horizontal=True)

    elif dataset_choice == "United States (Kaggle Schema)":
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Scale & Atmospheric Inputs")
            distance = st.number_input("Incident Affected Distance (mi):", min_value=0.0, max_value=50.0, value=0.1, step=0.1)
            temp = st.number_input("Temperature (F):", min_value=-40.0, max_value=130.0, value=65.0)
            humidity = st.slider("Humidity (%):", 0, 100, 60)
            visibility = st.slider("Visibility (mi):", 0, 10, 10)
            weather_us = st.selectbox("Weather Condition:", [
                "Clear", "Fair", "Cloudy", "Mostly Cloudy", "Partly Cloudy", 
                "Overcast", "Light Rain", "Rain", "Light Snow", "Fog", "Other"
            ])
            hour_us = st.slider("Hour of Day:", 0, 23, 8)
            month_us = st.slider("Month of Year:", 1, 12, 3)
            day_of_week_str_us = st.selectbox("Day of Week:", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            day_of_week_us = day_map[day_of_week_str_us]
            
            model_type_us = st.radio("Classification Model:", ["CatBoost (Recommended)", "Random Forest"])
            
        with col2:
            st.subheader("Road Geometry & Points of Interest")
            crossing = st.checkbox("Crossing (Pedestrian/Street)")
            junction = st.checkbox("Junction / Highway Intersection")
            traffic_signal = st.checkbox("Traffic Signal / Stoplight")
            stop = st.checkbox("Stop Sign")
            station = st.checkbox("Transit Station / Bus Stop")
            amenity = st.checkbox("Amenity (Buildings/Facilities nearby)")
            bump = st.checkbox("Speed Bump / Traffic Calming Device")
            give_way = st.checkbox("Give Way / Yield Sign")
            no_exit = st.checkbox("No Exit / Dead End Sign")
            railway = st.checkbox("Railway Crossing")
            roundabout = st.checkbox("Roundabout / Traffic Circle")
            traffic_calming = st.checkbox("Traffic Calming Features")
            
        if st.button("🔮 Predict US Incident Severity", type="primary"):
            if not assets["us_rf"] or not assets["us_cb"] or not assets["us_scaler"] or not assets["us_features"]:
                st.error("US model assets are not loaded. Please run the training pipeline.")
            else:
                # 1. Build raw input row
                raw_data = {
                    "month": month_us,
                    "day_of_week": day_of_week_us,
                    "hour": hour_us,
                    "Crossing": int(crossing),
                    "Junction": int(junction),
                    "Traffic_Signal": int(traffic_signal),
                    "Station": int(station),
                    "Stop": int(stop),
                    "Amenity": int(amenity),
                    "Bump": int(bump),
                    "Give_Way": int(give_way),
                    "No_Exit": int(no_exit),
                    "Railway": int(railway),
                    "Roundabout": int(roundabout),
                    "Traffic_Calming": int(traffic_calming),
                    "Turning_Loop": 0
                }
                
                # 2. Scale numeric columns
                numeric_input = np.array([[temp, humidity, visibility, distance]])
                scaled_numeric = assets["us_scaler"].transform(numeric_input)[0]
                
                raw_data["Temperature(F)"] = scaled_numeric[0]
                raw_data["Humidity(%)"] = scaled_numeric[1]
                raw_data["Visibility(mi)"] = scaled_numeric[2]
                raw_data["Distance(mi)"] = scaled_numeric[3]
                
                # 3. Handle one-hot encoding columns manually based on feature list
                features_list = assets["us_features"]
                input_df = pd.DataFrame(columns=features_list)
                input_df.loc[0] = 0.0 # initialize with zeros
                
                # Map numeric inputs
                for k, v in raw_data.items():
                    if k in input_df.columns:
                        input_df.loc[0, k] = v
                        
                # Map categorical inputs
                cat_col = f"weather_clean_{weather_us}"
                if cat_col in input_df.columns:
                    input_df.loc[0, cat_col] = 1.0
                
                # Ensure correct column order
                input_df = input_df[features_list]
                
                # 4. Predict
                model = assets["us_cb"] if model_type_us == "CatBoost (Recommended)" else assets["us_rf"]
                probs = model.predict_proba(input_df)[0]
                
                # Handle output shape mapping
                classes = ["Severity 1 (Low Impact)", "Severity 2", "Severity 3", "Severity 4 (Max Impact)"]
                colors = ["info", "success", "warning", "danger"]
                
                pred_idx = np.argmax(probs)
                
                st.write("---")
                st.subheader("Prediction Results")
                st.markdown(f"**Predicted Class**: :{colors[pred_idx]}[{classes[pred_idx]}]")
                st.write("*(Note: US Severity represents the impact of the accident on traffic flow)*")
                
                # Display bar charts
                prob_df = pd.DataFrame({
                    "US Traffic Severity": ["Severity 1", "Severity 2", "Severity 3", "Severity 4"],
                    "Probability (%)": [p * 100 for p in probs]
                })
                st.bar_chart(prob_df, x="US Traffic Severity", y="Probability (%)", horizontal=True)

# ----------------- PAGE 2: INTERACTIVE HOTSPOT MAPS -----------------
elif app_mode == "Interactive Hotspot Maps":
    st.header("🗺️ Interactive Geospatial Hotspot Maps")
    st.markdown("These maps plot accident coordinates as a heatmap and highlight the top 2% densest accident clusters (hotspots) calculated using 2D Kernel Density Estimation (KDE).")
    
    map_choice = st.selectbox("Choose Area to Explore:", ["United Kingdom (London & Great Britain)", "United States (Clustered Sample)"])
    
    if map_choice == "United Kingdom (London & Great Britain)":
        map_path = "outputs/uk_hotspots.html"
        title_text = "UK Accident Hotspots Map"
    else:
        map_path = "outputs/us_hotspots.html"
        title_text = "US Accident Hotspots Map"
        
    if os.path.exists(map_path):
        st.subheader(title_text)
        st.caption("Double click to zoom, hover/click on the red markers to inspect hotspot coordinates and density values.")
        
        # Read HTML and render in iframe
        with open(map_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=650, scrolling=True)
    else:
        st.error(f"Map file `{map_path}` was not found. Please run the model pipeline `python main.py` first to generate the Folium maps.")

# ----------------- PAGE 3: MODEL INSIGHTS & DIAGNOSTICS -----------------
elif app_mode == "Model Insights & Diagnostics":
    st.header("📊 Model Insights & Performance Metrics")
    st.markdown("Examine the metrics, confusion matrices, and risk factors isolated by the Random Forest and CatBoost models.")
    
    dataset_insight = st.selectbox("Select Target Region:", ["UK Accidents (STATS19)", "US Accidents"])
    model_insight = st.selectbox("Select Model:", ["CatBoost", "Random Forest"])
    
    region_key = "uk" if dataset_insight == "UK Accidents (STATS19)" else "us"
    model_key = "catboost" if model_insight == "CatBoost" else "random_forest"
    
    cm_img_path = f"outputs/{region_key}_{model_key}_cm.png"
    fi_img_path = f"outputs/{region_key}_{model_key}_fi.png"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        if os.path.exists(cm_img_path):
            st.image(cm_img_path, caption=f"Confusion Matrix for {model_insight} on {dataset_insight}", use_container_width=True)
        else:
            st.info("Confusion matrix image not found. Ensure the pipeline ran successfully.")
            
    with col2:
        st.subheader("Feature Importance Rankings")
        if os.path.exists(fi_img_path):
            st.image(fi_img_path, caption=f"Top Features for {model_insight} on {dataset_insight}", use_container_width=True)
        else:
            st.info("Feature importance image not found. Ensure the pipeline ran successfully.")
            
    st.markdown("---")
    st.subheader("💡 Key Takeaways & Risk Factors Analysis")
    if region_key == "uk":
        st.markdown("""
        - **Speed Limit** is by far the most dominant spatial risk factor for predicting collision severity in Great Britain. Accidents on higher-speed roads (national speed limit segments like 60-70 mph) are statistically correlated with higher mortality rates and serious injury severity.
        - **Infrastructure & Scale Factors**: The number of vehicles involved and total casualties are critical features. Single-carriageway sections are highly correlated with severity.
        - **Lighting & Lighting Infrastructure**: Darkness without streetlights (`light_Darkness_No_Lighting`) shows high importance, suggesting poor street lighting significantly worsens crash severity.
        """)
    else:
        st.markdown("""
        - **Temporal Patterns**: Time factors (`hour`, `month`, `day_of_week`) are highly predictive of traffic congestion severity in the US. Peak commute hours (7-9 AM, 4-6 PM) correlate with high-impact disruptions.
        - **Weather & Environmental Factors**: Visibility, temperature, and humidity represent environmental risk factors. Inclement weather conditions like heavy rain or dense fog reduce visibility, increasing congestion and severe delays.
        - **Points of Interest**: Features like `Traffic_Signal` and `Junction` are highly correlated with accidents, confirming intersections as significant risk hotspots.
        """)
