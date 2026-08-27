import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def map_uk_categories(df):
    """Map coded integers in UK dataset to descriptive string categories."""
    # Mappings according to STATS19 guide
    weather_map = {
        1: "Fine_No_Wind",
        2: "Raining_No_Wind",
        3: "Snowing_No_Wind",
        4: "Fine_High_Wind",
        5: "Raining_High_Wind",
        6: "Snowing_High_Wind",
        7: "Fog_or_Mist",
        8: "Other",
        9: "Unknown",
        -1: "Missing"
    }
    
    light_map = {
        1: "Daylight",
        4: "Darkness_Lights_Lit",
        5: "Darkness_Lights_Unlit",
        6: "Darkness_No_Lighting",
        7: "Darkness_Lighting_Unknown",
        -1: "Missing"
    }
    
    surface_map = {
        1: "Dry",
        2: "Wet_or_Damp",
        3: "Snow",
        4: "Frost_or_Ice",
        5: "Flood_Deep",
        6: "Oil_or_Diesel",
        7: "Mud",
        -1: "Missing"
    }
    
    road_type_map = {
        1: "Roundabout",
        2: "One_Way_Street",
        3: "Dual_Carriageway",
        6: "Single_Carriageway",
        7: "Slip_Road",
        9: "Unknown",
        -1: "Missing"
    }

    df = df.copy()
    df["weather"] = df["weather_conditions"].map(weather_map).fillna("Unknown")
    df["light"] = df["light_conditions"].map(light_map).fillna("Unknown")
    df["surface"] = df["road_surface_conditions"].map(surface_map).fillna("Unknown")
    df["road_type_desc"] = df["road_type"].map(road_type_map).fillna("Unknown")
    df["urban_rural"] = df["urban_or_rural_area"].map({1: "Urban", 2: "Rural"}).fillna("Unknown")
    
    return df

def preprocess_uk_data(file_path):
    """Load and preprocess the UK road safety dataset."""
    print("Preprocessing UK road safety data...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Drop rows with invalid coordinates
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    df = df[(df["latitude"] > 49) & (df["latitude"] < 61)]  # Approximate bounding box for UK
    df = df[(df["longitude"] > -9) & (df["longitude"] < 2)]
    
    # Map category codes to readable strings
    df = map_uk_categories(df)
    
    # Parse Temporal features
    df["datetime"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    # In case date format is yyyy-mm-dd
    mask = df["datetime"].isna()
    if mask.any():
        df.loc[mask, "datetime"] = pd.to_datetime(df.loc[mask, "date"], errors="coerce")
        
    df = df.dropna(subset=["datetime"])
    
    df["month"] = df["datetime"].dt.month
    df["day_of_week"] = df["datetime"].dt.dayofweek
    
    # Parse time
    df["hour"] = pd.to_datetime(df["time"], format="%H:%M", errors="coerce").dt.hour
    df["hour"] = df["hour"].fillna(df["time"].str.split(":").str[0].astype(float).fillna(12).astype(int))
    df["hour"] = df["hour"].astype(int)
    
    # Select features for classification
    # Target: collision_severity (1=Fatal, 2=Serious, 3=Slight)
    feature_cols = [
        "speed_limit", "number_of_vehicles", "number_of_casualties",
        "month", "day_of_week", "hour", "weather", "light", "surface",
        "road_type_desc", "urban_rural"
    ]
    
    X = df[feature_cols].copy()
    y = df["collision_severity"].copy()
    
    # Track lat/long for KDE before encoding
    coords = df[["latitude", "longitude"]].copy()
    
    # One-hot encode categoricals
    categorical_cols = ["weather", "light", "surface", "road_type_desc", "urban_rural"]
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Convert all columns to numeric, filling NaNs
    X = X.fillna(X.median(numeric_only=True))
    
    # Scale numeric columns
    numeric_cols = ["speed_limit", "number_of_vehicles", "number_of_casualties"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    # Save scaler and feature columns for prediction service
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/uk_scaler.joblib")
    joblib.dump(X.columns.tolist(), "models/uk_features.joblib")
    
    # Target values: map to 0-indexed for modeling (0=Fatal, 1=Serious, 2=Slight)
    y = y - 1
    
    print(f"UK Preprocessing complete. Cleaned shape: {X.shape}, Classes: {np.bincount(y)}")
    return X, y, coords

def preprocess_us_data(file_path):
    """Load and preprocess the US Accidents dataset."""
    print("Preprocessing US Accidents data...")
    # Read CSV, use chunksize or handle low memory flags since US data can be huge
    # Let's inspect the file size first to determine how many rows to read
    # If it is a large file, we can read a subset (e.g. 200k rows) to prevent memory crashes
    file_size_gb = os.path.getsize(file_path) / (1024**3)
    if file_size_gb > 0.5:
        print(f"Real US Accidents dataset is large ({file_size_gb:.2f} GB). Loading first 300,000 rows to prevent memory limits...")
        df = pd.read_csv(file_path, nrows=300000, low_memory=False)
    else:
        df = pd.read_csv(file_path, low_memory=False)
        
    # Drop rows with invalid coordinates
    df = df.dropna(subset=["Start_Lat", "Start_Lng"])
    df = df[(df["Start_Lat"] != 0) & (df["Start_Lng"] != 0)]
    
    # Extract temporal features
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], errors="coerce")
    df = df.dropna(subset=["Start_Time"])
    
    df["month"] = df["Start_Time"].dt.month
    df["day_of_week"] = df["Start_Time"].dt.dayofweek
    df["hour"] = df["Start_Time"].dt.hour
    
    # Convert road feature booleans to 0/1 integers
    bool_cols = [
        "Crossing", "Junction", "Traffic_Signal", "Station", "Stop", 
        "Amenity", "Bump", "Give_Way", "No_Exit", "Railway", 
        "Roundabout", "Traffic_Calming", "Turning_Loop"
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
            
    # Clean Weather_Condition
    df["Weather_Condition"] = df["Weather_Condition"].fillna("Unknown")
    # Simplify weather condition categories (top 10 + Other)
    top_weather = df["Weather_Condition"].value_counts().index[:12]
    df["weather_clean"] = df["Weather_Condition"].apply(lambda x: x if x in top_weather else "Other")
    
    # Numerical features
    num_cols = ["Temperature(F)", "Humidity(%)", "Visibility(mi)", "Distance(mi)"]
    for col in num_cols:
        if col in df.columns:
            # Fill missing numericals with median
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
            
    # Target: Severity (values 1 to 4)
    # Map to 0-indexed (0 to 3)
    df["target"] = df["Severity"].astype(int) - 1
    
    # Bounding box filter for US (approximate)
    df = df[(df["Start_Lat"] > 24) & (df["Start_Lat"] < 50)]
    df = df[(df["Start_Lng"] > -125) & (df["Start_Lng"] < -66)]
    
    # Feature columns list
    feature_cols = [
        "month", "day_of_week", "hour", "weather_clean"
    ] + [b for b in bool_cols if b in df.columns] + [n for n in num_cols if n in df.columns]
    
    X = df[feature_cols].copy()
    y = df["target"].copy()
    
    # Coordinates for KDE
    coords = df[["Start_Lat", "Start_Lng"]].rename(columns={"Start_Lat": "latitude", "Start_Lng": "longitude"}).copy()
    
    # One-hot encode weather
    X = pd.get_dummies(X, columns=["weather_clean"], drop_first=True)
    
    # Fill remaining NaNs
    X = X.fillna(X.median(numeric_only=True))
    
    # Scale numerical columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Save scaler and feature columns for prediction service
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/us_scaler.joblib")
    joblib.dump(X.columns.tolist(), "models/us_features.joblib")
    
    print(f"US Preprocessing complete. Cleaned shape: {X.shape}, Classes: {np.bincount(y)}")
    return X, y, coords

def split_and_prep_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
