import os
import requests
import pandas as pd
import numpy as np

def setup_directories():
    """Create necessary directories for data, models, and outputs."""
    directories = ["data", "models", "outputs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")

def download_uk_data():
    """Download the UK road safety collision data for 2022 if not already present."""
    uk_url = "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2022.csv"
    dest_path = os.path.join("data", "uk_collisions_2022.csv")
    
    if os.path.exists(dest_path):
        print(f"UK Collision data already exists at: {dest_path}")
        return dest_path

    print(f"Downloading UK collision data from {uk_url}...")
    try:
        response = requests.get(uk_url, stream=True)
        response.raise_for_status()
        
        # Download and write file
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Successfully downloaded UK collision data to {dest_path}")
    except Exception as e:
        print(f"Error downloading UK data: {e}")
        # Fallback: create a small mock UK dataset in case of network issues
        print("Creating fallback mock UK dataset...")
        create_mock_uk_data(dest_path)
        
    return dest_path

def create_mock_uk_data(dest_path):
    """Creates a mock UK dataset in case the download fails."""
    np.random.seed(42)
    n_rows = 5000
    
    # Clustered coordinates around London (51.5074, -0.1278) and Edinburgh (55.9533, -3.1883)
    centers = [(51.5074, -0.1278), (55.9533, -3.1883)]
    coords = []
    for _ in range(n_rows):
        center = centers[np.random.choice([0, 1], p=[0.8, 0.2])]
        lat = center[0] + np.random.normal(0, 0.05)
        lng = center[1] + np.random.normal(0, 0.05)
        coords.append((lat, lng))
        
    df = pd.DataFrame({
        "collision_index": [f"MOCK_UK_{i}" for i in range(n_rows)],
        "collision_year": [2022] * n_rows,
        "collision_ref_no": [f"REF_{i}" for i in range(n_rows)],
        "location_easting_osgr": np.random.randint(400000, 600000, n_rows),
        "location_northing_osgr": np.random.randint(100000, 300000, n_rows),
        "longitude": [c[1] for c in coords],
        "latitude": [c[0] for c in coords],
        "police_force": np.random.choice([1, 12, 17, 30, 45], n_rows),
        "collision_severity": np.random.choice([1, 2, 3], n_rows, p=[0.02, 0.15, 0.83]),
        "number_of_vehicles": np.random.choice([1, 2, 3, 4], n_rows, p=[0.4, 0.45, 0.12, 0.03]),
        "number_of_casualties": np.random.choice([1, 2, 3], n_rows, p=[0.75, 0.20, 0.05]),
        "date": pd.date_range(start="2022-01-01", end="2022-12-31", periods=n_rows).strftime("%d/%m/%Y"),
        "day_of_week": np.random.randint(1, 8, n_rows),
        "time": [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}" for _ in range(n_rows)],
        "local_authority_district": np.random.randint(1, 100, n_rows),
        "local_authority_ons_district": [f"E0600000{np.random.randint(1,9)}" for _ in range(n_rows)],
        "local_authority_highway": [f"E0600000{np.random.randint(1,9)}" for _ in range(n_rows)],
        "local_authority_highway_current": [f"E0600000{np.random.randint(1,9)}" for _ in range(n_rows)],
        "first_road_class": np.random.choice([1, 2, 3, 4, 5, 6], n_rows),
        "first_road_number": np.random.randint(0, 1000, n_rows),
        "road_type": np.random.choice([1, 2, 3, 6, 7, 9], n_rows),
        "speed_limit": np.random.choice([20, 30, 40, 50, 60, 70], n_rows, p=[0.1, 0.5, 0.1, 0.05, 0.15, 0.1]),
        "junction_detail": np.random.choice([0, 1, 2, 3, 5, 6, 8, 9], n_rows),
        "junction_control": np.random.choice([-1, 0, 1, 2, 4], n_rows),
        "second_road_class": np.random.choice([-1, 1, 2, 3, 6], n_rows),
        "second_road_number": np.random.randint(-1, 1000, n_rows),
        "pedestrian_crossing": np.random.choice([0, 1, 4, 5, 8], n_rows),
        "light_conditions": np.random.choice([1, 4, 5, 6, 7], n_rows, p=[0.70, 0.20, 0.02, 0.06, 0.02]),
        "weather_conditions": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9], n_rows, p=[0.80, 0.10, 0.02, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02]),
        "road_surface_conditions": np.random.choice([1, 2, 3, 4, 5], n_rows, p=[0.70, 0.25, 0.03, 0.01, 0.01]),
        "special_conditions_at_site": np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], n_rows, p=[0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.003, 0.002]),
        "carriageway_hazards": np.random.choice([0, 1, 2, 3, 6, 7], n_rows, p=[0.97, 0.005, 0.005, 0.005, 0.005, 0.01]),
        "urban_or_rural_area": np.random.choice([1, 2], n_rows, p=[0.6, 0.4]),
        "did_police_officer_attend_scene_of_accident": np.random.choice([1, 2, 3], n_rows),
        "trunk_road_flag": np.random.choice([1, 2], n_rows),
        "lsoa_of_accident_location": [f"E0101206{np.random.randint(1,9)}" for _ in range(n_rows)]
    })
    df.to_csv(dest_path, index=False)
    print(f"Fallback UK dataset created at: {dest_path}")

def check_or_generate_us_data():
    """Checks if a US accidents CSV exists. If not, generates a sample dataset."""
    # Look for files starting with 'US_Accidents' and ending with '.csv' in data directory
    csv_files = [f for f in os.listdir("data") if f.lower().startswith("us_accidents") and f.endswith(".csv")]
    
    # Exclude sample file if there is a real one
    real_csv_files = [f for f in csv_files if "sample" not in f]
    
    if real_csv_files:
        us_path = os.path.join("data", real_csv_files[0])
        print(f"Found existing US Accidents dataset at: {us_path}")
        return us_path, False # returns path and is_sample flag
        
    # Check if sample already exists
    sample_path = os.path.join("data", "us_accidents_sample.csv")
    if os.path.exists(sample_path):
        print(f"Using existing synthetic US Accidents sample at: {sample_path}")
        return sample_path, True
        
    print("US Accidents dataset not found in data/. Generating synthetic US Accidents sample...")
    np.random.seed(42)
    n_rows = 15000
    
    # Cluster locations around major US cities to make geospatial hotspot analysis realistic
    cities = {
        "New York": (40.7128, -74.0060),
        "Los Angeles": (34.0522, -118.2437),
        "Chicago": (41.8781, -87.6298),
        "Houston": (29.7604, -95.3698),
        "Miami": (25.7617, -80.1918)
    }
    
    city_keys = list(cities.keys())
    city_probs = [0.3, 0.25, 0.2, 0.15, 0.1]
    
    coords = []
    for _ in range(n_rows):
        city = city_keys[np.random.choice(len(city_keys), p=city_probs)]
        center_lat, center_lng = cities[city]
        lat = center_lat + np.random.normal(0, 0.08)
        lng = center_lng + np.random.normal(0, 0.08)
        coords.append((lat, lng))
        
    # Generate timestamp ranges
    start_dates = pd.date_range(start="2023-01-01", end="2023-12-31", periods=n_rows)
    durations_min = np.random.exponential(scale=45, size=n_rows) + 15
    end_dates = start_dates + pd.to_timedelta(durations_min, unit="m")
    
    # Severity is heavily imbalanced (mostly 2, some 3, few 1 and 4)
    severity = np.random.choice([1, 2, 3, 4], n_rows, p=[0.05, 0.70, 0.20, 0.05])
    
    # Create DataFrame
    df = pd.DataFrame({
        "ID": [f"A-{i+1}" for i in range(n_rows)],
        "Severity": severity,
        "Start_Time": start_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "End_Time": end_dates.strftime("%Y-%m-%d %H:%M:%S"),
        "Start_Lat": [c[0] for c in coords],
        "Start_Lng": [c[1] for c in coords],
        "Distance(mi)": np.random.exponential(scale=0.5, size=n_rows),
        "Temperature(F)": np.random.normal(loc=65, scale=15, size=n_rows),
        "Humidity(%)": np.clip(np.random.normal(loc=60, scale=20, size=n_rows), 0, 100),
        "Visibility(mi)": np.clip(np.random.normal(loc=9.2, scale=2.0, size=n_rows), 0, 10),
        "Weather_Condition": np.random.choice(
            ["Clear", "Fair", "Cloudy", "Mostly Cloudy", "Partly Cloudy", "Overcast", "Light Rain", "Rain", "Light Snow", "Fog"],
            n_rows,
            p=[0.3, 0.2, 0.15, 0.1, 0.1, 0.05, 0.04, 0.03, 0.01, 0.02]
        ),
        # POI indicators (road geometry)
        "Crossing": np.random.choice([True, False], n_rows, p=[0.08, 0.92]),
        "Junction": np.random.choice([True, False], n_rows, p=[0.12, 0.88]),
        "Traffic_Signal": np.random.choice([True, False], n_rows, p=[0.15, 0.85]),
        "Station": np.random.choice([True, False], n_rows, p=[0.02, 0.98]),
        "Stop": np.random.choice([True, False], n_rows, p=[0.02, 0.98]),
        "Amenity": np.random.choice([True, False], n_rows, p=[0.01, 0.99]),
        "Bump": np.random.choice([True, False], n_rows, p=[0.001, 0.999]),
        "Give_Way": np.random.choice([True, False], n_rows, p=[0.005, 0.995]),
        "No_Exit": np.random.choice([True, False], n_rows, p=[0.001, 0.999]),
        "Railway": np.random.choice([True, False], n_rows, p=[0.01, 0.99]),
        "Roundabout": np.random.choice([True, False], n_rows, p=[0.002, 0.998]),
        "Traffic_Calming": np.random.choice([True, False], n_rows, p=[0.002, 0.998]),
        "Turning_Loop": [False] * n_rows
    })
    
    df.to_csv(sample_path, index=False)
    print(f"Generated synthetic US Accidents dataset at: {sample_path}")
    return sample_path, True

def check_or_generate_bengaluru_data():
    """Checks if a Bengaluru accidents CSV exists. If not, generates a sample dataset."""
    csv_files = [f for f in os.listdir("data") if f.lower().startswith("bengaluru_accidents") and f.endswith(".csv")]
    real_csv_files = [f for f in csv_files if "sample" not in f]
    
    if real_csv_files:
        blr_path = os.path.join("data", real_csv_files[0])
        print(f"Found existing Bengaluru Accidents dataset at: {blr_path}")
        return blr_path, False
        
    sample_path = os.path.join("data", "bengaluru_accidents_sample.csv")
    if os.path.exists(sample_path):
        print(f"Using existing synthetic Bengaluru Accidents sample at: {sample_path}")
        return sample_path, True
        
    print("Bengaluru Accidents dataset not found in data/. Generating synthetic Bengaluru sample...")
    np.random.seed(42)
    n_rows = 10000
    
    # Bengaluru high-accident junctions
    junctions = {
        "Silk Board": (12.9175, 77.6225),
        "Hebbal": (13.0358, 77.5975),
        "Majestic": (12.9766, 77.5726),
        "Tin Factory": (12.9930, 77.6740),
        "Indiranagar": (12.9719, 77.6412),
        "Electronic City": (12.8488, 77.6601)
    }
    
    j_keys = list(junctions.keys())
    j_probs = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
    
    coords = []
    for _ in range(n_rows):
        junc = j_keys[np.random.choice(len(j_keys), p=j_probs)]
        center_lat, center_lng = junctions[junc]
        lat = center_lat + np.random.normal(0, 0.04)
        lng = center_lng + np.random.normal(0, 0.04)
        coords.append((lat, lng))
        
    # Temporal
    dates = pd.date_range(start="2024-01-01", end="2024-12-31", periods=n_rows)
    times = [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}" for _ in range(n_rows)]
    
    # Severity (MORTH schema: 1=Fatal, 2=Grievous, 3=Minor, 4=Non-Injury)
    severity = np.random.choice([1, 2, 3, 4], n_rows, p=[0.05, 0.25, 0.60, 0.10])
    
    df = pd.DataFrame({
        "ID": [f"BLR-{i+1}" for i in range(n_rows)],
        "Severity": severity,
        "Date": dates.strftime("%d/%m/%Y"),
        "Time": times,
        "Latitude": [c[0] for c in coords],
        "Longitude": [c[1] for c in coords],
        "Traffic_Volume": np.clip(np.random.normal(loc=1500, scale=500, size=n_rows), 100, 3000),
        "Weather_Condition": np.random.choice(
            ["Sunny/Clear", "Raining", "Foggy", "Overcast"],
            n_rows,
            p=[0.75, 0.15, 0.05, 0.05]
        ),
        "Road_Condition": np.random.choice(
            ["Dry", "Wet", "Potholes/Damaged", "Under Construction"],
            n_rows,
            p=[0.60, 0.15, 0.20, 0.05]
        ),
        "Speed_Breaker": np.random.choice([1, 0], n_rows, p=[0.10, 0.90]),
        "Traffic_Signal": np.random.choice([1, 0], n_rows, p=[0.20, 0.80]),
        "Junction": np.random.choice([1, 0], n_rows, p=[0.25, 0.75]),
        "Crossing": np.random.choice([1, 0], n_rows, p=[0.12, 0.88])
    })
    
    df.to_csv(sample_path, index=False)
    print(f"Generated synthetic Bengaluru Accidents dataset at: {sample_path}")
    return sample_path, True

if __name__ == "__main__":
    setup_directories()
    download_uk_data()
    check_or_generate_us_data()
    check_or_generate_bengaluru_data()
