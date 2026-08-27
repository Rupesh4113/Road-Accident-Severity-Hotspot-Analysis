import os
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from scipy.stats import gaussian_kde

def perform_kde_and_find_hotspots(coords, n_sample=5000, quantile=0.98):
    """
    Performs Kernel Density Estimation on lat/long coordinates.
    Returns the coordinates, their density values, and the top 'quantile' hotspot coordinates.
    """
    print("Performing Geospatial Kernel Density Estimation (KDE)...")
    
    # Sample data if it's too large to prevent memory/performance issues
    if len(coords) > n_sample:
        coords_sample = coords.sample(n=n_sample, random_state=42).copy()
    else:
        coords_sample = coords.copy()
        
    lats = coords_sample["latitude"].values
    lngs = coords_sample["longitude"].values
    
    # Run 2D gaussian KDE
    # scipy.stats.gaussian_kde expects shape (2, N)
    points = np.vstack([lngs, lats])
    
    try:
        kde = gaussian_kde(points)
        # Compute density value for each point in the sample
        densities = kde(points)
        coords_sample["density"] = densities
        
        # Identify hotspots (top quantile of density)
        threshold = coords_sample["density"].quantile(quantile)
        hotspots = coords_sample[coords_sample["density"] >= threshold]
        print(f"Identified {len(hotspots)} hotspot points at threshold {threshold:.4f} (top {(1-quantile)*100}% density).")
        return coords_sample, hotspots
    except Exception as e:
        print(f"Error performing KDE: {e}")
        # Return empty/fallback dataframes in case of singular matrix issues
        coords_sample["density"] = 1.0
        return coords_sample, coords_sample.head(10)

def generate_interactive_map(coords, hotspots, output_filename, dataset_name="UK"):
    """
    Generates an interactive Folium map showing accident points as a HeatMap
    and marking the highest density Hotspots.
    """
    print(f"Generating interactive Folium map for {dataset_name} accidents...")
    
    # Center map on the mean of coordinates
    mean_lat = coords["latitude"].mean()
    mean_lng = coords["longitude"].mean()
    
    # Create base map
    m = folium.Map(location=[mean_lat, mean_lng], zoom_start=6 if dataset_name == "UK" else 4, 
                   tiles="OpenStreetMap")
    
    # Prepare data for HeatMap (list of [lat, lng, weight])
    heat_data = coords[["latitude", "longitude"]].dropna().values.tolist()
    
    # Add HeatMap layer
    HeatMap(heat_data, radius=15, blur=10, name="Accident Heatmap").add_to(m)
    
    # Group hotspots into clusters/representative zones to avoid marker spam
    # We can cluster them or simply display a small set of top hotspots (e.g. top 50)
    top_hotspots = hotspots.sort_values(by="density", ascending=False).head(50)
    
    # Create a FeatureGroup for Hotspots
    hotspot_group = folium.FeatureGroup(name="Top Hotspot Centers")
    
    for idx, row in top_hotspots.iterrows():
        lat = row["latitude"]
        lng = row["longitude"]
        dens = row["density"]
        
        # Add circle marker for each hotspot
        folium.CircleMarker(
            location=[lat, lng],
            radius=8,
            popup=f"Hotspot Zone<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}<br>Density: {dens:.4f}",
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.6,
            weight=1
        ).add_to(hotspot_group)
        
    hotspot_group.add_to(m)
    folium.LayerControl().add_to(m)
    
    # Save map
    output_path = os.path.join("outputs", output_filename)
    m.save(output_path)
    print(f"Interactive map saved to: {output_path}")
    return output_path
