import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import custom modules
from src.downloader import setup_directories, download_uk_data, check_or_generate_us_data
from src.data_preprocessing import preprocess_uk_data, preprocess_us_data, split_and_prep_data
from src.model_training import (
    balance_classes_smote, train_random_forest, train_catboost,
    evaluate_model, analyze_and_plot_feature_importance, save_trained_model
)
from src.spatial_analysis import perform_kde_and_find_hotspots, generate_interactive_map

def run_uk_pipeline(uk_file_path):
    print("\n=========================================")
    print("RUNNING PIPELINE FOR UK ROAD SAFETY DATA")
    print("=========================================")
    
    # 1. Preprocessing
    X, y, coords = preprocess_uk_data(uk_file_path)
    
    # 2. Split
    X_train, X_test, y_train, y_test = split_and_prep_data(X, y)
    
    # 3. SMOTE
    X_train_res, y_train_res = balance_classes_smote(X_train, y_train)
    
    # 4. Train Models
    # Classes: 0=Fatal (1), 1=Serious (2), 2=Slight (3)
    class_names = ["Fatal", "Serious", "Slight"]
    
    # Random Forest
    rf_model = train_random_forest(X_train_res, y_train_res, "UK")
    rf_preds, rf_acc, rf_f1 = evaluate_model(rf_model, X_test, y_test, class_names, "Random Forest", "UK")
    analyze_and_plot_feature_importance(rf_model, X.columns, "Random Forest", "UK")
    save_trained_model(rf_model, "uk_random_forest.joblib")
    
    # CatBoost
    cb_model = train_catboost(X_train_res, y_train_res, "UK")
    cb_preds, cb_acc, cb_f1 = evaluate_model(cb_model, X_test, y_test, class_names, "CatBoost", "UK")
    analyze_and_plot_feature_importance(cb_model, X.columns, "CatBoost", "UK")
    save_trained_model(cb_model, "uk_catboost.joblib")
    
    # 5. Spatial Analysis
    coords_sample, hotspots = perform_kde_and_find_hotspots(coords)
    map_path = generate_interactive_map(coords_sample, hotspots, "uk_hotspots.html", "UK")
    
    return {
        "RF": {"accuracy": rf_acc, "f1_macro": rf_f1},
        "CatBoost": {"accuracy": cb_acc, "f1_macro": cb_f1},
        "map_path": map_path
    }

def run_us_pipeline(us_file_path, is_sample):
    print("\n=========================================")
    print(f"RUNNING PIPELINE FOR US ACCIDENTS DATA {'(SAMPLE)' if is_sample else ''}")
    print("=========================================")
    
    # 1. Preprocessing
    X, y, coords = preprocess_us_data(us_file_path)
    
    # 2. Split
    X_train, X_test, y_train, y_test = split_and_prep_data(X, y)
    
    # 3. SMOTE
    X_train_res, y_train_res = balance_classes_smote(X_train, y_train)
    
    # 4. Train Models
    # Classes: Severity 1 to 4 -> 0 to 3 index
    class_names = ["Severity 1", "Severity 2", "Severity 3", "Severity 4"]
    
    # Check what classes are actually present after splitting
    present_classes = np.sort(np.unique(y_test))
    eval_class_names = [class_names[c] for c in present_classes]
    
    # Random Forest
    rf_model = train_random_forest(X_train_res, y_train_res, "US")
    rf_preds, rf_acc, rf_f1 = evaluate_model(rf_model, X_test, y_test, eval_class_names, "Random Forest", "US")
    analyze_and_plot_feature_importance(rf_model, X.columns, "Random Forest", "US")
    save_trained_model(rf_model, "us_random_forest.joblib")
    
    # CatBoost
    cb_model = train_catboost(X_train_res, y_train_res, "US")
    cb_preds, cb_acc, cb_f1 = evaluate_model(cb_model, X_test, y_test, eval_class_names, "CatBoost", "US")
    analyze_and_plot_feature_importance(cb_model, X.columns, "CatBoost", "US")
    save_trained_model(cb_model, "us_catboost.joblib")
    
    # 5. Spatial Analysis
    coords_sample, hotspots = perform_kde_and_find_hotspots(coords)
    map_path = generate_interactive_map(coords_sample, hotspots, "us_hotspots.html", "US")
    
    return {
        "RF": {"accuracy": rf_acc, "f1_macro": rf_f1},
        "CatBoost": {"accuracy": cb_acc, "f1_macro": cb_f1},
        "map_path": map_path
    }

def main():
    print("Initializing Road Accident Severity & Hotspot Analysis Pipeline...")
    
    # Set up folders
    setup_directories()
    
    # Check/retrieve data
    uk_path = download_uk_data()
    us_path, is_sample = check_or_generate_us_data()
    
    # Execute pipelines
    uk_results = run_uk_pipeline(uk_path)
    us_results = run_us_pipeline(us_path, is_sample)
    
    # Print final summary
    print("\n" + "="*50)
    print("ROAD ACCIDENT ANALYSIS PIPELINE SUMMARY")
    print("="*50)
    print("\n[UK DATASET RESULTS]")
    print(f"Random Forest - Accuracy: {uk_results['RF']['accuracy']:.4f}, Macro F1: {uk_results['RF']['f1_macro']:.4f}")
    print(f"CatBoost      - Accuracy: {uk_results['CatBoost']['accuracy']:.4f}, Macro F1: {uk_results['CatBoost']['f1_macro']:.4f}")
    print(f"Interactive Map Saved to: {uk_results['map_path']}")
    
    print("\n[US DATASET RESULTS]")
    print(f"Random Forest - Accuracy: {us_results['RF']['accuracy']:.4f}, Macro F1: {us_results['RF']['f1_macro']:.4f}")
    print(f"CatBoost      - Accuracy: {us_results['CatBoost']['accuracy']:.4f}, Macro F1: {us_results['CatBoost']['f1_macro']:.4f}")
    print(f"Interactive Map Saved to: {us_results['map_path']}")
    
    print("\nAll evaluation figures (confusion matrices & feature importances) have been saved to the 'outputs/' directory.")
    print("Trained models have been saved to the 'models/' directory.")
    print("="*50)

if __name__ == "__main__":
    main()
