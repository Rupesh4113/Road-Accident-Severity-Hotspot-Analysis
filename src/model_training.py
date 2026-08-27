import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier

def balance_classes_smote(X_train, y_train):
    """Applies SMOTE to balance minority classes in the training data."""
    print("Applying SMOTE to balance classes...")
    class_counts = np.bincount(y_train)
    print(f"Class counts before SMOTE: {dict(enumerate(class_counts))}")
    
    # Verify that we have enough samples for SMOTE k_neighbors
    # Default is k_neighbors=5. If the smallest class has fewer than 6 samples,
    # we need to adjust k_neighbors or drop extremely rare classes.
    min_samples = np.min([count for count in class_counts if count > 0])
    
    if min_samples < 6:
        k_neighbors = max(1, min_samples - 1)
        print(f"Adjusting SMOTE k_neighbors to {k_neighbors} because of small class sizes.")
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    else:
        smote = SMOTE(random_state=42)
        
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    class_counts_after = np.bincount(y_train_res)
    print(f"Class counts after SMOTE: {dict(enumerate(class_counts_after))}")
    return X_train_res, y_train_res

def train_random_forest(X_train, y_train, dataset_name="dataset"):
    """Trains a Random Forest classifier."""
    print(f"Training Random Forest Classifier on {dataset_name}...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf

def train_catboost(X_train, y_train, dataset_name="dataset"):
    """Trains a CatBoost classifier."""
    print(f"Training CatBoost Classifier on {dataset_name}...")
    # CatBoost works well out-of-the-box. We use CPU training.
    # We set iterations=300 to train reasonably fast but with good accuracy.
    cb = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, 
                            random_state=42, verbose=50)
    cb.fit(X_train, y_train)
    return cb

def evaluate_model(model, X_test, y_test, class_names, model_name="Model", dataset_name="UK"):
    """Evaluates the model, prints classification reports, and saves plots."""
    print(f"\n--- Evaluation for {model_name} on {dataset_name} Accidents ---")
    preds = model.predict(X_test)
    
    # Handle CatBoost outputs which can be shape (N, 1) instead of (N,)
    if len(preds.shape) > 1 and preds.shape[1] == 1:
        preds = preds.squeeze()
        
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1-Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=class_names))
    
    # Generate and save Confusion Matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} Confusion Matrix - {dataset_name} Accidents")
    plt.ylabel("Actual Severity")
    plt.xlabel("Predicted Severity")
    plt.tight_layout()
    
    output_cm_path = os.path.join("outputs", f"{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}_cm.png")
    plt.savefig(output_cm_path)
    plt.close()
    print(f"Saved confusion matrix plot to: {output_cm_path}")
    
    return preds, acc, f1

def analyze_and_plot_feature_importance(model, feature_names, model_name="Model", dataset_name="UK"):
    """Extracts, displays, and plots the top feature importances of the model."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        print(f"Feature importance not supported for {model_name}.")
        return None
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    print(f"\nTop 10 Feature Importances for {model_name} ({dataset_name} Accidents):")
    print(df_imp.head(10).to_string(index=False))
    
    # Plot top 15 features
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=df_imp.head(15), hue="Feature", legend=False, palette="viridis")
    plt.title(f"Top 15 Feature Importances - {model_name} ({dataset_name})")
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    output_fi_path = os.path.join("outputs", f"{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}_fi.png")
    plt.savefig(output_fi_path)
    plt.close()
    print(f"Saved feature importance plot to: {output_fi_path}")
    
    return df_imp

def save_trained_model(model, filename):
    """Saves the trained model to models/ directory."""
    path = os.path.join("models", filename)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")
