import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

# Script basic training buat nyoba-nyoba awal (autolog)
# Dataset: Rice (Cammeo & Osmancik)
TRAIN_PATH = "namadataset_preprocessing/rice_preprocessing_train.csv"
TEST_PATH = "namadataset_preprocessing/rice_preprocessing_test.csv"

def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']
    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']
    
    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")

    # Set Experiment
    mlflow.set_experiment("Rice_Classification_Basic")

    # Autologging
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest_Autolog"):
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\nAccuracy: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

    print("\n✅ Basic Rice model trained and logged to MLflow!")

if __name__ == "__main__":
    train_model()
