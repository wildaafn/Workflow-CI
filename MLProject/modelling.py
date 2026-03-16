"""
modelling.py

Script untuk melatih model Rice Classification
menggunakan MLflow Project.

Author: Wilda Ariffatul Faisalnur
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import mlflow
import mlflow.sklearn
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data():
    """Memuat dataset yang sudah dipreprocessing."""
    train_data = pd.read_csv('namadataset_preprocessing/rice_preprocessing_train.csv')
    test_data = pd.read_csv('namadataset_preprocessing/rice_preprocessing_test.csv')

    X_train = train_data.drop('Class', axis=1)
    y_train = train_data['Class']
    X_test = test_data.drop('Class', axis=1)
    y_test = test_data['Class']

    return X_train, X_test, y_train, y_test


def main():
    """Fungsi utama untuk training model dalam MLflow Project."""
    X_train, X_test, y_train, y_test = load_preprocessed_data()

    print(f"Data Beras: Train {X_train.shape}, Test {X_test.shape}")

    # Hyperparameter tuning yang lebih ringan biar cepat
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10]
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Prediksi
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # MLflow Manual Logging
    with mlflow.start_run(run_name="Rice_Workflow_Training"):
        # Log parameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Log model
        mlflow.sklearn.log_model(best_model, "model")

        # Artifacts
        os.makedirs("artifacts", exist_ok=True)

        # Classification report - Format Indonesia
        report = classification_report(y_test, y_pred)
        with open("artifacts/classification_report.txt", 'w') as f:
            f.write(report)
        mlflow.log_artifact("artifacts/classification_report.txt")

        # Feature importance - Sesuai Fitur Beras
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices], color='mediumseagreen')
        plt.xticks(range(len(importances)),
                   [X_train.columns[i] for i in indices],
                   rotation=45, ha='right')
        plt.title('Fitur Paling Berpengaruh (Rice Dataset)')
        plt.tight_layout()
        plt.savefig("artifacts/feature_importance.png", dpi=150)
        plt.close()
        mlflow.log_artifact("artifacts/feature_importance.png")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
        plt.title('Confusion Matrix - Klasifikasi Beras')
        plt.xlabel('Prediksi')
        plt.ylabel('Aktual')
        plt.tight_layout()
        plt.savefig("artifacts/confusion_matrix.png", dpi=150)
        plt.close()
        mlflow.log_artifact("artifacts/confusion_matrix.png")

        # Tags
        mlflow.set_tag("author", "Wilda Ariffatul Faisalnur")
        mlflow.set_tag("dataset", "Rice (Cammeo and Osmancik)")

        print(f"\nAkurasi: {accuracy:.4f}")
        print(f"F1: {f1:.4f}")
        print("\n✅ Training model beras selesai!")


if __name__ == "__main__":
    main()
