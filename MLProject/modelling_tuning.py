import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub

# --- Konfigurasi Lokasi File ---
# Saya simpan di folder preprocessing biar rapi sesuai instruksi
TRAIN_PATH = "namadataset_preprocessing/rice_preprocessing_train.csv"
TEST_PATH = "namadataset_preprocessing/rice_preprocessing_test.csv"

def load_data():
    """Fungsi buat ambil data training sama testing"""
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    # Pisahin fitur sama targetnya (Class)
    X_train = train_df.drop(columns=['Class'])
    y_train = train_df['Class']
    X_test = test_df.drop(columns=['Class'])
    y_test = test_df['Class']
    
    return X_train, X_test, y_train, y_test

def train_with_tuning():
    """Main function buat training + hyperparameter tuning"""
    X_train, X_test, y_train, y_test = load_data()
    print(f"Data ready. Train: {X_train.shape}, Test: {X_test.shape}")

    # Biar bisa dipantau online lewat DagsHub
    dagshub.init(repo_owner='wildaafn', repo_name='Membangun_model', mlflow=True)
    mlflow.set_experiment("Rice_Classification_Tuning")

    with mlflow.start_run(run_name="RandomForest_Tuned_Manual"):
        # 1. Tuning dulu pake GridSearchCV biar hasilnya maksimal
        # Catatan: n_estimators sengaja nggak gede-gede banget biar nggak berat
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        
        print("\n[INFO] Mulai cari parameter terbaik...")
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        print(f"Parameter ketemu: {grid_search.best_params_}")

        # 2. Prediksi & Evaluasi
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log parameter manual biar rapi di dashboard
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        
        print(f"\nHasil: Acc={acc:.4f}, F1={f1:.4f}")

        # 3. Simpan Model
        mlflow.sklearn.log_model(best_model, "model")

        # 4. Bikin visualisasi (Ini penting buat bukti di Dicoding)
        # --- Feature Importance ---
        plt.figure(figsize=(10, 6))
        features = X_train.columns
        importances = best_model.feature_importances_
        indices = np.argsort(importances)
        plt.title('Fitur Paling Berpengaruh (Beras)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # --- Confusion Matrix ---
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - Rice Classification')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # --- Report ---
        with open("classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred))
        mlflow.log_artifact("classification_report.txt")

    print("\n✅ Mantap! Training selesai dan semua log sudah masuk MLflow.")

if __name__ == "__main__":
    train_with_tuning()
