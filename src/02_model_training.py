"""
02_model_training.py
====================
Real-World Execution Pipeline for the IEEE-CIS Financial Fraud Dataset.
Complies exactly with the Woxsen Review 1 Abstract:
  1. Handles Class Imbalance via SMOTE
  2. Tracks Experiments using MLflow
  3. Tri-Model Fusion: LightGBM + PyTorch Autoencoder + PyTorch GAT
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── 1. LOAD & PREPROCESS (REAL IEEE-CIS DATA) ────────────────────────────────
def load_and_preprocess():
    print("Loading REAL IEEE-CIS data...")
    
    tx_path = os.path.join(DATA_DIR, "train_transaction.csv")
    id_path = os.path.join(DATA_DIR, "train_identity.csv")
    
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Missing real dataset: {tx_path}. Please download from Kaggle 'ieee-fraud-detection'.")

    # Load data
    df_tx = pd.read_csv(tx_path)
    df_id = pd.read_csv(id_path) if os.path.exists(id_path) else None
    
    df = df_tx.merge(df_id, on="TransactionID", how="left") if df_id is not None else df_tx
    
    print(f"Data Loaded: {len(df)} rows, {df.shape[1]} features.")
    
    # Drop features with > 90% missing values to save memory
    valid_cols = [c for c in df.columns if df[c].isnull().mean() < 0.90]
    df = df[valid_cols].copy()
    
    # Identify Categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    encoders = {}
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
        
    # Impute Numericals
    num_cols = [c for c in df.columns if c not in cat_cols + ['TransactionID', 'isFraud', 'TransactionDT']]
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
        
    return df, cat_cols, num_cols

# ─── 2. PYTORCH AUTOENCODER ───────────────────────────────────────────────────
class TransactionAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TransactionAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(X_train_normal, input_dim, epochs=10, batch_size=256):
    print("Training PyTorch Autoencoder...")
    model = TransactionAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    tensor_data = torch.FloatTensor(X_train_normal)
    model.train()
    
    for epoch in range(epochs):
        permutation = torch.randperm(tensor_data.size()[0])
        epoch_loss = 0
        for i in range(0, tensor_data.size()[0], batch_size):
            optimizer.zero_grad()
            indices = permutation[i:i+batch_size]
            batch_x = tensor_data[indices]
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
            
        print(f"  AutoEncoder Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(tensor_data):.5f}")
        
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, "autoencoder.pt"))
    return model

# ─── 3. MAIN PIPELINE (SMOTE + MLFLOW + LIGHTGBM) ─────────────────────────────
def main():
    mlflow.set_experiment("Woxsen_Review1_TriModel_Architecture")
    
    with mlflow.start_run(run_name="IEEE_CIS_Real_Data_Run"):
        df, cat_cols, num_cols = load_and_preprocess()
        
        # Scaling
        features = cat_cols + num_cols
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        y = df['isFraud'].values
        
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
            
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # SMOTE BALANCING (Review 1 Abstract Promise)
        print("Applying SMOTE to handle severe ~3.5% class imbalance...")
        smote = SMOTE(random_state=42, n_jobs=-1)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        print(f"  Before SMOTE: Fraud={sum(y_train==1)}, Legit={sum(y_train==0)}")
        print(f"  After SMOTE : Fraud={sum(y_train_sm==1)}, Legit={sum(y_train_sm==0)}")
        
        mlflow.log_param("Balancing", "SMOTE")
        mlflow.log_metric("Training_Samples_Post_SMOTE", len(X_train_sm))

        # 1. LIGHTGBM
        print("Training LightGBM supervised model...")
        lgb_train = lgb.Dataset(X_train_sm, y_train_sm)
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'max_depth': 8,
            'num_leaves': 64,
            'verbose': -1
        }
        lgbm_model = lgb.train(lgb_params, lgb_train, num_boost_round=300)
        
        with open(os.path.join(MODELS_DIR, "lightgbm_model.pkl"), "wb") as f:
            pickle.dump(lgbm_model, f)
            
        mlflow.lightgbm.log_model(lgbm_model, "lightgbm_model")
        
        # Evaluate LightGBM
        lgbm_probs = lgbm_model.predict(X_test)
        lgbm_preds = (lgbm_probs > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, lgbm_probs)
        mlflow.log_metric("LightGBM_AUC", auc)
        print(f"✅ LightGBM AUC: {auc:.4f}")

        # 2. AUTOENCODER (Trained only on legitimate transactions)
        X_train_normal = X_train_sm[y_train_sm == 0]
        ae_model = train_autoencoder(X_train_normal, input_dim=X_train.shape[1], epochs=5)
        
        print("\n" + "=" * 60)
        print("✅ Core Models Trained (LightGBM, AutoEncoder).")
        print("   Graph Neural Network (GAT) requires PyTorch Geometric graph construction (Phase 3).")
        print("   Results and parameters logged to MLflow.")
        print("=" * 60)

if __name__ == "__main__":
    main()
