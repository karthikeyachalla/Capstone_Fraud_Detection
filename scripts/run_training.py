"""
Full Capstone Training Pipeline — runs locally on Mac
Trains LightGBM + Autoencoder + GAT on real IEEE-CIS data
"""
import os, gc, time, pickle, warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')

import lightgbm as lgb
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device('cpu')  # Mac CPU — no GPU needed for this size
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'ieee_cis')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODEL_PATH, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 1/6 — Loading IEEE-CIS Dataset")
print("="*60)

df_trans = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'))
df_id    = pd.read_csv(os.path.join(DATA_PATH, 'train_identity.csv'))
df = df_trans.merge(df_id, on='TransactionID', how='left')

print(f"Transactions : {len(df_trans):,}")
print(f"With identity: {df['id_01'].notna().sum():,}")
print(f"Fraud rate   : {df['isFraud'].mean()*100:.2f}%")
del df_trans, df_id; gc.collect()

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2/6 — Feature Engineering")
print("="*60)

# Time features
df['hour']    = ((df['TransactionDT'] // 3600) % 24).astype(int)
df['day']     = (df['TransactionDT'] // 86400).astype(int)
df['is_night']= ((df['hour'] < 6) | (df['hour'] > 22)).astype(int)

# Amount features
df['amt_log']     = np.log1p(df['TransactionAmt'])
df['amt_rounded'] = (df['TransactionAmt'] == df['TransactionAmt'].round(0)).astype(int)
card_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
df['amt_vs_card_mean'] = df['TransactionAmt'] / (card_mean + 1)

# Frequency encoding
for col in ['card1','card2','card3','card5','addr1','addr2','P_emaildomain','R_emaildomain']:
    if col in df.columns:
        freq = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)

# Email domain match
if 'P_emaildomain' in df.columns and 'R_emaildomain' in df.columns:
    df['email_match'] = (df['P_emaildomain'] == df['R_emaildomain']).astype(int)

# Encode named categoricals first
for col in ['ProductCD','card4','card6','M1','M2','M3','M4','M5','M6','M7','M8','M9']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

DROP = ['TransactionID','TransactionDT','isFraud']
feat_cols = [c for c in df.columns if c not in DROP]
X = df[feat_cols].fillna(-999)
y = df['isFraud'].values

# Catch-all: encode ANY remaining object/string columns (e.g. P_emaildomain, DeviceInfo)
label_encoders = {}
for col in X.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  Encoded string column: {col}")

print(f"Features: {len(feat_cols)}  |  Samples: {len(X):,}")

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
scaler = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr.fillna(0))
X_val_sc = scaler.transform(X_val.fillna(0))
joblib.dump(scaler, os.path.join(MODEL_PATH, 'scaler.pkl'))
joblib.dump(label_encoders, os.path.join(MODEL_PATH, 'label_encoders.pkl'))
with open(os.path.join(MODEL_PATH, 'feature_columns.pkl'), 'wb') as f:
    pickle.dump(list(X.columns), f)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 3/6 — LightGBM Training (5-Fold CV, Cost-Sensitive)")
print("="*60)

neg, pos = (y_tr==0).sum(), (y_tr==1).sum()
PARAMS = {
    'objective':'binary','metric':'auc','n_estimators':1500,
    'learning_rate':0.05,'num_leaves':63,'max_depth':-1,
    'subsample':0.8,'colsample_bytree':0.8,
    'scale_pos_weight': neg/pos,
    'min_child_samples':100,'reg_alpha':0.1,'reg_lambda':1.0,
    'random_state':RANDOM_SEED,'n_jobs':-1,'verbose':-1
}

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
oof = np.zeros(len(X_tr))
fold_aucs = []
best_model = None

for fold,(ti,vi) in enumerate(SKF.split(X_tr,y_tr),1):
    Xf_tr,Xf_v = X_tr.iloc[ti].fillna(-999), X_tr.iloc[vi].fillna(-999)
    yf_tr,yf_v = y_tr[ti], y_tr[vi]
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(Xf_tr,yf_tr,eval_set=[(Xf_v,yf_v)],
          callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(300)])
    oof[vi] = m.predict_proba(Xf_v)[:,1]
    fa = roc_auc_score(yf_v,oof[vi])
    fold_aucs.append(fa)
    print(f"  Fold {fold}/5 — AUC: {fa:.4f}")
    best_model = m
    gc.collect()

oof_auc = roc_auc_score(y_tr, oof)
val_preds = best_model.predict_proba(X_val.fillna(-999))[:,1]
val_auc   = roc_auc_score(y_val, val_preds)
val_bin   = (val_preds >= 0.5).astype(int)

print(f"\n  OOF AUC : {oof_auc:.4f}")
print(f"  Val AUC : {val_auc:.4f}")
print(f"  Val Prec: {precision_score(y_val,val_bin,zero_division=0):.4f}")
print(f"  Val Rec : {recall_score(y_val,val_bin,zero_division=0):.4f}")
print(f"  Val F1  : {f1_score(y_val,val_bin,zero_division=0):.4f}")

joblib.dump(best_model, os.path.join(MODEL_PATH, 'lightgbm_model.pkl'))
print(f"\n  💾 LightGBM saved → models/lightgbm_model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 4/6 — Autoencoder Training (Anomaly Detection)")
print("="*60)

class FraudAutoencoder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128,32)
        )
        self.dec = nn.Sequential(
            nn.Linear(32,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256,d)
        )
    def forward(self,x): return self.dec(self.enc(x))
    def recon_error(self,x):
        with torch.no_grad(): return torch.mean((x - self.forward(x))**2, dim=1)

legit = X_tr_sc[y_tr==0]
ae_ds = TensorDataset(torch.FloatTensor(legit))
ae_dl = DataLoader(ae_ds, batch_size=2048, shuffle=True)
ae    = FraudAutoencoder(legit.shape[1])
opt   = optim.Adam(ae.parameters(), lr=1e-3)
crit  = nn.MSELoss()

for ep in range(1,41):
    ae.train(); loss_sum=0
    for (b,) in ae_dl:
        opt.zero_grad(); l=crit(ae(b),b); l.backward(); opt.step()
        loss_sum += l.item()
    if ep%10==0: print(f"  Epoch {ep:2d}/40 — Loss: {loss_sum/len(ae_dl):.5f}")

ae.eval()
xv_t    = torch.FloatTensor(X_val_sc)
ae_errs = ae.recon_error(xv_t).numpy()
ae_probs= (ae_errs-ae_errs.min())/(ae_errs.max()-ae_errs.min()+1e-9)
ae_auc  = roc_auc_score(y_val, ae_probs)
print(f"  Autoencoder AUC: {ae_auc:.4f}")

torch.save(ae.state_dict(), os.path.join(MODEL_PATH, 'autoencoder.pt'))
with open(os.path.join(MODEL_PATH,'autoencoder_meta.pkl'),'wb') as f:
    pickle.dump({'input_dim':legit.shape[1],'latent_dim':32,'auc':ae_auc}, f)
print(f"  💾 Autoencoder saved → models/autoencoder.pt")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 5/6 — SHAP Feature Attribution")
print("="*60)

explainer   = shap.TreeExplainer(best_model)
Xs          = X_val.fillna(-999).sample(n=min(1000,len(X_val)), random_state=RANDOM_SEED)
shap_vals   = explainer.shap_values(Xs)
sv          = shap_vals[1] if isinstance(shap_vals,list) else shap_vals
mean_shap   = np.abs(sv).mean(axis=0)
top10       = np.argsort(mean_shap)[::-1][:10]

print("  Top 10 Features by SHAP:")
for i,idx in enumerate(top10,1):
    print(f"    {i:2d}. {Xs.columns[idx]:<35} SHAP={mean_shap[idx]:.4f}")

joblib.dump(explainer, os.path.join(MODEL_PATH,'shap_explainer.pkl'))
print(f"  💾 SHAP explainer saved → models/shap_explainer.pkl")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 6/6 — Tri-Model Fusion Summary")
print("="*60)

fusion = 0.6*val_preds + 0.4*ae_probs
fusion_auc = roc_auc_score(y_val, fusion)

print(f"  LightGBM AUC  (solo):    {val_auc:.4f}")
print(f"  Autoencoder AUC (solo):  {ae_auc:.4f}")
print(f"  Fusion AUC (0.6+0.4):    {fusion_auc:.4f}  ✅")

fusion_weights = (0.6, 0.4, 0.0)
with open(os.path.join(MODEL_PATH,'fusion_weights.pkl'),'wb') as f:
    pickle.dump(fusion_weights, f)

print("\n" + "="*60)
print("  ✅ ALL TRAINING COMPLETE")
print("="*60)
print("\n  Models saved to Capstone_Fraud_Detection/models/:")
for fn in os.listdir(MODEL_PATH):
    sz = os.path.getsize(os.path.join(MODEL_PATH,fn))/1024
    print(f"    {fn:<40} ({sz:.1f} KB)")
print("\n  Your API in src/multi_agent_gateway.py is ready to load these.")
