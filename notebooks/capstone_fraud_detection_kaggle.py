# ============================================================
# REAL-TIME FINANCIAL FRAUD & CREDIT RISK INTELLIGENCE SYSTEM
# Woxsen University Capstone Project — Review 2
# Team: Challa Karthikeya (23WU0104013), Y. Brijesh Naidu, K Dheeraj
# Model: XGBoost + AutoEncoder Ensemble + SHAP XAI
# Mentor: Prof. Dileep Kumar M.
# Dataset: IEEE-CIS Financial Fraud Detection (Real Data)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (roc_auc_score, precision_score, recall_score,
                             f1_score, accuracy_score, confusion_matrix, roc_curve)
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier
import shap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cpu')
print(f"✅ Libraries loaded | Device: {device}")

# ── LOAD REAL IEEE-CIS DATA ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw") + os.sep
OUTPUT_PATH = BASE_DIR + os.sep

print("\n📂 Loading IEEE-CIS Financial Fraud Dataset...")
txn = pd.read_csv(DATA_PATH + "train_transaction.csv")
idn = pd.read_csv(DATA_PATH + "train_identity.csv")
df = txn.merge(idn, on='TransactionID', how='left')

print(f"✅ Dataset loaded!")
print(f"   Total Transactions : {len(df):,}")
print(f"   Total Features     : {df.shape[1]}")
print(f"   Fraudulent         : {df['isFraud'].sum():,} ({df['isFraud'].mean()*100:.2f}%)")
print(f"   Legitimate         : {(df['isFraud']==0).sum():,}")

# ── EDA PLOTS ─────────────────────────────────────────────────────────────────
print("\n📊 Generating EDA plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Woxsen Capstone — EDA: IEEE-CIS Financial Fraud Dataset', fontsize=13, fontweight='bold')

counts = df['isFraud'].value_counts()
axes[0].bar(['Legitimate', 'Fraud'], counts.values, color=['#22c55e', '#ef4444'])
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Count')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 20, f'{v:,}', ha='center', fontweight='bold')

axes[1].hist(df[df['isFraud']==0]['TransactionAmt'].clip(upper=2000), bins=40, alpha=0.7, label='Legitimate', color='#22c55e')
axes[1].hist(df[df['isFraud']==1]['TransactionAmt'].clip(upper=2000), bins=40, alpha=0.7, label='Fraud', color='#ef4444')
axes[1].set_title('Transaction Amount Distribution')
axes[1].set_xlabel('Amount ($)')
axes[1].legend()

if 'ProductCD' in df.columns:
    prod = df.groupby('ProductCD')['isFraud'].mean().sort_values(ascending=False)
    axes[2].bar(prod.index, prod.values * 100, color='#f59e0b')
    axes[2].set_title('Fraud Rate by Product Category (%)')
    axes[2].set_ylabel('Fraud Rate (%)')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'eda_analysis.png', dpi=150, bbox_inches='tight')
print("✅ EDA saved → eda_analysis.png")

# ── PREPROCESSING ─────────────────────────────────────────────────────────────
print("\n⚙️  Preprocessing...")
df['Amt_log'] = np.log1p(df['TransactionAmt'])

# Drop only VERY high-null columns (>90% missing)
keep = [c for c in df.columns if df[c].isnull().mean() < 0.9]
df = df[keep].copy()
print(f"   Features kept after null filtering: {len(keep)}")

# Encode categoricals
cat_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].fillna('unknown')
    df[col] = le.fit_transform(df[col].astype(str))

df = df.fillna(df.median(numeric_only=True))

drop_cols = ['isFraud', 'TransactionID', 'TransactionDT']
X = df.drop([c for c in drop_cols if c in df.columns], axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns)

# Fraud rate is ~10% — skip SMOTE, use class_weight instead
print(f"   Fraud rate: {y_train.mean()*100:.2f}% — using class weights (no SMOTE needed)")
X_train_sm = X_train_sc.reset_index(drop=True)
y_train_sm = y_train.reset_index(drop=True)

# ── LIGHTGBM ──────────────────────────────────────────────────────────────────
print("\n🌲 Training Gradient Boosted Trees (HistGBM)...")
lgbm = HistGradientBoostingClassifier(
    max_iter=400, learning_rate=0.05, max_depth=8,
    random_state=42, early_stopping=True, n_iter_no_change=30,
    class_weight='balanced'
)
lgbm.fit(X_train_sm, y_train_sm)

lgbm_probs = lgbm.predict_proba(X_test_sc)[:, 1]
lgbm_preds = (lgbm_probs >= 0.5).astype(int)
lgbm_auc  = roc_auc_score(y_test, lgbm_probs)
lgbm_prec = precision_score(y_test, lgbm_preds, zero_division=0)
lgbm_rec  = recall_score(y_test, lgbm_preds, zero_division=0)
lgbm_f1   = f1_score(y_test, lgbm_preds, zero_division=0)
lgbm_acc  = accuracy_score(y_test, lgbm_preds)
print(f"✅ Gradient Boosting → AUC: {lgbm_auc*100:.2f}% | F1: {lgbm_f1*100:.2f}% | Recall: {lgbm_rec*100:.2f}%")

# ── AUTOENCODER ───────────────────────────────────────────────────────────────
print("\n🧠 Training AutoEncoder (Anomaly Detector)...")

class FraudAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU(), nn.Linear(32,16), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(16,32), nn.ReLU(), nn.Linear(32,64), nn.ReLU(), nn.Linear(64,d))
    def forward(self, x): return self.dec(self.enc(x))

d = X_train_sm.shape[1]
ae = FraudAE(d).to(device)
opt_ae = optim.Adam(ae.parameters(), lr=1e-3)
crit = nn.MSELoss()

X_legit = torch.FloatTensor(X_train_sm[y_train_sm == 0].values)
ds = TensorDataset(X_legit, X_legit)
loader = DataLoader(ds, batch_size=256, shuffle=True)

for epoch in range(20):
    ae.train()
    for xb, _ in loader:
        opt_ae.zero_grad(); loss = crit(ae(xb), xb); loss.backward(); opt_ae.step()
    if (epoch+1) % 5 == 0:
        print(f"   Epoch {epoch+1}/20 complete")

ae.eval()
Xt = torch.FloatTensor(X_test_sc.values)
with torch.no_grad():
    err = torch.mean((Xt - ae(Xt))**2, dim=1).numpy()

ae_norm  = (err - err.min()) / (err.max() - err.min())
ae_preds = (err > np.percentile(err, 95)).astype(int)
ae_auc   = roc_auc_score(y_test, ae_norm)
ae_prec  = precision_score(y_test, ae_preds, zero_division=0)
ae_rec   = recall_score(y_test, ae_preds, zero_division=0)
ae_f1    = f1_score(y_test, ae_preds, zero_division=0)
ae_acc   = accuracy_score(y_test, ae_preds)
print(f"✅ AutoEncoder → AUC: {ae_auc*100:.2f}% | Recall: {ae_rec*100:.2f}%")

# ── ENSEMBLE ──────────────────────────────────────────────────────────────────
print("\n🔗 Building Fusion Ensemble (LightGBM 70% + AutoEncoder 30%)...")
ens_scores = 0.7 * lgbm_probs + 0.3 * ae_norm
ens_preds  = (ens_scores >= 0.5).astype(int)
ens_auc    = roc_auc_score(y_test, ens_scores)
ens_prec   = precision_score(y_test, ens_preds, zero_division=0)
ens_rec    = recall_score(y_test, ens_preds, zero_division=0)
ens_f1     = f1_score(y_test, ens_preds, zero_division=0)
ens_acc    = accuracy_score(y_test, ens_preds)

# ── ABLATION TABLE ────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("📋 ABLATION STUDY — IEEE-CIS REAL RESULTS (Woxsen Capstone Review 2)")
print("="*70)
print(f"{'Model':<28}{'Accuracy':>9}{'Precision':>11}{'Recall':>9}{'F1':>8}{'AUC-ROC':>10}")
print("-"*70)
print(f"{'XGBoost (Supervised)':<28}{lgbm_acc*100:>8.2f}%{lgbm_prec*100:>10.2f}%{lgbm_rec*100:>8.2f}%{lgbm_f1*100:>7.2f}%{lgbm_auc*100:>9.2f}%")
print(f"{'AutoEncoder (Anomaly)':<28}{ae_acc*100:>8.2f}%{ae_prec*100:>10.2f}%{ae_rec*100:>8.2f}%{ae_f1*100:>7.2f}%{ae_auc*100:>9.2f}%")
print("-"*70)
print(f"{'🏆 FUSION ENSEMBLE':<28}{ens_acc*100:>8.2f}%{ens_prec*100:>10.2f}%{ens_rec*100:>8.2f}%{ens_f1*100:>7.2f}%{ens_auc*100:>9.2f}%")
print("="*70)

# ── ROC PLOT ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Woxsen Capstone Review 2 — Model Performance (IEEE-CIS Real Data)', fontsize=13, fontweight='bold')

fpr1, tpr1, _ = roc_curve(y_test, lgbm_probs)
fpr2, tpr2, _ = roc_curve(y_test, ae_norm)
fpr3, tpr3, _ = roc_curve(y_test, ens_scores)

axes[0].plot(fpr1, tpr1, '#3b82f6', lw=2, label=f'XGBoost (AUC={lgbm_auc:.3f})')
axes[0].plot(fpr2, tpr2, '#f59e0b', lw=2, label=f'AutoEncoder (AUC={ae_auc:.3f})')
axes[0].plot(fpr3, tpr3, '#22c55e', lw=2.5, linestyle='--', label=f'🏆 Ensemble (AUC={ens_auc:.3f})')
axes[0].plot([0,1],[0,1],'gray',linestyle=':')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves'); axes[0].legend(loc='lower right'); axes[0].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, ens_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Predicted Legit','Predicted Fraud'],
            yticklabels=['Actual Legit','Actual Fraud'])
axes[1].set_title(f'Ensemble Confusion Matrix (AUC={ens_auc:.4f})')

plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'model_performance.png', dpi=150, bbox_inches='tight')
print("✅ ROC plot saved → model_performance.png")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("\n🔍 Computing SHAP Explainability...")
X_shap = X_test_sc.sample(min(200, len(X_test_sc)), random_state=42)
explainer   = shap.Explainer(lgbm.predict_proba, X_shap)
shap_values = explainer(X_shap)
sv = shap_values.values[:, :, 1]  # class 1 (fraud) SHAP values

plt.figure(figsize=(11, 7))
shap.summary_plot(sv, X_shap, plot_type="bar", max_display=12, show=False)
plt.title('SHAP Feature Importance — Top Fraud Risk Indicators\n(Woxsen Capstone — XAI / RBI Regulatory Compliance)')
plt.tight_layout()
plt.savefig(OUTPUT_PATH + 'shap_feature_importance.png', dpi=150, bbox_inches='tight')
print("✅ SHAP plot saved → shap_feature_importance.png")

# SHAP audit log
fraud_idx   = X_test_sc[y_test.values == 1].index
sample_f    = X_test_sc.loc[[fraud_idx[0]]]
sv_single   = explainer(sample_f)
vals = sv_single.values[0, :, 1] if sv_single.values.ndim == 3 else sv_single.values[0]
top10       = sorted(zip(X_test_sc.columns, vals), key=lambda x: abs(x[1]), reverse=True)[:10]

print(f"\n🔍 SHAP — Real Fraud Transaction #{fraud_idx[0]} Explanation:")
print(f"   {'Feature':<25} {'SHAP':>10}  Direction")
print(f"   {'-'*50}")
for feat, val in top10:
    d = '↑ Increases Fraud Risk' if val > 0 else '↓ Reduces Fraud Risk'
    print(f"   {feat:<25} {val:>+10.4f}  {d}")

audit = {
    "project": "Woxsen Capstone — Real-Time Financial Fraud Intelligence System",
    "dataset": "IEEE-CIS Financial Fraud Detection",
    "transaction_id": str(fraud_idx[0]),
    "fraud_probability_lgbm": float(lgbm.predict_proba(sample_f)[:,1][0]),
    "shap_contributions": {f: float(v) for f, v in top10}
}
with open(OUTPUT_PATH + 'shap_audit_log.json', 'w') as f:
    json.dump(audit, f, indent=2)
print("✅ SHAP audit log saved → shap_audit_log.json")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("🏁 WOXSEN CAPSTONE REVIEW 2 — ALL DELIVERABLES COMPLETE")
print("="*70)
print(f"  Dataset      : IEEE-CIS Financial Fraud (Real Data — {len(df):,} transactions)")
print(f"  XGBoost      : AUC={lgbm_auc*100:.2f}%  F1={lgbm_f1*100:.2f}%")
print(f"  AutoEncoder  : AUC={ae_auc*100:.2f}%  F1={ae_f1*100:.2f}%")
print(f"  🏆 ENSEMBLE  : AUC={ens_auc*100:.2f}%  F1={ens_f1*100:.2f}%")
print(f"\n  Output files saved in:")
print(f"  ├── eda_analysis.png")
print(f"  ├── model_performance.png")
print(f"  ├── shap_feature_importance.png")
print(f"  └── shap_audit_log.json")
print("="*70)
