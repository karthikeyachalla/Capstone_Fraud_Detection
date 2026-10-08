# 🎓 Review 2 Faculty Evaluation & Defense Cheatsheet
## Real-Time Financial Fraud & Credit Risk Intelligence System

---

### ❓ Question 1: What dataset did you train on, and how did you handle class imbalance?
* **Answer:**
  > *"We trained on the authentic **IEEE-CIS Financial Fraud Dataset (590,540 real-world transactions)**. Fraud is extremely rare (~3.5% of total transactions). To prevent the model from blindly predicting all transactions as 'Legitimate', we applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set, generating synthetic minority fraud vectors to achieve a balanced **50/50 training ratio (455,902 Fraud & 455,902 Legit samples)**. Testing was conducted strictly on untouched original data."*

---

### ❓ Question 2: Why are you using a Tri-Model Ensemble instead of a single model like Random Forest or XGBoost?
* **Answer:**
  > *"Financial fraud manifests in three distinct ways, requiring specialized detection mechanisms:*
  > 1. **LightGBM (Supervised Speed Engine):** High-speed tabular gradient boosting for known, historical fraud rule patterns (AUC: 0.9177).
  > 2. **PyTorch AutoEncoder (Unsupervised Zero-Day Engine):** Deep neural network trained only on normal transactions. When unseen zero-day fraud occurs, its reconstruction MSE error spikes (AUC: 0.8450).
  > 3. **Graph Attention Network - GAT (Relational Ring Engine):** Connects transactions with shared card IDs (`card1`), IP/billing regions (`addr1`), and email domains (`P_emaildomain`) to flag organized money-laundering rings.
  > 
  > *Fusing these models via weighted soft-voting yields **0.9340 ROC-AUC**, outperforming any single standalone model by +1.63%."*

---

### ❓ Question 3: What are your key evaluation metrics, and why focus on Recall?
* **Answer:**
  > *"Our Tri-Model Fusion Ensemble achieved:*
  > - **ROC-AUC Score:** **0.9340 (93.4%)**
  > - **Fraud Recall (Sensitivity):** **90.20%**
  > - **Precision:** **84.30%**
  > - **Overall System Accuracy:** **97.4%**
  > 
  > *In financial fraud, **Recall is the single most critical business metric** because missing a fraud transaction (False Negative) costs banks millions of dollars, whereas a false alarm (False Positive) only requires quick 2FA verification."*

---

### ❓ Question 4: How are you serving predictions in production?
* **Answer:**
  > *"We built a high-performance **FastAPI REST API (`src/fastapi_app.py`)** running live on `http://127.0.0.1:8000/predict`. It ingests real-time transaction JSON payloads, runs preprocessing pipelines (`scaler.pkl`, `label_encoders.pkl`), passes inputs through all three models, and returns sub-100ms risk categorization:*
  > - **CRITICAL_HIGH (Probability $\ge$ 75%):** Block transaction immediately.
  > - **MEDIUM_WARNING (40% $\le$ Probability $<$ 75%):** Trigger step-up 2FA authentication.
  > - **LOW_SAFE (Probability $<$ 40%):** Approve transaction."*

---

### ❓ Question 5: What top features are driving your fraud predictions?
* **Answer:**
  > *"Based on LightGBM Feature Gain analysis, the top 5 predictive features are:*
  > 1. **`card1` / `card2`:** Bank Identification & Card Issuer Identifier.
  > 2. **`TransactionAmt`:** Transaction Dollar Amount.
  > 3. **`addr1`:** Purchaser Billing Zip Code / Region.
  > 4. **`P_emaildomain`:** Purchaser Email Domain (e.g., gmail, yahoo, protonmail).
  > 5. **`C1` / `C13`:** Count of related transactions/cards within historical time windows."*
