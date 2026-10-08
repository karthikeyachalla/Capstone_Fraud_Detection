# Capstone Architecture & Implementation Defense

This document provides the exact technical rationale for your project. If the panel asks you **"Why did you choose this?"**, these are your answers.

## 1. Why the IEEE-CIS Fraud Detection Dataset?
**Panel Question:** *"Why are you using the IEEE-CIS dataset instead of the standard Kaggle Credit Card Fraud dataset?"*

**Your Defense:** 
> "Most public fraud datasets (like the standard Kaggle Credit Card one) are purely numerical, PCA-transformed features (V1, V2, V3, etc.) with all identity and metadata stripped for privacy. 
> 
> We chose the **IEEE-CIS dataset** because it is the *only* major benchmark that contains two linked tables: `Transactions` and `Identity` (Device Info, OS, Email domains). **Without identity linkage, it is impossible to build our PyTorch Graph Attention Network (GAT).** We need the shared device vectors and email domains to construct graph nodes and detect collusive, multi-account fraud rings. The IEEE-CIS dataset is the only one that supports this advanced graph-mining."

---

## 2. Why SMOTE?
**Panel Question:** *"Why are you using SMOTE?"*

**Your Defense:**
> "The IEEE-CIS dataset suffers from a severe class imbalance—fraud only accounts for roughly **3.5%** of the transactions. If we train our models on raw data, they will develop a 'majority class bias' and achieve 96.5% accuracy just by blindly predicting 'Legitimate' every time. 
> 
> We use **SMOTE (Synthetic Minority Over-sampling Technique)** instead of simple duplication. SMOTE interpolates between existing minority samples to generate statistically valid synthetic fraud examples, balancing our training set to 50/50 without causing the model to overfit on identical duplicated records."

---

## 3. Why the Tri-Model Fusion (LightGBM, Autoencoder, PyTorch GAT)?
**Panel Question:** *"Why use three different models instead of just one big Neural Network?"*

**Your Defense:**
> "Fraud is not a single problem; it has three distinct behaviors. We built a Tri-Model architecture because a single algorithm cannot catch everything:
> 
> 1. **LightGBM (The Speed Engine):** It processes tabular data extremely fast. It catches known, historical fraud patterns instantaneously, which is critical for our sub-100ms real-time API goal.
> 2. **PyTorch Autoencoder (The Zero-Day Catcher):** Fraudsters constantly invent new tactics (zero-day attacks) that LightGBM hasn't learned yet. We train the unsupervised Autoencoder *only* on legitimate transactions. When a novel fraud tactic hits the system, the Autoencoder fails to reconstruct it, flagging it as an anomaly.
> 3. **PyTorch GAT (The Syndicate Catcher):** Smart criminals use multiple accounts and devices to launder money (collusive rings). Standard models look at transactions in isolation. The Graph Attention Network looks at the *relationships* (shared devices, IPs) to shut down the entire ring."

---

## 4. Why MLflow?
**Panel Question:** *"Why did you integrate MLflow?"*

**Your Defense:**
> "In a real-world enterprise MLOps environment, you can't just train models in a Jupyter Notebook and forget the parameters. **MLflow** allows us to rigorously track every single experiment. When we apply SMOTE, MLflow logs the new class distributions, tracks the hyperparameters of our LightGBM, and logs the AUC scores. This ensures our research is reproducible and ready for production."

---

## 5. Why SHAP (Phase 3)?
**Panel Question:** *"Why is explainability so important?"*

**Your Defense:**
> "Under financial regulations like the **RBI guidelines and GDPR**, if a bank rejects a transaction or denies a loan using AI, they must be able to explain *why* to the customer. Black-box deep learning models fail this legal requirement. By integrating **SHAP (Shapley Additive Explanations)**, our system generates an instant audit log for every blocked transaction, proving exactly which features triggered the fraud alert to satisfy regulatory compliance."
