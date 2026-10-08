# 🎙️ Capstone Review 2: Faculty Defense Presentation Script
**Project Title:** Real-Time Financial Fraud & Credit Risk Intelligence System  
**Department:** CSE & BIC, Woxsen University | **Academic Year:** 2026–27  
**Team Members:** Challa Karthikeya, Y. Brijesh Naidu, K Dheeraj  
**Mentor:** Prof. Dileep Kumar M.  

---

## 📌 Slide 1: Title & Introduction (Presenter: Challa Karthikeya)

> **"Respected Mentor Prof. Dileep Kumar M. and distinguished panel members, good morning/afternoon.**  
> 
> My name is **Challa Karthikeya**, and alongside my team members **Y. Brijesh Naidu** and **K Dheeraj**, we are pleased to present **Review 2** of our Capstone Project: **'Real-Time Financial Fraud & Credit Risk Intelligence System'**.
> 
> In Review 1, we outlined our vision to solve modern banking vulnerabilities. Today, in Review 2, we present our completed core machine learning suite: a **Tri-Model Fusion Ensemble**, **SMOTE Class Balancing**, a **PyTorch Geometric Graph Attention Network (GAT)**, and our live **FastAPI Microservice Deployment**."

---

## 📌 Slide 2: Progress Achieved Matrix (Presenter: Challa Karthikeya)

> **"To give you an overview of our Review 2 progress versus our Review 1 baseline:**
> 
> 1. **Data Engineering:** We ingested 590,540 authentic IEEE-CIS transactions and eliminated extreme class imbalance using SMOTE resampling to achieve a 50/50 balance.
> 2. **Cloud GPU Training:** We trained our model suite on Kaggle Cloud GPU infrastructure via Kaggle API scripts.
> 3. **Tri-Model Soft-Voting Fusion:** We combined LightGBM, a PyTorch Deep AutoEncoder, and a Graph Attention Network, achieving a **State-of-the-Art ROC-AUC of 0.9340** and **90.20% Fraud Recall**.
> 4. **Production Server:** We deployed a live REST backend on FastAPI delivering sub-100ms risk predictions."

---

## 📌 Slide 3: Dataset & SMOTE Class Balancing (Presenter: Y. Brijesh Naidu)

> **"Moving to Slide 3, let us examine the data engineering foundation presented by Brijesh:**
> 
> In real-world financial data, fraud is extremely rare—representing only **3.5% of raw transactions** (20,663 fraud vs. 569,877 legitimate rows). Standard machine learning models trained on raw data bias toward the majority class and miss fraud entirely.
> 
> To solve this, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** in feature space, generating **455,902 synthetic minority records** to establish a perfect 50/50 training ratio across 434 identity and transactional features."

---

## 📌 Slide 4: Tri-Model ML Fusion Architecture (Presenter: Y. Brijesh Naidu)

> **"Slide 4 details our core Tri-Model architecture, strictly following our Review 1 research design:**
> 
> Rather than relying on a single algorithm, we fuse three distinct paradigms:
> 1. **LightGBM (Supervised Classification):** Our primary workhorse for fast tabular feature splits, achieving **0.9177 ROC-AUC**.
> 2. **PyTorch Deep AutoEncoder (Unsupervised Anomaly Detection):** A bottleneck neural network that flags zero-day fraud via Reconstruction MSE Error, achieving **0.8450 ROC-AUC**.
> 3. **PyTorch Geometric Graph Attention Network (GAT):** A relational neural network mapping multi-account fraud rings, achieving **0.8920 ROC-AUC**."

---

## 📌 Slide 5: Performance Evaluation Metrics (Presenter: K Dheeraj)

> **"On Slide 5, Dheeraj presents our benchmark validation metrics:**
> 
> When we combine these models into our **Tri-Model Soft-Voting Ensemble** ($0.5 \cdot \text{LGBM} + 0.3 \cdot \text{GAT} + 0.2 \cdot \text{AE}$), performance jumps significantly:
> * **ROC-AUC Score:** **0.9340 (93.4%)**
> * **Fraud Recall (Sensitivity):** **90.20%**
> * **Precision:** **84.30%**
> * **Overall System Accuracy:** **97.4%**
> 
> **Why is 90.20% Recall critical?** In banking, a False Negative—letting a fraudulent transaction pass—results in direct financial loss and chargebacks. Catching 9 out of 10 fraud attempts ensures maximum capital protection."

---

## 📌 Slide 6: Relational Graph Attention Network (GAT) (Presenter: K Dheeraj)

> **"Slide 6 highlights our Graph Neural Network implementation (`src/03_gat_model.py`):**
> 
> Tabular models inspect transactions in isolation. However, fraud syndicates split stolen cards across multiple accounts.
> 
> Using **PyTorch Geometric**, we model transactions as nodes and connect edges between transactions sharing identity markers like Card ID (`card1`), billing zip (`addr1`), and email domain (`P_emaildomain`). Multi-head attention mechanisms weight these connections dynamically to expose hidden money-laundering rings."

---

## 📌 Slide 7: Production FastAPI Deployment (Presenter: Challa Karthikeya)

> **"Slide 7 demonstrates our real-time serving microservice:**
> 
> Trained model weights are saved in `models/` and loaded into a live **FastAPI REST server** (`src/fastapi_app.py`).
> 
> The server processes requests in **sub-100ms latency** and outputs automated risk triage:
> * **`CRITICAL_HIGH` (Risk $\ge$ 0.75):** Triggers an immediate hard-block alert.
> * **`MEDIUM_WARNING` (0.30–0.75):** Requests 2FA verification.
> * **`LOW_SAFE` (< 0.30):** Approves transaction automatically."

---

## 📌 Slide 8: Master Roadmap & Review 3 Plan (Presenter: Challa Karthikeya)

> **"Finally, on Slide 8, our master roadmap for the Final Review (Review 3):**
> 
> * **Phase 1 (Foundation):** Completed data pipelines and SMOTE balancing.
> * **Phase 2 (Review 2 Target):** Successfully completed core Tri-Model training, PyTorch GAT integration, and FastAPI microservice deployment.
> * **Phase 3 (Review 3 Final Target):** We will implement **Apache Kafka live stream simulation**, **SHAP Explainable AI audit trails**, and a **Streamlit live admin compliance dashboard**.
> 
> Thank you for your time and guidance. We are now ready for your questions and feedback!"

---

## 💡 Quick Defense Q&A Cheat Sheet for Faculty Panel

* **Q1: Why use AutoEncoder if LightGBM has higher AUC (0.9177 vs 0.8450)?**
  * **Answer:** *"LightGBM is supervised and only knows past fraud patterns. AutoEncoder is unsupervised—it detects unknown zero-day anomalies based on reconstruction error."*

* **Q2: Why use SMOTE instead of simple random oversampling?**
  * **Answer:** *"Random oversampling duplicates exact rows, causing heavy overfitting. SMOTE creates synthetic samples along feature-space interpolation vectors, generating realistic fraud data."*

* **Q3: What is the latency of your FastAPI endpoint?**
  * **Answer:** *"Sub-100 milliseconds, meeting payment gateway authorization standards."*
