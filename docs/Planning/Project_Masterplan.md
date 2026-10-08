# Capstone Project: Real-Time Financial Fraud & Credit Risk Intelligence System

## 1. Project Overview & Objective
**Goal:** Build a FAANG-grade, end-to-end Machine Learning pipeline that detects zero-day financial fraud and uncovers collusive money-laundering rings in real-time (<100ms latency), while strictly adhering to financial regulations (RBI/GDPR) via Explainable AI.

**Core Dataset:** IEEE-CIS Financial Fraud Detection (Kaggle). Chosen specifically because it provides both transaction metadata AND identity linkages (devices, IP features, emails), which are strictly required for Graph Neural Network (GNN) mining.

---

## 2. The Tri-Model Fusion Architecture
Fraud manifests in three ways, so we use three specialized engines:
1. **LightGBM (The Speed Engine):** High-speed tabular classifier trained on historical fraud signatures.
2. **PyTorch AutoEncoder (The Zero-Day Catcher):** Unsupervised neural network trained exclusively on legitimate transactions. It flags novel/unseen fraud tactics as mathematical anomalies.
3. **PyTorch GAT - Graph Attention Network (The Syndicate Catcher):** Maps relationships between shared devices, emails, and accounts to uncover coordinated fraud rings.

---

## 3. Technology Stack & Vibe Coding Workflow
To maintain context and execute flawlessly in a "vibe coding" flow, we adhere strictly to this stack:

### Data Engineering & Streaming
- **Pandas / NumPy:** Core data manipulation.
- **SMOTE (imbalanced-learn):** Resolves the severe ~3.5% minority class imbalance.
- **Apache Kafka:** Simulates live sub-second transaction streams.

### Machine Learning & Deep Learning
- **LightGBM:** Supervised tabular classification.
- **PyTorch:** Custom AutoEncoder for anomaly detection.
- **PyTorch Geometric (PyG):** Handles the Graph Attention Network (GAT) layer.

### MLOps & Production
- **MLflow:** Tracks all experimental hyperparameters, SMOTE distributions, and AUC scores.
- **ADWIN (scikit-multiflow/River):** Monitors live data streams for concept drift.
- **FastAPI:** High-performance async Python backend to expose the models as REST APIs.
- **Streamlit:** Frontend dashboard for bank admins.

### Explainability & Generative AI (The "Killer Features")
- **SHAP:** Generates Shapley values to prove *why* a transaction was blocked (Regulatory Audit).
- **LLM Integration (GenAI):** Translates SHAP mathematical outputs into human-readable incident reports for bank admins.

---

## 4. Master Plan of Action (Roadmap)

### ✅ Phase 1: Foundation (Completed)
- Set up the project structure.
- Define the problem statement, literature review, and baseline architecture.

### ✅ Phase 2: Core Model Execution (Current/Completed for Review 2)
- Ingest real IEEE-CIS dataset.
- Apply SMOTE for class balancing.
- Train the LightGBM supervised model.
- Train the PyTorch AutoEncoder.
- Log metrics to MLflow.
- *(Note: Currently awaiting user to download Kaggle data into `data/raw/` to run `02_model_training.py`).*

### 🚀 Phase 3: Graph Mining & Production (Next Steps for Review 3)
- **Step 1:** Implement PyTorch GAT on the identity features to detect fraud rings.
- **Step 2:** Wrap the Tri-Model fusion inside a `FastAPI` service.
- **Step 3:** Build the `Streamlit` compliance dashboard.
- **Step 4:** Integrate `SHAP` explainability plots into the dashboard.
- **Step 5:** Integrate the GenAI LLM to auto-generate incident reports from SHAP values.
- **Step 6:** Deploy Kafka stream to push mock live transactions into the API.

---

## 5. Developer "Vibe Coding" Guidelines
For seamless AI pair-programming:
- **No Mock Data:** We strictly use the real IEEE-CIS dataset.
- **Strict File Modularity:** Keep model training (`src/`), API deployment (`api/`), and frontend (`app/`) entirely decoupled.
- **State Tracking:** Always check MLflow logs before tuning hyperparameters.
- **Iterative Execution:** Write code -> Test locally -> Verify MLflow -> Move to next feature.
