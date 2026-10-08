# 📌 Project Roadmap & Living Tasks

> Track progress across sprints. Updated to reflect the complete Real-Time Digital Banking & Transfer Risk Intelligence architecture.

---

## Sprint 1: Architecture & Governance Alignment (Completed)
- [x] Align `PRD.md` with Real-Time Digital Banking & Transfer Fraud (UPI, IMPS, Mobile Banking) + Credit Risk Intelligence.
- [x] Update `ARCHITECTURE.md` with the Two-Tier Lambda serving flow (Near-Line GAT + Online LightGBM/Autoencoder).
- [x] Embed research-backed faculty viva defense scripts in `MEMORY.md` (citing Alarfaj 2024, Anuganti 2024, Li & Chen 2025).
- [x] Enforce engineering invariants in `AGENTS.md` (sub-10ms rule, circuit-breaker, entity consistency, no dummy zero-vectors).

---

## Sprint 2: Core Code Repair & Authentic Feature Ingestion (Completed)
- [x] **Fix Live Gateway (`src/multi_agent_gateway.py`):**
  - Eliminated the `np.zeros` dummy feature vector.
  - Ingested authentic 37-feature digital banking payload (amounts, balances, drain ratio, device trust, authentication method, bank name).
- [x] **Implement Real TreeSHAP in Explainability Agent:**
  - Integrated `shap.TreeExplainer` on the LightGBM booster.
  - Generates mathematical Shapley feature attribution values alongside regulatory RBI / DPDP Act audit narratives in 4.9ms (sub-10ms SLA verified).
- [x] **Fix Graph Neural Network (`src/03_gat_model.py`):**
  - Eliminated random noise training; built authentic relational bipartite transaction graph (1,200 nodes, 50,336 edges).
  - Trained 2-layer `FinancialFraudGAT` with multi-head self-attention.
  - Exported weights `models/gat_model.pt` and cached pre-computed near-line 32-dim Graph Mule Risk Embeddings (`models/mule_risk_embeddings.pkl`).
  - Integrated near-line GAT cache into `src/multi_agent_gateway.py` (<2ms lookup).

---

## Sprint 3: Model Training & SMOTE Ablation Table (Completed)
- [x] **Train Cost-Sensitive LightGBM (`src/train_indian_fraud.py`):**
  - Trained on 15,000 verified Indian digital payment transfers using `scale_pos_weight` asymmetric penalization.
  - Saved artifacts: `indian_fraud_model.pkl`, `indian_scaler.pkl`, `indian_encoders.pkl`, `indian_model_meta.pkl`.
- [x] **Run Empirical Ablation Study (`src/03_run_ablation_study.py`):**
  - Evaluated 6 architectures: Baseline Logistic Regression, Random Forest (100 Trees), Cost-Sensitive LightGBM, Deep Autoencoder, Relational GAT, and Tri-Model Fusion.
  - Saved CSV results to `models/ablation_metrics.csv`.
  - Updated `BENCHMARKS.md` with empirical ROC-AUC, PR-AUC, Recall, Precision, and p95 Latency.
- [x] **Semi-Supervised Deep Autoencoder Anomaly Detection:**
  - Evaluated reconstruction MSE anomaly scoring on legitimate transaction distributions.

---

## Sprint 4: Streaming Simulation & MLOps (Completed)
- [x] **Connect Apache Kafka Stream Simulator (`src/04_kafka_stream_simulator.py`):**
  - Streamlined relative path ingestion and Pydantic schema compatibility with FastAPI `/predict`.
  - Supports live streaming of Indian UPI and IEEE-CIS transactions with latency and policy intercept telemetry.
- [x] **Build Streaming Concept Drift Monitor (`src/05_concept_drift_monitor.py`):**
  - Implemented Two-Sample Kolmogorov-Smirnov (KS-test) for continuous distributions and Population Stability Index (PSI) across sliding temporal windows.
  - Successfully flags covariate shift (Diwali holiday shopping) vs. adversarial concept drift (night-time micro-UPI smurfing outbreak).
  - Exports JSON telemetry report to `reports/drift_report.json`.
- [x] **Package Multi-Container Setup (`Dockerfile`, `docker-compose.yml`):**
  - Production-ready `Dockerfile` running FastAPI with health checks.
  - `docker-compose.yml` orchestrating API (port 8000) and Web Presentation Dashboard (port 3000).
- [x] **Automated CI/CD Test Suite (`tests/test_pipeline.py`):**
  - 9 automated end-to-end tests passing in 0.47s (Health, Safe Approval, Fraud Block, XAI, Loan Underwriting, GAT cache, Drift Monitor, Stream Simulator).
