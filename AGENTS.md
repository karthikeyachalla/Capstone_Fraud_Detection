# 🛡️ Project Governance & Agent Rules: Capstone Fraud Detection

> **Target Environment:** Python 3.10+ | macOS / Linux | FastAPI | LightGBM | PyTorch  
> **Repository Root:** `Capstone_Fraud_Detection/`  
> **Project Scope:** Real-Time Financial Fraud & Credit Risk Intelligence System

---

## 1. Core Engineering Principles

1. **Sub-10ms Latency Rule:**  
   Every inference endpoint in `app/` and `src/fastapi_app.py` must respond in **under 10ms** (p95). Avoid unvectorized Python loops or expensive operations on the critical transaction path.
2. **Resilience & Circuit Breakers:**  
   If deep learning models (Autoencoder, GAT) encounter timeouts or missing graph features, the gateway MUST fall back gracefully to the LightGBM tabular model without crashing or rejecting valid transactions.
3. **Strict Input Validation:**  
   All transaction payloads received by API endpoints must be validated via Pydantic models. Negative amounts, missing timestamps, or invalid card categories must be cleanly rejected with HTTP 422.
4. **Data Privacy & Compliance:**  
   Adhere to **RBI Cyber Security Framework** and India's **DPDP Act 2023**. Never log unmasked PAN (card numbers), CVVs, or cleartext customer PII.

---

## 2. Code Quality & Standards

* **Typing:** Use strict Python type annotations (`typing.Dict`, `typing.Optional`, `pydantic.BaseModel`) on all functions.
* **Error Handling:** Never use bare `except: pass`. Log all exceptions with context.
* **Dependencies:** Do not introduce unpinned or heavy new libraries without explicit justification. Keep `requirements.txt` clean and reproducible.
* **Environment Variables:** All secrets, database URLs, and MLflow ports must be loaded via `python-dotenv` from a `.env` file, falling back to sensible local defaults.

---

## 3. Agent Execution Guidelines

* **Verification First:** Before marking any task complete, the agent must execute syntax compilation (`python -m py_compile`) and run automated test suites (`pytest`).
* **Preserve Research Context:** Do not delete existing benchmark tables, evaluation notes, or faculty defense documentation in `docs/`.
* **Atomic Changes:** Modify one file or component at a time, test it, and verify before proceeding.

---

## 4. Advanced Machine Learning & Architectural Invariants

1. **Entity Consistency in Multi-Model Ensembles:**
   - Multi-model fusion pipelines (e.g., GBDT + GNN + Autoencoder) must share an authentic, unified entity relationship. Graph nodes (senders, receivers, devices) must link directly to the tabular records via common keys (`TransactionID`, `card1`, `device_id`).
   - Never train GNN components on independent random noise or disjoint datasets. Both LightGBM and GAT must be trained on the same IEEE-CIS `TransactionID` key space.

2. **Class Imbalance Policy (Beyond SMOTE):**
   - For ultra-imbalanced financial streams, prioritize **Cost-Sensitive Learning** (`scale_pos_weight`, asymmetric loss weighting) and **Focal Loss** over synthetic oversampling (SMOTE).
   - Document empirical trade-offs via an explicit Ablation Study comparing Baseline, SMOTE, and Cost-Sensitive approaches. Results must be recorded in `BENCHMARKS.md`.

3. **Authentic Feature Ingestion at Serving Time:**
   - All REST inference endpoints (`POST /predict`) must transform live payload parameters directly into the exact feature columns expected by the trained model.
   - Using placeholder zero-vectors (`np.zeros`) or dummy feature vectors at runtime is **strictly prohibited**.

4. **Continuous Concept Drift & Degradation Monitoring:**
   - In production streams, monitor incoming feature distributions using Kolmogorov-Smirnov (KS) tests and Population Stability Index (PSI).
   - Automatic retraining alerts must trigger when PSI ≥ 0.2 or KS p-value < 0.05.
