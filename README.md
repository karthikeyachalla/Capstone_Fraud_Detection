---
title: Financial Fraud Risk Intelligence Gateway
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 💳 Real-Time Financial Fraud & Credit Risk Intelligence System
> **Woxsen University — Final Year B.Tech Capstone Project — I (Academic Year 2026–27)**  
> **Student Team:** Challa Karthikeya (23WU0104013), Y. Brijesh Naidu (23WU0104011), K Dheeraj (23WU0104039)  
> **Faculty Mentor:** Prof. Dileep Kumar M.  
> **Architecture:** Production Multi-Agent Gateway (LightGBM Booster + PyTorch AutoEncoder + PyTorch Multi-Head GAT)  
> **Benchmark Performance (IEEE-CIS Benchmark):** **0.9412 ROC-AUC** | **69.47% Fraud Recall** | **53.88% Precision** | **0.6069 F1-Score** | **Median Serving Latency ~5.6 ms**  

---

## 📌 System Architecture & Pipeline Overview

```
                                  ┌──────────────────────────────┐
                                  │     INCOMING PAYMENT EVENT   │
                                  │   (UPI, Card, IMPS, NetBank) │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   AGENT 1: PREDICTION AGENT  │
                                  │ • Cost-Sensitive LightGBM    │
                                  │ • Near-Line GAT Mule Cache   │
                                  │ • Deep Autoencoder Anomaly   │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │     AGENT 2: TRIAGE AGENT    │
                                  │ • Policy Decision Engine:    │
                                  │   Score >= 0.75 ──> BLOCK    │
                                  │   0.30 - 0.74   ──> 2FA OTP  │
                                  │   Score < 0.30  ──> APPROVE  │
                                  └──────────────┬───────────────┘
                                                 │
                       ┌─────────────────────────┴─────────────────────────┐
                       ▼                                                   ▼
        ┌─────────────────────────────┐                     ┌─────────────────────────────┐
        │   AGENT 3: EXPLAINABILITY   │                     │   AGENT 4: CREDIT & LOAN    │
        │           AGENT             │                     │     UNDERWRITING AGENT      │
        ├─────────────────────────────┤                     ├─────────────────────────────┤
        │ • Real TreeSHAP Attribution │                     │ • Evaluates monthly income, │
        │ • Mathematical Shapley Push │                     │   CIBIL score (300-900),    │
        │ • RBI & DPDP Act 2023 Audit │                     │   DTI ratio, and debt EMI:  │
        │   Narrative Generation      │                     │   CIBIL >= 750 ──> APPROVED │
        │                             │                     │   CIBIL < 650  ──> REJECTED │
        └─────────────────────────────┘                     └─────────────────────────────┘
```

---

## 📂 Project Directory Structure

```text
Capstone_Fraud_Detection/
├── data/
│   └── raw/                                 <-- IEEE-CIS & Indian Banking Datasets (15k records)
├── src/
│   ├── 01_exploratory_data_analysis.py      <-- EDA & Imbalance Profiling
│   ├── 02_model_training.py                 <-- Supervised & AutoEncoder Training
│   ├── 03_gat_model.py                      <-- Relational Graph Attention Network & Mule Cache
│   ├── 03_run_ablation_study.py             <-- Scientific Ablation Study Benchmarking Suite
│   ├── 04_kafka_stream_simulator.py         <-- Real-Time Streaming Ingestion Simulator
│   ├── 05_concept_drift_monitor.py          <-- KS-Test & PSI Streaming Drift Surveillance
│   ├── multi_agent_gateway.py               <-- 4-Agent Orchestrator (Predict, Triage, XAI, Credit)
│   └── fastapi_app.py                       <-- Sub-10ms Serving REST Microservice
├── models/                                  <-- Serialized Trained Model Weights & Embeddings
│   ├── indian_fraud_model.pkl               <-- LightGBM Supervised Booster
│   ├── autoencoder.pt                       <-- PyTorch Deep AutoEncoder Weights
│   ├── gat_model.pt                         <-- Relational GAT Network Weights
│   ├── mule_risk_embeddings.pkl             <-- 32-dim Near-Line Graph Mule Cache
│   └── ablation_metrics.csv                 <-- Empirical Benchmark Metrics
├── app/                                     <-- Interactive Presentation & Screener Web UI
├── scripts/
│   ├── benchmark_latency.py                 <-- 500-Transaction Latency SLA Profiler (<10ms)
│   └── run_training.py                      <-- Local Pipeline Runner
├── tests/                                   <-- 9/9 Automated CI/CD Unit Test Suite
├── Dockerfile                               <-- Production Container Definition
├── docker-compose.yml                       <-- Multi-Container Orchestration (API + Dashboard)
├── 06_Review_2_Submission/                  <-- Review 2 Deliverables & Reports
└── 07_Final_Capstone_Submission/            <-- Final Dissertation, Deck & Viva Defense
```

---

## ⚡ How to Run the Project Live:

### 1. Run Automated CI/CD Test Suite (9 Tests in 0.5s):
```bash
python3 -m unittest discover -s tests
```

### 2. Launch Real-Time FastAPI Endpoint:
```bash
python3 -m uvicorn src.fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 3. Screen a Transaction (Live API):
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "transaction_id": "TXN_UPI_9901",
       "transaction_amount": 65000.0,
       "payment_method": "UPI",
       "upi_app": "PhonePe",
       "bank_name": "SBI",
       "transaction_hour": 2,
       "new_device": "Yes",
       "location_match": "No",
       "failed_login_count": 3
     }'
```

### 4. Underwrite a Loan / Credit Card Application:
```bash
curl -X POST "http://127.0.0.1:8000/credit-underwrite" \
     -H "Content-Type: application/json" \
     -d '{
       "applicant_name": "Challa Karthikeya",
       "monthly_income": 95000.0,
       "cibil_score": 785,
       "existing_monthly_emi": 15000.0,
       "requested_loan_amount": 500000.0,
       "employment_type": "Salaried"
     }'
```

### 5. Real Apache Kafka Broker & Streaming Pipeline:
Apache Kafka 4.3.1 runs locally on `localhost:9092` with 3 partitions on topic `financial.transactions.stream`.

* **Inspect Kafka Topic:**
```bash
kafka-topics --bootstrap-server localhost:9092 --describe --topic financial.transactions.stream
```

* **Start Real Kafka Consumer (AI Inference Engine):**
```bash
python3 src/04_kafka_consumer.py
```

* **Start Real Kafka Producer (Transaction Publisher):**
```bash
python3 src/04_kafka_producer.py 30 0.15
```

### 6. Run Streaming Concept Drift Surveillance:
```bash
python3 src/05_concept_drift_monitor.py
```

### 7. Run Automated CI/CD Test Suite:
```bash
python3 -m unittest tests/test_pipeline.py
```

---

## 👥 Authors & Academic Affiliation
* **Department of Computer Science & Engineering** | Woxsen University  
* **Lead Architect:** Challa Karthikeya (23WU0104013)  
* **Data Engineer:** Y. Brijesh Naidu (23WU0104011)  
* **Backend & Benchmarking:** K Dheeraj (23WU0104039)  
* **Faculty Mentor:** Prof. Dileep Kumar M.  
