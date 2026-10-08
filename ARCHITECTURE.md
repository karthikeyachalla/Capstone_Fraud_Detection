# 🏛️ System Architecture: Real-Time Digital Banking & Transfer Risk Intelligence

This document defines the authoritative architecture, data flows, and dual-tier model serving infrastructure for the system.

---

## 1. End-to-End System Topology

```mermaid
flowchart TD
    subgraph Ingestion ["1. REAL-TIME PAYMENT STREAM INGESTION"]
        TxIn["Incoming Payment Event\n(UPI / IMPS / Mobile Banking)"] -->|POST /predict (JSON)| API["FastAPI Microservice (Port 8000)"]
        API --> RuleGate["Tier 0: Deterministic Safety Rails (<1ms)\n• Impossible geo-velocity\n• Blacklisted accounts / device hashes"]
    end

    subgraph DualHorizonServing ["2. DUAL-HORIZON INTELLIGENCE ENGINE (<50ms SLA)"]
        RuleGate -->|Pass| Splitter["Feature Extraction & Dispatch"]
        
        subgraph OnlineTier ["Fast Online Scoring (<5ms)"]
            Splitter --> LGBM["Cost-Sensitive LightGBM Classifier\n• Asymmetric Loss Optimization (FN penalty >> FP)\n• Evaluates transfer amounts, balance velocity, time"]
            Splitter --> AE["Deep Autoencoder (PyTorch)\n• Trained on legitimate transfers\n• Reconstruction Error MSE = Zero-Day Anomaly"]
        end

        subgraph NearLineTier ["Near-Line Graph Engine (Redis Cache: <2ms)"]
            TxLedger[("Streaming Transfer Ledger")] --> GraphBuilder["PyTorch Geometric Bipartite Graph\n• Nodes: Senders (nameOrig) & Beneficiaries (nameDest)\n• Edges: Fund Transfer Links & Velocity"]
            GraphBuilder --> GAT["Heterogeneous GAT (GATConv)\n• Detects Star-Topology Mule Hubs\n• Computes 32-dim Graph Mule Risk Embeddings"]
            GAT -.->|Cache Embeddings| RedisCache[("Fast In-Memory / Redis Store")]
            RedisCache --> Splitter
        end
    end

    subgraph TriageTier ["3. RISK SYNTHESIS & TRIAGE"]
        LGBM --> Synthesis["Weighted Risk Synthesis Engine\nP(fraud) = 0.5×LGBM + 0.3×GAT + 0.2×AE"]
        AE --> Synthesis
        
        Synthesis --> Triage{"Banking Policy Triage"}
        Triage -->|Score < 0.25 & Low PD| Approve["✅ AUTO-APPROVE (Fast-track Transfer/Credit)"]
        Triage -->|0.25 <= Score < 0.70| Challenge["⚠️ STEP-UP 2FA / BIOMETRIC OTP"]
        Triage -->|Score >= 0.70 or Mule Match| Block["🚨 BLOCK TRANSFER & FREEZE MULE"]
    end

    subgraph ExplainabilityTier ["4. XAI COMPLIANCE & DRIFT ENGINE"]
        Synthesis --> SHAP["TreeSHAP Attribution Engine\n• Local feature weights for challenged/blocked payments\n• RBI & DPDP Act 2023 compliant audit logs"]
        TxIn --> DriftMon["Streaming Concept Drift Monitor\n• ADWIN & Two-Sample KS-Test\n• Population Stability Index (PSI >= 0.2 -> Retrain Alert)"]
    end
```

---

## 2. Component Specifications

### 2.1 Online Serving Microservice (`src/fastapi_app.py`, `src/multi_agent_gateway.py`)
* **Framework:** Asynchronous FastAPI running on Python 3.10+.
* **Input Schema:** Real-world banking fund transfer fields:
  * `transaction_id`, `amount`, `channel_type` (`UPI`, `IMPS`, `NET_BANKING`)
  * `sender_id` (`nameOrig`), `recipient_id` (`nameDest`)
  * `sender_balance_before`, `sender_balance_after`, `recipient_balance_before`, `recipient_balance_after`
  * `device_id`, `ip_address`, `auth_method`, `login_failures`
* **Latency SLA:** Evaluates full ensemble in **$<30\text{ms}$**, comfortably within the 50ms banking switch limit.

### 2.2 Machine Learning Engine (`src/02_model_training.py`)
* **Algorithm:** LightGBM Gradient Boosted Decision Trees.
* **Class Imbalance Policy:** **Cost-Sensitive Learning (`scale_pos_weight`)** and $F_\beta$ threshold tuning ($\beta = 3$) prioritizing high recall over false alarms.

### 2.3 Relational Graph Network (`src/03_gat_model.py`)
* **Framework:** PyTorch Geometric (PyG).
* **Graph Schema:** Bipartite graph modeling sender accounts (`nameOrig`) and destination accounts (`nameDest`).
* **Mule Ring Detection:** Flags star-topology hubs receiving rapid fan-in micro-transfers from multiple distinct origins.
* **Serving Pattern:** Pre-computes 32-dim graph embeddings near-line to eliminate real-time graph traversal overhead.

### 2.4 Streaming Concept Drift Monitor (`src/05_concept_drift_monitor.py`)
* **Continuous Metrics:** Kolmogorov-Smirnov (KS) test over sliding temporal windows ($p < 0.05$ indicates distribution shift).
* **Categorical Metrics:** Population Stability Index ($\text{PSI} \ge 0.2$ triggers automated model update alerts).
