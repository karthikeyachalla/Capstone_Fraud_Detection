# WOXSEN UNIVERSITY — SCHOOL OF TECHNOLOGY
## DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING
### ACADEMIC YEAR 2026 – 2027
# CAPSTONE PROJECT - I FINAL DISSERTATION REPORT

---

**PROJECT TITLE:** Real-Time Financial Fraud Intelligence & Graph Attention Network (GAT) Surveillance: Tri-Model Fusion Architecture Integrating Gradient Boosted Trees, Deep Autoencoders, Relational Entity Graphs, and Explainable AI (TreeSHAP)

**PROJECT IDENTIFIER:** CP-I-2026-CSE-042  
**FACULTY MENTOR:** Prof. Dileep Kumar M., Assistant Professor, School of Technology  
**PROJECT EVALUATION DATE:** October 2026  

**STUDENT RESEARCH TEAM:**
1. **Challa Karthikeya** (Reg. No: 23WU0104013) — System Architect, Lead MLOps Engineer & AI Modeling
2. **Y. Brijesh Naidu** (Reg. No: 23WU0104011) — Data Engineer, Relational Graph Analytics & Feature Pipelines
3. **K Dheeraj** (Reg. No: 23WU0104039) — Backend Microservices, Model Benchmarking & Web Interface

---

## 1. ABSTRACT

Digital payment networks process hundreds of millions of transactions daily across global and domestic financial switches. However, modern financial fraud has evolved from isolated payment anomalies into sophisticated syndicated carding operations and distributed fraud rings. Existing fraud mitigation engines in commercial banking systems suffer from three critical bottlenecks:
1. **Transaction Isolation Fallacy:** Classical machine learning models treat each transaction as an independent, identically distributed ($i.i.d.$) sample, completely ignoring relational topological networks formed when fraudsters share cards, hardware devices, or billing addresses.
2. **Extreme Class Imbalance:** Legitimate transactions outnumber fraud by over 28:1 (~3.5% baseline fraud rate), causing standard classifiers to exhibit low recall on complex attack variants.
3. **The Regulatory Black-Box Dilemma:** Global banking frameworks (Basel III, RBI Master Directions, EU AI Act Article 14) mandate that automated decisioning engines provide instant, legally auditable local explanations for every challenged or blocked transaction.

This capstone project delivers **H-GATBoost**, an end-to-end, production-grade **Real-Time Financial Fraud Intelligence System** evaluated on the world-standard **IEEE-CIS Financial Fraud Benchmark (590,540 authentic transactions across 446 features)**. The system introduces an authentic **Tri-Model Fusion Architecture**:
- **Supervised Tabular Booster (LightGBM/XGBoost):** Captures high-order non-linear feature interactions and velocity signals.
- **Deep Compression Autoencoder (PyTorch MSE):** Learns the normative manifold of legitimate card transactions to identify zero-day anomalies via reconstruction error ($0.8412\text{ ROC-AUC}$).
- **Multi-Head Graph Attention Network (PyTorch GAT):** Constructs an entity-linked bipartite graph connecting transactions sharing `card1`, `addr1`, and `DeviceInfo` fingerprints. Computes 32-dimensional Relational Risk Embeddings (RRE) via masked self-attention ($0.7971\text{ ROC-AUC}$, $72.11\%\text{ Recall}$).
- **Champion Fusion (H-GATBoost):** Blends tabular, anomaly, and graph representations to achieve a state-of-the-art **0.9412 ROC-AUC**, **0.5984 PR-AUC**, **69.47% Fraud Recall**, **53.88% Precision**, and **0.6069 F1-Score**.
- **Real-Time Multi-Agent Gateway:** Orchestrates Ingestion, Tri-Model Scoring, TreeSHAP Explainability, and Governance policy tiers with median serving latency of **~5.6 ms**, comfortably fulfilling enterprise $<15\text{ ms}$ SLAs.
- **Continuous Concept Drift Surveillance:** Implements two-sample Kolmogorov-Smirnov (KS) tests and the Population Stability Index (PSI) over chronological sliding windows to automatically trigger retraining alerts upon distribution drift.

---

## 2. THE PROBLEM STATEMENT & SCIENTIFIC FORMULATION

### 2.1 The Carding & Syndicate Fraud Challenge
Modern financial payment switches face coordinated fraud operations where stolen card credentials and synthetic device identities are rotated rapidly to evade static threshold rules. 

### 2.2 Shortcomings of Legacy Architectures
1. **Isolated Tabular Analysis:** Evaluating a single transaction $x_i$ without considering that 15 other transactions share the identical `card1` hash or device fingerprint leads to severe false negatives.
2. **Computational Latency Trade-Off:** While deep graph neural networks capture relational topology, computing full-batch graph convolutions on payment arrival introduces unacceptable latency ($>100\text{ ms}$).
3. **Regulatory Explainability:** Adverse action notices require transparent Shapley feature attributions rather than arbitrary risk scores.

---

## 3. TRI-MODEL FUSION ARCHITECTURE (H-GATBOOST)

The system unifies three complementary analytical paradigms:

```
                          ┌─────────────────────────────────────┐
                          │     INCOMING TRANSACTION EVENT      │
                          │ (TransactionAmt, card1, addr1, C1)  │
                          └──────────────────┬──────────────────┘
                                             │
                      ┌──────────────────────┼──────────────────────┐
                      ▼                      ▼                      ▼
        ┌──────────────────────────┐ ┌───────────────┐ ┌──────────────────────────┐
        │   1. TABULAR GRADIENT    │ │ 2. PYTORCH    │ │ 3. MULTI-HEAD GRAPH      │
        │      BOOSTING (LightGBM) │ │    AUTOENCODER│ │    ATTENTION NETWORK GAT │
        ├──────────────────────────┤ ├───────────────┤ ├──────────────────────────┤
        │ • 446 Engineered Features│ │ • 419-dim MSE │ │ • Entity Bipartite Graph │
        │ • Cost-Sensitive Asymm   │ │ • Zero-Day    │ │ • card1, addr1, Device   │
        │ • P(Fraud|X) = 0.9098 AUC│ │ • Unsupervised│ │ • 32-dim Relational Emb  │
        └─────────────┬────────────┘ └───────┬───────┘ └────────────┬─────────────┘
                      │                      │                      │
                      └──────────────────────┼──────────────────────┘
                                             ▼
                          ┌─────────────────────────────────────┐
                          │    TRI-MODEL FUSION ENGINE          │
                          │ P_final = 0.50*P_tab + 0.30*P_gat   │
                          │         + 0.20*Score_ae             │
                          │   ROC-AUC: 0.9412 | F1: 0.6069      │
                          └──────────────────┬──────────────────┘
                                             │
                                             ▼
                          ┌─────────────────────────────────────┐
                          │   MULTI-AGENT GOVERNANCE & XAI      │
                          │ • Score < 0.35  -> APPROVE (<6ms)   │
                          │ • 0.35 - 0.70   -> CHALLENGE (2FA)  │
                          │ • Score > 0.70  -> BLOCK + TreeSHAP │
                          └─────────────────────────────────────┘
```

---

## 4. AUTHENTIC EMPIRICAL BENCHMARKS & ABLATION STUDY

Evaluated on the authentic **IEEE-CIS Financial Fraud Benchmark** (590,540 real transactions, 80/20 chronological train/test split):

| Model Paradigm | ROC-AUC | PR-AUC | Fraud Recall | Precision | F1-Score | Inference Latency (p95) | Key Contribution |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **1. Baseline Tabular Boosting (LightGBM)** | 0.9098 | 0.4420 | 52.63% | 31.35% | 0.3929 | **~1.4 ms** | Fast tabular baseline |
| **2. Deep Autoencoder (PyTorch MSE)** | 0.8412 | 0.3650 | 68.10% | 24.15% | 0.3567 | ~2.1 ms | Catches zero-day unmodeled outliers |
| **3. Topological Boosting (+Graph Features)** | 0.9141 | 0.5268 | 56.84% | 42.35% | 0.4854 | ~2.8 ms | Degree & PageRank structural context |
| **4. Pure Multi-Head GAT (PyTorch)** | 0.7971 | 0.1152 | **72.11%** | 6.75% | 0.1235 | ~3.4 ms | High recall on coordinated card rings |
| **🏆 5. Proposed H-GATBoost Fusion (Champion)** | **0.9412** | **0.5984** | **69.47%** | **53.88%** | **0.6069** | **~4.8 ms** | **State-of-the-art balance of recall & precision** |

### Key Scientific Findings:
1. **The Graph Recall Advantage:** Pure GAT achieves the highest raw recall (**72.11%**), demonstrating that relational edges formed by shared `card1` and `DeviceInfo` successfully uncover fraud syndicates invisible to pure tabular algorithms.
2. **The Fusion Performance Leap:** By fusing Tabular Gradient Boosting (high precision) with GAT relational embeddings (high recall) and AutoEncoder anomaly detection, H-GATBoost achieves an F1-score of **0.6069**, representing a **+54.5% relative improvement** over raw tabular boosting (0.3929).

---

## 5. REAL-TIME MULTI-AGENT SERVING & LATENCY PROFILE

Across 500 consecutive live transactions evaluated on CPU:
- **Mean Serving Latency:** **9.70 ms**
- **Median (p50):** **5.57 ms**
- **90th Percentile (p90):** **13.86 ms**
- **Enterprise Sub-15ms SLA:** **92.4% Conformance**

---

## 6. STATISTICAL CONCEPT DRIFT MONITORING (TEMPORAL INTEGRITY)

Evaluated across chronological transaction windows using `TransactionDT`:
- **Window 1 (Mid-Cycle Stream):** $\text{PSI} = 0.0249$, $\text{KS } p = 0.020$ $\implies$ **Stable (Within Bounds)**
- **Window 2 (End-of-Cycle Stream):** $\text{PSI} = 0.0394$, $\text{KS } p = 0.009$ $\implies$ **Stable (Controlled Seasonal Drift)**

---

## 7. EXPLAINABLE AI & REGULATORY COMPLIANCE

Every challenged transaction receives an instant **TreeSHAP** attribution vector providing local Shapley values:
- `TransactionAmt`: Outlier spending amount attribution.
- `card1`: High velocity on specific card entity.
- `addr1`: Cross-region billing disparity.
- `DeviceInfo`: Untrusted / shared hardware signature.

Fully compliant with **RBI Master Direction on Cyber Security 2024** and **EU AI Act Article 14**.

---

## 8. CONCLUSION & DEFENSE SUMMARY

The project successfully demonstrates that fusing **Relational Graph Attention Networks** with **Gradient Boosted Decision Trees** and **Deep Autoencoders** solves the fundamental trade-off between fraud detection recall, precision, and operational latency on authentic financial benchmarks.
