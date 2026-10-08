# Capstone Project I — Final Project Document
## Real-Time Financial Fraud & Credit Risk Intelligence System
### Karthikeya Challa | 23WU0104013 | Woxsen University | 2026–27
### Domain: Data Science | Team Size: 3 | Duration: 16 Weeks (4 Months)

---

## 1. Problem Statement

The Indian banking sector is experiencing a severe escalation in digital financial fraud. UPI-related fraud cases increased by 85% between FY2022–FY2024 (RBI Annual Report, 2024). Empirical studies (IARJSET, 2026) reveal that 96% of the Indian public has been targeted by fraudulent communications, with 42% facing direct financial encounters, and only 6% aware of "money mule" fraud. While regulatory interventions by the RBI have curbed fraud in public sector banks, digital fraud in private sector banks continues to rise sharply (SRCC Trend Analysis, 2025).

Existing fraud detection systems fail for three critical reasons:

1. **Batch Processing:** They detect fraud only after money has already moved, rather than in real time.
2. **Transaction Isolation:** They evaluate each transaction independently using basic tabular ML, missing coordinated fraud rings and money mule networks.
3. **Black-Box Compliance Gap:** They operate without explainability, failing emerging regulatory expectations that automated financial decisions be transparent and justifiable.

A further gap exists in the literature itself: systematic reviews (Danda & Haritha, 2026; Chen et al., 2025) identify Graph Neural Networks and Explainable AI as the critical emerging direction for fraud detection, yet applied papers in this space (UPI fraud studies, Stanford/VIT baseline papers) still rely entirely on traditional tabular ML, which cannot capture coordinated, multi-account fraud behavior. No existing work combines graph-based fraud-ring detection with regulatory-aligned explainability in one deployable system.

This project builds a three-model fraud intelligence system that scores individual transactions in real time, uses unsupervised anomaly detection to catch previously unseen fraud patterns, applies a Graph Attention Network to detect coordinated fraud-ring behavior invisible to transaction-level models, and provides SHAP-based explainability for every decision.

---

## 2. Dataset

**Primary — IEEE-CIS Fraud Detection (Kaggle, free)**
590,540 real transactions from Vesta Corporation, a real payment processor, with 433 features including card identifiers, email domains, device type, browser, OS, and billing address. Fraud rate ~3.5% (realistic class imbalance, not a toy dataset).

**Why this dataset over the standard ULB Credit Card dataset:** ULB's features are PCA-anonymized (V1–V28) with no identifiers, meaning no transaction graph can be built and no meaningful feature engineering is possible on hidden features. IEEE-CIS's real card, email, device, and address fields are exactly what is needed to construct a genuine fraud-ring graph.

**Scope boundary (stated explicitly):** No public, free dataset of real Indian UPI transactions or fraudulent QR codes currently exists for student research use; existing UPI/QR fraud studies rely on private institutional data (e.g., partnerships with cybercrime cells) not available to this project. This system therefore validates its architecture on the richest available public alternative (IEEE-CIS). Extending the architecture to UPI-specific data is noted as future work. As a partial proxy, PaySim (Kaggle, free) — a synthetic mobile-money dataset with explicit sender/receiver fields structurally similar to UPI peer-to-peer transfers — may optionally be used for a supplementary money-mule detection illustration, given the absence of public UPI datasets.

---

## 3. The Transaction Graph — Core Technical Differentiator

From IEEE-CIS features, a transaction graph is constructed:

- **Nodes:** Each transaction is one node (590,540 nodes total)
- **Edges (four types):**
  - Same card identifiers (card1 + card2 fields) → card-sharing edge
  - Same email domain (P_emaildomain field) → email edge
  - Same device type + browser combination → device fingerprint edge
  - Same billing address code → address edge

**Result:** A heterogeneous transaction graph. Legitimate transactions are mostly isolated nodes with few connections. Fraud rings form dense clusters — many transactions tightly connected through shared identifiers. The Graph Attention Network learns to assign high fraud probability to transactions inside dense suspicious clusters, conceptually similar to how card network internal graph-based fraud systems operate.

---

## 4. The Three-Model Core

### Model 1 — LightGBM (Gradient Boosted Trees)
**Role:** Primary classifier on tabular features.
**Data used:** All 433 tabular features from IEEE-CIS.
**Why:** Fast, handles imbalanced data well, and is the most reliable baseline for tabular fraud detection — used in virtually every winning Kaggle fraud solution. Tuned via Optuna; evaluated on PR-AUC rather than accuracy given the severe class imbalance.

### Model 2 — Autoencoder (Unsupervised Anomaly Detection)
**Role:** Learns what a normal transaction looks like; flags anything that does not reconstruct well as anomalous.
**Data used:** Trained only on legitimate transactions. Reconstruction error serves as the anomaly score.
**Why:** Catches completely new fraud patterns that the supervised model has never seen — zero-day fraud detection, a capability neither LightGBM nor GAT provide on their own.

### Model 3 — Graph Attention Network (GAT)
**Role:** Detects fraud rings by analyzing the transaction network. Each transaction attends to its neighbors in the graph; fraud clusters exhibit distinctive graph structure.
**Data used:** The transaction graph built from IEEE-CIS card/email/device/address connections.
**Why:** The only model that sees beyond individual transactions — catches coordinated, multi-account fraud that both other models miss.

**Fusion:** All three models produce a fraud probability score. A weighted ensemble (weights tuned on the validation set) produces the final score. A transaction is flagged if the ensemble score exceeds a threshold optimized for F1 on the test set.

*(Note: TabNet and ECOD were considered and deliberately excluded to ensure the three-person team can deeply understand, tune, and defend every model in the system within the available timeframe, rather than spreading effort thinly across five. These are noted as future work.)*

---

## 5. Explainability Layer — SHAP

For every transaction the ensemble flags, SHAP (SHapley Additive exPlanations) values are computed on the LightGBM model, producing a human-readable breakdown:

```
Transaction #TXN_4829 — HIGH RISK (score: 0.91)
+0.34  Amount is 8.2x this card's historical average
+0.28  Email domain linked to 23 prior fraud cases
+0.19  Device/browser not seen for this card in 90 days
+0.14  Billing address reused across 7 transactions this week
-0.05  Transaction time (2 PM) is within normal hours
```

This output is human-readable and can be shown to a compliance officer, a fraud analyst, or directly to a customer — directly addressing the black-box compliance gap identified in the problem statement.

---

## 6. Credit Risk Module

Beyond a binary fraud flag, the system computes transaction-level risk scoring:

- **Risk tier:** LOW / MEDIUM / HIGH / CRITICAL
- **Estimated financial exposure:** based on transaction amount × fraud probability
- **Merchant category risk profile**
- **Card velocity score:** number of transactions on this card in the last hour/day

This is credit risk at the transaction level — the same concept payment processors use to set dynamic transaction limits.

---

## 7. System Layers

- **Data Layer:** IEEE-CIS (and optionally PaySim) ingestion, feature engineering pipeline, graph construction module
- **Model Layer:** Three trained models — LightGBM, Autoencoder, GAT — saved as `.pkl`/`.pt` files
- **Inference Layer:** FastAPI REST endpoint. Single transaction JSON in → fraud score + explanation JSON out.
- **Dashboard Layer:** Streamlit web application. Upload a CSV of transactions → visual fraud analysis. Individual transaction detail view with SHAP explanation chart.

*(Deployment kept lean: Docker, CI/CD, and cloud hosting are optional additions only if the core three models are validated with buffer time remaining. A working local Streamlit + FastAPI demo is fully sufficient and defensible for a viva.)*

---

## 8. Tech Stack

| Category | Tool | Why |
|---|---|---|
| Language | Python 3.11 | Standard for ML |
| Gradient Boosting | LightGBM | Best tabular fraud baseline |
| Anomaly Detection | Autoencoder (PyTorch/Keras) | Zero-day fraud detection |
| Graph ML | PyTorch + PyTorch Geometric (GAT) | Transaction graph learning |
| Graph Construction | NetworkX | Build transaction network |
| Explainability | SHAP | Regulatory-aligned explanations |
| Imbalance Handling | imbalanced-learn (SMOTE) | Handles rare fraud class |
| Hyperparameter Tuning | Optuna | Automated tuning |
| REST API | FastAPI | Production-grade inference |
| Dashboard | Streamlit | Interactive demo |
| Data Processing | Pandas, NumPy | Standard |
| Visualization | Plotly | Interactive charts |
| Environment | Google Colab / Kaggle Notebooks | Free-tier GPU for GAT/Autoencoder |

---

## 9. Team Breakdown (3 Members)

**Member 1 — Data, Feature Engineering & Anomaly Detection**
EDA, statistical validation, feature engineering, SMOTE, graph construction (card/email/device/address edges), Autoencoder training and evaluation.

**Member 2 — Supervised Classification & Explainability**
LightGBM training and tuning, evaluation (PR-AUC, F1), SHAP integration.

**Member 3 — Graph Learning, Fusion & Deployment**
GAT training on the transaction graph, ensemble fusion logic, FastAPI backend, Streamlit dashboard.

---

## 10. Project Timeline (16 Weeks)

| Weeks | Activity |
|---|---|
| 1–2 | Dataset exploration, EDA, literature review finalization, graph construction design |
| 3–5 | LightGBM training and tuning; Autoencoder training (parallel workstreams) |
| 6–9 | GAT training on transaction graph (hardest component — most time allocated) |
| 10–11 | Three-model ensemble fusion, weight tuning on validation set |
| 12 | SHAP explainability integration |
| 13–14 | FastAPI + Streamlit dashboard build; ablation study (individual vs. fused performance) |
| 15 | End-to-end testing, refinement, report writing |
| 16 | Buffer, final presentation and viva preparation |

---

## 11. Expected Results

- **AUC-ROC:** > 0.90 (LightGBM alone typically achieves ~0.89 on IEEE-CIS; fusion should exceed this)
- **F1 Score:** > 0.80 on the fraud class
- **Ablation table:** comparing LightGBM-only, Autoencoder-only, GAT-only, and fused ensemble performance — this quantified comparison is what elevates the report from "we built models" to "we demonstrate complementary value," and is the single strongest piece of evidence for both the viva and any future publication attempt
- **Explainability:** SHAP values for every flagged transaction
- **Working demo:** local FastAPI + Streamlit application

---

## 12. What Makes This Project Distinct

Most student fraud detection projects use the PCA-anonymized Kaggle credit card dataset, train one or two basic models, and stop at a confusion matrix. This project:

- Uses IEEE-CIS — richer, real industry-grade identifier data
- Builds an actual transaction graph for GAT-based fraud-ring detection, rather than treating this as a theoretical concept
- Combines three complementary detection paradigms — supervised (LightGBM), unsupervised (Autoencoder), and graph-based (GAT) — each with a distinct, justified role
- Implements SHAP explainability, directly addressing a documented regulatory and literature-identified gap
- Produces a working, demoable system (FastAPI + Streamlit), not only a notebook

---

## 13. Literature Grounding (Summary)

This project is positioned directly against a documented gap in the literature: systematic reviews (Danda & Haritha, 2026, Research Square; Chen et al., 2025, *Data Science and Management*) identify Graph Neural Networks and Explainable AI as the critical, underexplored direction for financial fraud detection, while applied studies in this space (UPI-focused ML papers, Stanford/VIT baseline studies) continue to rely on traditional tabular models only. No existing applied paper combines graph-based fraud-ring detection with SHAP-based explainability in a single deployed system — this is the specific gap this project addresses.

---

*This document reflects the final agreed project scope as of the current planning stage and is ready for mentor review and Review 1 submission.*
