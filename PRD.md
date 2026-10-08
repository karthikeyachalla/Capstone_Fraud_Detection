# 📋 Product Requirements Document (PRD)
## Real-Time Digital Banking & Instant Fund Transfer Fraud Intelligence System

---

## 1. Executive Summary & Problem Statement
Digital banking payment networks (such as India's **UPI, IMPS, and Mobile Banking**) process billions of instant retail transactions with strict settlement windows. This rapid digital adoption has amplified sophisticated financial crime vectors:
* **Account Takeovers (ATO) & Phishing:** Stolen MPINs, session hijacking, and social engineering draining customer balances in seconds.
* **Money Mule Networks & Syndicates:** Stolen funds rapidly split and routed into newly opened intermediary accounts (mules) across different financial institutions.
* **Sub-50ms Latency SLA:** Core payment switches (e.g., NPCI, CBS) require fraud and risk decisions in under 50ms, precluding sequential, heavy deep graph traversals on the critical path.
* **Dual Risk Exposure (Fraud + Credit):** Fraud rings exploit instant digital credit lines (pre-approved overdrafts, Buy-Now-Pay-Later) using synthetic identities, leading to immediate defaults.
* **Regulatory Compliance (RBI & DPDP Act 2023):** Mandatory explainability (XAI) for any automated block, decline, or credit denial.

This platform provides an **authentic, real-time banking fraud prevention and credit risk gateway** engineered specifically for high-velocity instant fund transfer rails.

---

## 2. Target Users & Beneficiaries

* **Everyday Banking Customers:** Protected from unauthorized account drainage and unfair transaction declines; provided with clear explanations for any step-up challenge.
* **Bank Fraud & Risk Operations:** Reduces manual review false alarms by 70%+, automatically surfaces money mule rings, and minimizes unrecoverable losses.
* **Credit Underwriting Committee:** Protects instant lending portfolios against synthetic identity bust-out syndicates.
* **Regulatory & Audit Teams:** Meets RBI Master Directions, DPDP Act 2023, and Basel III standards with auditable mathematical feature attributions.

---

## 3. Core Functional Requirements

### 3.1 Digital Banking Fund Transfer Coverage
The system evaluates transactions across digital banking payment rails:
* **Instant Account Transfers:** UPI (Google Pay, PhonePe, Paytm, BHIM) and IMPS.
* **Core Banking Transfers:** Mobile Banking and Net Banking fund movements.
* **Key Risk Entities:** Sender Account (`nameOrig`), Receiver/Beneficiary Account (`nameDest`), Device Fingerprint (`device_id`), and IP Geohash.

### 3.2 Dual-Tier Hybrid Intelligence Architecture
* **Tier 0: Deterministic Safety Rails (<1ms):** Rule checks for impossible geo-velocity and blacklisted devices/accounts.
* **Tier 1: Online Real-Time Gateway (<5ms):** Cost-Sensitive LightGBM scoring amounts, balance-to-amount ratios, and velocity under asymmetric loss weighting.
* **Tier 2: Near-Line Relational Graph Engine (GNN):** PyTorch Geometric Graph Attention Network (GAT) analyzing Sender ↔ Beneficiary bipartite topologies to identify money mule rings; pre-computes 32-dim graph risk embeddings cached in memory/Redis (<2ms lookup).
* **Tier 3: Unsupervised Anomaly Detection:** Deep Autoencoder trained on normal legitimate transfers; flags novel zero-day attack patterns via Reconstruction Error (MSE).
* **Tier 4: Regulatory Explainability (XAI):** TreeSHAP generating instant top-3 feature attributions for challenged or blocked transactions.
* **Tier 5: Streaming Concept Drift Monitor:** ADWIN and Kolmogorov-Smirnov (KS) streaming window tests detecting evolving fraud strategies.

### 3.3 Credit Risk Intelligence Engine
* Evaluates Probability of Default (PD) for instant digital credit applications.
* Checks applicant subgraphs for links to known default clusters or shared device syndicates.

### 3.4 Three-Stage Triage Policy
* **Tier I: Auto-Approve ($P(\text{fraud}) < 0.25$ & Low Credit Risk):** Processed seamlessly.
* **Tier II: Step-Up Challenge ($0.25 \le P(\text{fraud}) < 0.70$):** Triggers instant 2FA OTP / Biometric verification.
* **Tier III: Immediate Block ($P(\text{fraud}) \ge 0.70$ or Mule Hub Match):** Transaction blocked instantly, account flagged, and audit alert queued.

---

## 4. Technical Constraints & SLAs
* **Inference Latency:** p50 $< 10\text{ms}$, p95 $< 30\text{ms}$ on CPU/RAM.
* **System Availability:** High-resilience circuit-breaker pattern guaranteeing fallback operation under component timeouts.
* **Class Imbalance Strategy:** Cost-Sensitive Learning and asymmetric loss penalization over synthetic oversampling.
