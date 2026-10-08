# FINAL CAPSTONE VIVA DEFENSE: QUESTIONS & ANSWERS (Q&A)
## School of Technology, Woxsen University — Academic Year 2026–2027
### Project: Real-Time Financial Fraud Intelligence & Graph Attention Network (GAT) Surveillance

---

### Q1: "What dataset are you using, and why?"
**Candidate Answer:**
> "We evaluate our architecture on the **IEEE-CIS Financial Fraud Benchmark**, provided by Vesta Corporation. It contains **590,540 authentic financial transactions** across 446 features, including card entities (`card1` to `card6`), billing locations (`addr1`, `addr2`), velocity counters (`C1` to `C14`), and device fingerprints (`DeviceInfo`, `DeviceType`, `id_01` to `id_38`). It is the global gold standard in academic financial fraud research, featuring a realistic ~3.5% baseline fraud rate."

---

### Q2: "A CSV file is a flat table. How do you convert it into a graph for a GNN?"
**Candidate Answer:**
> "Standard tabular models evaluate each transaction $x_i$ in isolation. However, in card fraud, criminals reuse stolen cards and hardware devices. We construct a bipartite relational graph where:
> 1. **Nodes:** Individual transactions.
> 2. **Edges:** Placed between transactions that share identical entities:
>    - Same `card1` (shared cardholder entity velocity).
>    - Same `DeviceInfo` (shared hardware device syndicate).
>    - Same `addr1` within tight temporal proximity.
> Our Multi-Head Graph Attention Network (GAT) computes self-attention weights $\alpha_{ij}$ over these connected neighbors, pulling structural risk into a 32-dimensional Relational Risk Embedding (RRE)."

---

### Q3: "What are the empirical results? Did GNN actually improve fraud detection?"
**Candidate Answer:**
> "Yes, rigorously proven in our 5-model ablation study on the test split:
> - **Raw Tabular LightGBM:** Achieved 0.9098 ROC-AUC with **52.63% Recall** and **31.35% Precision** ($F_1 = 0.3929$).
> - **Pure Multi-Head GAT:** Achieved **72.11% Recall** ($0.7971\text{ ROC-AUC}$), capturing coordinated carding rings that tabular trees completely missed.
> - **Proposed H-GATBoost Fusion (Champion):** By fusing tabular boosting with GAT relational embeddings and AutoEncoder reconstruction anomalies, our model jumped to **0.9412 ROC-AUC**, **69.47% Recall**, **53.88% Precision**, and **0.6069 F1-score** — a **+54.5% relative improvement in F1-score** over tabular baselines."

---

### Q4: "What is the role of the Autoencoder? Isn't it just feature extraction?"
**Candidate Answer:**
> "The PyTorch Autoencoder ($419 \to 128 \to 64 \to 32 \to 64 \to 128 \to 419$) acts as an **unsupervised zero-day anomaly detector**. It is trained strictly on legitimate card transactions to minimize reconstruction Mean Squared Error (MSE). When a novel or unmodeled attack variant arrives that does not match historical decision tree splits, the Autoencoder cannot reconstruct it, producing a high reconstruction loss ($MSE > 2.0$) that alerts the ensemble."

---

### Q5: "How does the system achieve real-time latency (<15ms SLA)?"
**Candidate Answer:**
> "Running a full-batch graph convolution across 500k nodes during live transaction authorization takes $>100\text{ ms}$, which would violate banking SLAs. We overcome this via a two-tier serving pipeline:
> - The multi-agent gateway uses vectorized NumPy conversions and isolated single-step forward passes.
> - On our 500-transaction live latency benchmark on standard CPU hardware, the median serving latency is **~5.6 ms**, with a mean of **9.7 ms**, safely operating within the enterprise $<15\text{ ms}$ SLA."

---

### Q6: "How do you handle Concept Drift over time?"
**Candidate Answer:**
> "Financial data undergoes continuous distribution shift. We utilize the chronological `TransactionDT` field (time in seconds over 182 days) to partition traffic into temporal sliding windows. Our monitor computes two statistical tests:
> 1. **Two-Sample Kolmogorov-Smirnov (KS) Test:** Flags distributional divergence on key features ($p < 0.05$).
> 2. **Population Stability Index (PSI):** Measures shift magnitude ($PSI < 0.10$ stable, $0.10 \le PSI < 0.25$ moderate alert, $PSI \ge 0.25$ triggers automated model retraining)."

---

### Q7: "How is the system compliant with banking explainability regulations?"
**Candidate Answer:**
> "Under RBI Master Directions (2024) and EU AI Act Article 14, automated adverse actions (CHALLENGE or BLOCK) must provide transparent justifications. Our Explainability Agent computes instant **TreeSHAP** local Shapley attributions, identifying the exact top risk drivers (e.g. `TransactionAmt`, `card1`, `addr1`) and logging compliant audit trails."
