# Capstone Project Final Submission Email Draft

---

**To:** dileep.kumar@woxsen.edu.in *(or your mentor's official email)*  
**Cc:** *(optional: your team members' emails)*  
**Subject:** Final Capstone Project Submission & Live Cloud Demo — Real-Time Financial Fraud Intelligence Gateway | Team 23WU0104013

---

**Respected Prof. Dileep Kumar M.,**

We hope this email finds you well.

We are writing to formally submit the final deliverables for our Capstone Project:  
**"Real-Time Financial Fraud Intelligence Gateway — High-Throughput Autonomous Detection & Regulatory Compliance Pipeline"**  
*(Woxsen University — Final Year B.Tech Capstone Project — I, AY 2026–27)*.

All project models, microservices, cloud deployments, and interactive dashboards are fully developed, verified, and live for your evaluation.

---

### 🌐 1. Live Deployment & Repository Links

* **Live Cloud Interactive Dashboard:**  
  👉 **https://capstone-fraud-detection-3wup.onrender.com**  
  *(Hosted in cloud production with real-time Kafka simulation, Tri-Model AI screening, TreeSHAP mathematical explainability, and regulatory audit trail).*

* **Interactive OpenAPI / Swagger Documentation:**  
  👉 **https://capstone-fraud-detection-3wup.onrender.com/docs**  
  *(Live API endpoints for `/predict`, `/stream/next`, `/stream/inject-attack`, `/health`, and `/metrics`).*

* **GitHub Source Code Repository:**  
  👉 **https://github.com/karthikeyachalla/Capstone_Fraud_Detection**  
  *(Contains complete training pipelines, GAT models, test suites, Dockerfile, and architecture specs).*

---

### 🧪 2. How to Test the Live Dashboard

1. **Open the Live Link:**  
   When opened, the dashboard starts in **Standby Mode** (0 transactions) to give you complete manual control.
2. **Start the Stream:**  
   Click the large **`▶ START LIVE DETECTION STREAM`** button to initiate transaction processing (1 transaction ingested and screened every 1.8 seconds).
3. **Inspect Models & XAI:**  
   Click on any transaction row in the live feed table to inspect:
   * **TreeSHAP Explainability:** Mathematical decomposition of top risk factors.
   * **Regulatory Audit Certificate:** Full cryptographic audit record complying with US FCRA § 1681m and OCC SR 11-7 standards.
4. **Simulate Adversarial Scenarios:**  
   You can click any of the scenario buttons to test specific payment flows:
   * **`Coffee $18.50`** — Instant sub-5ms low-risk Fast-Path authorization.
   * **`Geo-Hop $750`** — Elevated anomaly triggering mandatory 2FA OTP step-up.
   * **`Botnet Burst $1,450`** — Coordinated mule attack intercepted and blocked by the PyTorch GAT graph.
5. **Pause Anytime:**  
   Click **`⏸ PAUSE LIVE DETECTION STREAM`** to freeze the stream for in-depth inspection.

---

### 📊 3. Key Technical & Quantitative Achievements

* **Tri-Model Ensemble Architecture:** Combines a cost-sensitive **LightGBM Tabular Booster**, a **PyTorch Deep AutoEncoder** (unsupervised anomaly detection), and a **PyTorch Multi-Head Graph Attention Network (GAT)** (relational mule-ring detection).
* **Performance Metrics (IEEE-CIS Benchmark):**
  * **ROC-AUC:** **0.9412**
  * **PR-AUC:** **0.8863** (a **+19.4% relative gain** over single-model baselines)
  * **Inference Latency:** **~5.6 ms median** (well within the strict < 15ms banking switch SLA).
* **Regulatory & Compliance Layer:** Automated explainability (TreeSHAP) with deterministic reason codes to satisfy anti-black-box banking regulations.

---

### 📁 4. Attached Project Documents

We have attached the complete project documentation to this submission:
1. `Capstone_Project_I_Review_III_V3.pdf` — Comprehensive Final Project Report.
2. `Capstone_Review3_Presentation.pdf` — Final Review & Defense Slide Deck.

Please let us know if any further information is required or if you would like a scheduled demonstration.

Thank you very much for your invaluable guidance, feedback, and mentorship throughout this project.

With sincere regards,

**Student Team:**  
* **Challa Karthikeya** (ID: 23WU0104013) — *Lead Developer*  
* **Y. Brijesh Naidu** (ID: 23WU0104011)  
* **K. Dheeraj** (ID: 23WU0104039)  

Department of Computer Science & Engineering  
School of Technology, Woxsen University  
