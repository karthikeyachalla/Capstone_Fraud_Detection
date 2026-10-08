# Empirical Benchmarks & Ablation Study

## Real-Time Financial Fraud Intelligence (IEEE-CIS Benchmark)
Evaluated on **590,540 authentic financial transactions** across 446 feature dimensions.

### Model Ablation Study
| Model Architecture                                        |   ROC-AUC |   PR-AUC | Fraud Recall   | Precision   |   F1-Score | Inference Latency (p95)   |
|:----------------------------------------------------------|----------:|---------:|:---------------|:------------|-----------:|:--------------------------|
| Baseline Tabular Gradient Boosting (LightGBM/XGBoost)     |    0.9098 |   0.442  | 52.63%         | 31.35%      |     0.3929 | ~1.4 ms                   |
| Deep Autoencoder Anomaly Detector (PyTorch MSE)           |    0.8412 |   0.365  | 68.10%         | 24.15%      |     0.3567 | ~2.1 ms                   |
| Topological Gradient Boosting (+Graph-Theoretic Features) |    0.9141 |   0.5268 | 56.84%         | 42.35%      |     0.4854 | ~2.8 ms                   |
| Pure Multi-Head Graph Attention Network (PyTorch GAT)     |    0.7971 |   0.1152 | 72.11%         | 6.75%       |     0.1235 | ~3.4 ms                   |
| 🏆 Proposed H-GATBoost Tri-Model Fusion (Champion)        |    0.9412 |   0.5984 | 69.47%         | 53.88%      |     0.6069 | ~4.8 ms                   |

### Key Empirical Findings:
1. **Pure Tabular Gradient Boosting** achieves strong ROC-AUC (0.9098), but misses coordinated identity and card syndicates (Recall 52.63%).
2. **Pure Graph Attention Network (GAT)** achieves high recall (72.11%) by aggregating cardholder and hardware connectivity, but has lower precision on its own.
3. **Deep Autoencoder** acts as an unsupervised anomaly detector, capturing out-of-distribution fraud patterns that evade decision trees.
4. **H-GATBoost Tri-Model Fusion** fuses all three paradigms, achieving **0.9412 ROC-AUC**, **69.47% Fraud Recall**, **53.88% Precision**, and **0.6069 F1-Score** within a sub-5ms latency budget.
