"""
tests/test_pipeline.py
======================
Automated test suite verifying:
  1. Health check & model readiness
  2. Safe transaction approval policy (Agent 1 + Agent 2)
  3. Fraudulent transaction blocking & XAI explainability (Agent 1 + Agent 2 + Agent 3)
  4. Credit and loan underwriting eligibility (Agent 4)
Compatible with both pytest and python -m unittest.
"""

import unittest
from fastapi.testclient import TestClient
from src.fastapi_app import app

class TestFraudDetectionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("Real-Time Financial Fraud", data["system"])

    def test_02_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
        self.assertTrue(response.json()["multi_agent_system_ready"])

    def test_03_safe_transaction_approval(self):
        payload = {
            "transaction_id": "TXN_TEST_SAFE_01",
            "transaction_amount": 500.0,
            "payment_method": "UPI",
            "upi_app": "GooglePay",
            "bank_name": "HDFC Bank",
            "new_device": "No",
            "location_match": "Yes",
            "transaction_hour": 14
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["risk_tier"], "LOW_SAFE")
        self.assertEqual(res["action"], "APPROVE_TRANSACTION")
        self.assertLess(res["fraud_probability"], 0.35)

    def test_04_high_risk_transaction_blocking(self):
        payload = {
            "transaction_id": "TXN_TEST_FRAUD_99",
            "transaction_amount": 88000.0,
            "payment_method": "UPI",
            "upi_app": "PhonePe",
            "bank_name": "SBI",
            "new_device": "Yes",
            "location_match": "No",
            "failed_login_count": 3,
            "transaction_hour": 2
        }
        response = self.client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["risk_tier"], "CRITICAL_HIGH")
        self.assertEqual(res["action"], "BLOCK_TRANSACTION_IMMEDIATELY")
        self.assertGreaterEqual(res["fraud_probability"], 0.75)
        self.assertGreaterEqual(len(res["audit_explanations"]), 3)

    def test_05_credit_underwriting_approval(self):
        payload = {
            "applicant_name": "Karthikeya Challa",
            "monthly_income": 90000.0,
            "cibil_score": 780,
            "existing_monthly_emi": 10000.0,
            "requested_loan_amount": 400000.0
        }
        response = self.client.post("/credit-underwrite", json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["application_status"], "APPROVED")
        self.assertIn("PRIME", res["credit_tier"])
        self.assertGreater(res["max_recommended_credit_limit"], 0)

    def test_06_credit_underwriting_rejection(self):
        payload = {
            "applicant_name": "Subprime Applicant",
            "monthly_income": 25000.0,
            "cibil_score": 580,
            "existing_monthly_emi": 18000.0,
            "requested_loan_amount": 600000.0
        }
        response = self.client.post("/credit-underwrite", json=payload)
        self.assertEqual(response.status_code, 200)
        res = response.json()
        self.assertEqual(res["application_status"], "REJECTED")
        self.assertIn("SUBPRIME", res["credit_tier"])

    def test_07_gat_mule_embeddings_cache_loaded(self):
        from src.fastapi_app import mas
        self.assertIsNotNone(mas.predictor.mule_cache)
        self.assertIn("customer_embeddings", mas.predictor.mule_cache)
        self.assertGreater(mas.predictor.mule_cache["total_nodes_cached"], 0)

    def test_08_concept_drift_monitor_execution(self):
        import importlib
        import numpy as np
        drift_mod = importlib.import_module("src.05_concept_drift_monitor")
        calculate_psi = drift_mod.calculate_psi

        baseline = np.random.normal(0, 1, 1000)
        same_dist = np.random.normal(0, 1, 1000)
        drifted_dist = np.random.normal(3, 1, 1000)

        psi_low = calculate_psi(baseline, same_dist)
        psi_high = calculate_psi(baseline, drifted_dist)
        self.assertLess(psi_low, 0.10)
        self.assertGreater(psi_high, 0.20)

    def test_09_kafka_stream_payload_generation(self):
        import importlib
        sim_mod = importlib.import_module("src.04_kafka_stream_simulator")
        events, source = sim_mod.get_transaction_stream(n_events=5)
        self.assertEqual(len(events), 5)
        self.assertIn("transaction_amount", events[0])
        self.assertIn("payment_method", events[0])


if __name__ == "__main__":
    unittest.main()

