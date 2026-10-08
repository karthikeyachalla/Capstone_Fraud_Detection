# 📚 Data Dictionary: Indian Banking & UPI Fraud Dataset

Dataset Location: `data/raw/Fraud_Detection_Dataset.csv`  
Records: 15,000 Transactions | Target: `fraud_label` (Fraudulent: 1, Legitimate: 0)

---

## 1. Feature Specifications

| Field Name | Type | Description | Values / Range | Fraud Relevance |
| :--- | :--- | :--- | :--- | :--- |
| `transaction_id` | String | Unique transaction alphanumeric code | `TXN1000000`... | Primary Key |
| `customer_id` | String | Customer account unique identifier | `CUST46048`... | Node in Graph Network |
| `bank_name` | Categorical | Originating banking institution | SBI, HDFC, ICICI, Axis, Canara, BoB | Institutional risk baselines |
| `account_type` | Categorical | Account category | Current, Savings, Salary | Baseline transaction patterns |
| `transaction_amount` | Float | Transfer value in Indian Rupees (₹) | ₹1.00 – ₹1,00,000.00+ | Large or anomalous spikes |
| `currency` | Categorical | Payment currency | INR | Domestic payment scope |
| `payment_method` | Categorical | Digital payment channel | UPI, IMPS, RTGS, Net Banking, Card | Channel-specific vulnerability |
| `upi_app` | Categorical | Client payment application | Google Pay, PhonePe, Paytm, BHIM, Amazon Pay, Navi | App ecosystem tracking |
| `transaction_type` | Categorical | Nature of transaction | Purchase, Recharge, Bill Payment, P2P | Transaction intent context |
| `transaction_hour` | Integer | Hour of day (24-hour clock) | 0 to 23 | Off-peak / late-night anomalies |
| `account_balance` | Float | Remaining funds in customer account (₹) | ₹0.00 – ₹10,00,000.00+ | Account draining detection |
| `login_attempts` | Integer | Total app logins in last session | 1 to 10 | Brute force / credential stuffing |
| `failed_login_count` | Integer | Unsuccessful authentication attempts | 0 to 5 | High indicator of takeover |
| `device_trust_score` | Float | Behavioral hardware integrity score | 0.0 to 100.0 | Spoofed / rooted device detection |
| `ip_risk_score` | Float | IP reputation and VPN detection | 0.0 to 100.0 | High score = proxy / known malicious subnet |
| `beneficiary_bank` | Categorical | Destination bank receiving funds | Kotak, SBI, ICICI, HDFC, Indian Bank | Mule network graph destination |
| `beneficiary_age_days`| Integer | Days since beneficiary account creation | 1 to 4,000 days | **Key Mule Signal:** <30 days = high risk |
| `authentication_method`| Categorical | Credential verification type | UPI PIN, MPIN, OTP, Biometric | Authentication strength |
| `location_match` | Boolean | Does GPS city match IP registration? | Yes, No | Location mismatch alert |
| `new_device` | Boolean | Is this transaction from a new device? | Yes, No | First-time hardware risk |
| `fraud_label` | Categorical | Ground truth classification | Fraudulent / Legitimate | Binary Target ($Y$) |
