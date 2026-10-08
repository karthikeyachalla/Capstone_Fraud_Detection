"""
04_kafka_producer.py
====================
Real Apache Kafka Message Producer for Financial Transaction Ingestion.
Publishes genuine IEEE-CIS transactions into the authentic Kafka Broker on localhost:9092.
Topic: financial.transactions.stream (3 Partitions, Keyed by card1 entity)
"""

import os
import sys
import time
import json
import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "train_transaction.csv")
KAFKA_BROKER = os.getenv("KAFKA_BOOTSTRAP_SERVER", "localhost:9092")
TOPIC_NAME = "financial.transactions.stream"


def create_producer(retries: int = 5, retry_delay: int = 2) -> KafkaProducer:
    """Connects to the live local Apache Kafka broker on port 9092."""
    for attempt in range(1, retries + 1):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[KAFKA_BROKER],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
                acks="all",
                retries=3,
                linger_ms=10
            )
            print(f"✅ Connected to Apache Kafka Broker at {KAFKA_BROKER}")
            return producer
        except Exception as ex:
            print(f"⏳ Attempt {attempt}/{retries}: Kafka broker error ({ex}). Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
    raise RuntimeError(f"❌ Could not connect to Kafka Broker at {KAFKA_BROKER}. Is Kafka running?")


def run_kafka_producer(n_events: int = 50, delay_sec: float = 0.15):
    print("=" * 80)
    print("🚀 APACHE KAFKA REAL-TIME TRANSACTION PRODUCER")
    print(f"   Target Broker : {KAFKA_BROKER}")
    print(f"   Target Topic  : {TOPIC_NAME}")
    print(f"   Dataset Source: IEEE-CIS Financial Fraud Benchmark")
    print("=" * 80)

    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset not found at: {DATA_PATH}")
        return

    try:
        producer = create_producer()
    except Exception as e:
        print(f"{e}")
        return

    print(f"\n📂 Reading {n_events} authentic transaction records from IEEE-CIS dataset...")
    df = pd.read_csv(DATA_PATH, nrows=n_events)

    sent_count = 0
    start_time = time.time()

    for idx, row in df.iterrows():
        tx_id = str(row.get("TransactionID", f"TXN_{idx+1000}"))
        amt = float(row.get("TransactionAmt", 100.0))
        card_id = int(row.get("card1", 13500)) if pd.notnull(row.get("card1")) else 13500
        card_network = str(row.get("card4", "visa")) if pd.notnull(row.get("card4")) else "visa"
        addr_zone = int(row.get("addr1", 315)) if pd.notnull(row.get("addr1")) else 315

        payload = row.to_dict()
        # Clean any float NaNs for strict JSON serialization
        clean_payload = {k: (0.0 if (isinstance(v, float) and pd.isna(v)) else v) for k, v in payload.items()}
        clean_payload["TransactionID"] = tx_id
        clean_payload["TransactionAmt"] = amt
        clean_payload["card1"] = card_id
        clean_payload["card4"] = card_network
        clean_payload["addr1"] = addr_zone
        clean_payload["kafka_producer_ts"] = time.time()

        # Keying by card_id ensures transactions from the same card route to the same Kafka partition
        future = producer.send(TOPIC_NAME, key=str(card_id), value=clean_payload)
        record_metadata = future.get(timeout=10)

        sent_count += 1
        t_now = time.strftime("%H:%M:%S")

        print(f"[{t_now}] [Kafka-P{record_metadata.partition}] Offset: {record_metadata.offset:<6} | "
              f"Tx: {tx_id:<12} | Amt: ${amt:>8.2f} | Card: {card_network.upper()} #{card_id:<5} | "
              f"Keyed Partition -> P{record_metadata.partition}")

        time.sleep(delay_sec)

    producer.flush()
    producer.close()

    elapsed = time.time() - start_time
    tps = sent_count / max(0.001, elapsed)

    print("\n" + "=" * 80)
    print("✅ KAFKA PRODUCER EXECUTION COMPLETE")
    print(f"   • Total Messages Published: {sent_count}")
    print(f"   • Ingestion Throughput     : {tps:.1f} events/sec")
    print(f"   • Target Kafka Topic       : {TOPIC_NAME}")
    print("=" * 80)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    delay = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
    run_kafka_producer(n_events=count, delay_sec=delay)
