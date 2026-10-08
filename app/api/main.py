"""
api/main.py
===========
Production FastAPI Gateway entry point for web dashboard and external clients.
Delegates cleanly to src.fastapi_app to provide unified 4-Agent Risk Intelligence.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from fastapi_app import app, TransactionPayload, CreditApplicationPayload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
