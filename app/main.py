# app/main.py
from fastapi import FastAPI
from app.api.routes import router
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("liveness.log", encoding="utf-8"),
    ],
)



app = FastAPI(title="Liveness Anti-Spoofing Service", version="0.1.0")
app.include_router(router)
