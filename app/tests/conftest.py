# app/tests/conftest.py
import logging
import pytest

@pytest.fixture(autouse=True)
def _configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(name)s | %(message)s",
    )
    # Por si el root logger ya estaba configurado:
    logging.getLogger("liveness").setLevel(logging.INFO)
