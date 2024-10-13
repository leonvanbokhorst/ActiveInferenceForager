import logging
from loguru import logger

def setup_logging():
    logger.remove()
    logger.add(
        "active_inference_forager.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        level="INFO",
    )
    logging.basicConfig(
        filename="active_inference_forager.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
