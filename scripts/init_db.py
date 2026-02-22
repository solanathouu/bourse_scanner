"""Initialise la base de données SQLite avec les tables nécessaires."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from loguru import logger


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "trades.db")
    db = Database(db_path)
    db.init_db()
    logger.info(f"Base de données créée/vérifiée: {db_path}")


if __name__ == "__main__":
    main()
