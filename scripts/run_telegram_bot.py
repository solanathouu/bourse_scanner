"""Bot Telegram interactif — ecoute les reponses de Nicolas.

Lance le bot en polling pour recevoir les clics PRIS/PASSE
et les messages libres (TP, SL, commentaires).

Usage:
    uv run python scripts/run_telegram_bot.py
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.database import Database
from src.alerts.telegram_interactive import TelegramInteractive

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trades.db")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "telegram_bot.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger.add(LOG_PATH, rotation="10 MB", retention="7 days", level="INFO")


def main():
    db = Database(DB_PATH)
    db.init_db()

    interactive = TelegramInteractive(db)

    print("Bot Telegram interactif demarre")
    print("En attente de reponses PRIS/PASSE et messages...")
    print("Ctrl+C pour arreter\n")

    app = interactive.create_application()
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
