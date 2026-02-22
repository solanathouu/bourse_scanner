"""Script de collecte de donnees historiques.

Usage:
    uv run python scripts/collect_historical.py          # Tout collecter
    uv run python scripts/collect_historical.py --prices  # Seulement les prix
    uv run python scripts/collect_historical.py --news    # Seulement les news
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from src.core.database import Database
from src.data_collection.price_collector import PriceCollector
from src.data_collection.news_collector import NewsCollector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trades.db")


def main():
    parser = argparse.ArgumentParser(description="Collecte donnees historiques")
    parser.add_argument("--prices", action="store_true", help="Collecter les prix")
    parser.add_argument("--news", action="store_true", help="Collecter les news")
    args = parser.parse_args()

    # Si aucun flag, tout collecter
    collect_prices = args.prices or (not args.prices and not args.news)
    collect_news = args.news or (not args.prices and not args.news)

    db = Database(DB_PATH)
    db.init_db()

    print(f"Base: {DB_PATH}")
    print(f"Trades en base: {db.count_trades()}")
    print()

    if collect_prices:
        print("=== Collecte des prix ===")
        collector = PriceCollector(db)
        result = collector.collect_all()
        print(f"Prix collectes: {result['total_prices']}")
        if result["errors"]:
            print(f"Erreurs: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['action']}: {e['error']}")
        print()

    if collect_news:
        print("=== Collecte des news ===")
        collector = NewsCollector(db)
        result = collector.collect_all()
        print(f"News collectees: {result['total_news']}")
        if result["errors"]:
            print(f"Erreurs: {len(result['errors'])}")
            for e in result["errors"]:
                print(f"  - {e['action']}: {e['error']}")
        print()

    print("=== Bilan ===")
    print(f"Total prix en base: {db.count_prices()}")
    print(f"Total news en base: {db.count_news()}")


if __name__ == "__main__":
    main()
