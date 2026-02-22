"""Script de collecte de donnees historiques — multi-sources.

Usage:
    uv run python scripts/collect_historical.py              # Tout collecter
    uv run python scripts/collect_historical.py --prices     # Prix seulement
    uv run python scripts/collect_historical.py --news       # News GNews seulement
    uv run python scripts/collect_historical.py --alphavantage  # News Alpha Vantage
    uv run python scripts/collect_historical.py --marketaux  # News Marketaux
    uv run python scripts/collect_historical.py --rss        # News flux RSS
    uv run python scripts/collect_historical.py --all-news   # Toutes les sources news
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger

from src.core.database import Database
from src.data_collection.price_collector import PriceCollector
from src.data_collection.news_collector import NewsCollector
from src.data_collection.alpha_vantage_collector import AlphaVantageCollector
from src.data_collection.marketaux_collector import MarketauxCollector
from src.data_collection.rss_collector import RSSCollector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trades.db")


def _print_errors(errors, key="action"):
    """Affiche les erreurs de collecte."""
    if errors:
        print(f"Erreurs: {len(errors)}")
        for e in errors:
            print(f"  - {e.get(key, '?')}: {e.get('error', '?')}")


def main():
    parser = argparse.ArgumentParser(description="Collecte donnees historiques multi-sources")
    parser.add_argument("--prices", action="store_true", help="Prix OHLCV (yfinance)")
    parser.add_argument("--news", action="store_true", help="News GNews")
    parser.add_argument("--alphavantage", action="store_true", help="News Alpha Vantage (sentiment)")
    parser.add_argument("--marketaux", action="store_true", help="News Marketaux (sentiment)")
    parser.add_argument("--rss", action="store_true", help="News flux RSS francais")
    parser.add_argument("--all-news", action="store_true", help="Toutes les sources news")
    args = parser.parse_args()

    # Si aucun flag, tout collecter
    no_flag = not any([args.prices, args.news, args.alphavantage, args.marketaux, args.rss, args.all_news])
    collect_prices = args.prices or no_flag
    collect_gnews = args.news or args.all_news or no_flag
    collect_av = args.alphavantage or args.all_news or no_flag
    collect_mx = args.marketaux or args.all_news or no_flag
    collect_rss = args.rss or args.all_news or no_flag

    db = Database(DB_PATH)
    db.init_db()

    print(f"Base: {DB_PATH}")
    print(f"Trades en base: {db.count_trades()}")
    print(f"News deja en base: {db.count_news()}")
    print()

    if collect_prices:
        print("=" * 50)
        print("  [1/5] PRIX — yfinance")
        print("=" * 50)
        collector = PriceCollector(db)
        result = collector.collect_all()
        print(f"  Prix collectes: {result['total_prices']}")
        _print_errors(result["errors"])
        print()

    if collect_gnews:
        print("=" * 50)
        print("  [2/5] NEWS — GNews (Google News)")
        print("=" * 50)
        collector = NewsCollector(db)
        result = collector.collect_all()
        print(f"  News collectees: {result['total_news']}")
        _print_errors(result["errors"])
        print()

    if collect_av:
        print("=" * 50)
        print("  [3/5] NEWS — Alpha Vantage (sentiment)")
        print("=" * 50)
        collector = AlphaVantageCollector(db)
        result = collector.collect_all()
        print(f"  News collectees: {result['total_news']}")
        _print_errors(result["errors"])
        print()

    if collect_mx:
        print("=" * 50)
        print("  [4/5] NEWS — Marketaux (sentiment)")
        print("=" * 50)
        collector = MarketauxCollector(db)
        result = collector.collect_all()
        print(f"  News collectees: {result['total_news']}")
        _print_errors(result["errors"])
        print()

    if collect_rss:
        print("=" * 50)
        print("  [5/5] NEWS — Flux RSS francais")
        print("=" * 50)
        collector = RSSCollector(db)
        result = collector.collect_all()
        print(f"  News pertinentes: {result['total_news']}")
        print(f"  Feeds OK: {result['feeds_ok']}/{result['feeds_ok'] + result['feeds_error']}")
        _print_errors(result["errors"], key="feed")
        print()

    print("=" * 50)
    print("  BILAN FINAL")
    print("=" * 50)
    print(f"  Total prix en base: {db.count_prices()}")
    print(f"  Total news en base: {db.count_news()}")

    # Stats par source
    conn = db._connect()
    rows = conn.execute(
        "SELECT source_api, COUNT(*) as nb FROM news GROUP BY source_api ORDER BY nb DESC"
    ).fetchall()
    conn.close()
    print()
    print("  News par source:")
    for r in rows:
        print(f"    {r['source_api'] or 'gnews':20} {r['nb']:5} articles")


if __name__ == "__main__":
    main()
