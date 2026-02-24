"""Analyse LLM des trades de Nicolas via GPT-4o-mini.

Usage:
    uv run python scripts/analyze_trades_llm.py           # Analyser tous les trades
    uv run python scripts/analyze_trades_llm.py --trade 42  # Analyser un trade specifique
    uv run python scripts/analyze_trades_llm.py --stats     # Stats des analyses existantes
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.analysis.llm_analyzer import LLMAnalyzer

DB_PATH = "data/trades.db"


def print_stats(db: Database):
    """Affiche les stats des analyses LLM existantes."""
    analyses = db.get_all_trade_analyses()
    total_trades = db.count_trades()

    print("=" * 60)
    print("ANALYSES LLM — Statistiques")
    print("=" * 60)
    print(f"\nTrades analyses: {len(analyses)}/{total_trades}")

    if not analyses:
        print("Aucune analyse. Lancez sans --stats pour analyser.")
        return

    # Distribution des types
    types = {}
    qualities = {}
    for a in analyses:
        t = a["catalyst_type"]
        types[t] = types.get(t, 0) + 1
        q = a["trade_quality"]
        qualities[q] = qualities.get(q, 0) + 1

    print(f"\nTypes de catalyseurs:")
    for t, count in sorted(types.items(), key=lambda x: -x[1]):
        print(f"  {t:20s}: {count:3d} ({100*count/len(analyses):.1f}%)")

    print(f"\nQualite des trades:")
    for q in ["EXCELLENT", "BON", "MOYEN", "MAUVAIS"]:
        count = qualities.get(q, 0)
        print(f"  {q:20s}: {count:3d} ({100*count/len(analyses):.1f}%)")

    # Confiance moyenne
    avg_conf = sum(a["catalyst_confidence"] for a in analyses) / len(analyses)
    print(f"\nConfiance moyenne: {avg_conf:.2f}")


def print_trade_analysis(db: Database, trade_id: int):
    """Affiche l'analyse LLM d'un trade specifique."""
    analysis = db.get_trade_analysis(trade_id)
    if not analysis:
        print(f"Pas d'analyse pour le trade #{trade_id}")
        return

    print(f"\n=== Trade #{trade_id} — Analyse LLM ===")
    print(f"Type catalyseur: {analysis['catalyst_type']}")
    print(f"Confiance:       {analysis['catalyst_confidence']:.2f}")
    print(f"Sentiment:       {analysis['news_sentiment']:.2f}")
    print(f"Qualite:         {analysis['trade_quality']}")
    print(f"\nResume: {analysis['catalyst_summary']}")
    print(f"\nRaison achat: {analysis['buy_reason']}")
    print(f"Raison vente: {analysis['sell_reason']}")


def main():
    parser = argparse.ArgumentParser(description="Analyse LLM des trades")
    parser.add_argument("--trade", type=int, help="Analyser un trade specifique")
    parser.add_argument("--stats", action="store_true", help="Stats seulement")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()

    if args.stats:
        print_stats(db)
        return

    if args.trade:
        # Analyser un seul trade
        analyzer = LLMAnalyzer(db)
        trades = db.get_all_trades()
        trade = next((t for t in trades if t["id"] == args.trade), None)
        if not trade:
            print(f"Trade #{args.trade} non trouve")
            return
        print(f"Analyse du trade #{args.trade}...")
        analyzer.analyze_trade(trade)
        print_trade_analysis(db, args.trade)
        return

    # Analyser tous les trades
    analyzer = LLMAnalyzer(db)
    print("Analyse LLM de tous les trades...")
    print("(reprise incrementale: les trades deja analyses sont ignores)\n")
    summary = analyzer.analyze_all_trades()
    print(f"\nResume: {summary['analyzed']} analyses, "
          f"{summary['skipped']} deja faits, {summary['errors']} erreurs")
    print_stats(db)


if __name__ == "__main__":
    main()
