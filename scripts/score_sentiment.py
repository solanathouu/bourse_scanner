"""Score le sentiment des news sans sentiment via GPT-4o-mini.

Usage:
    uv run python scripts/score_sentiment.py           # Scorer toutes les news
    uv run python scripts/score_sentiment.py --stats    # Distribution du sentiment
    uv run python scripts/score_sentiment.py --dry-run  # Combien a scorer
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.analysis.llm_sentiment import LLMSentimentScorer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trades.db")


def main():
    parser = argparse.ArgumentParser(description="Scoring sentiment LLM des news")
    parser.add_argument("--stats", action="store_true", help="Distribution du sentiment")
    parser.add_argument("--dry-run", action="store_true", help="Combien a scorer (sans executer)")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()

    if args.stats:
        conn = db._connect()
        total = conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        with_sent = conn.execute("SELECT COUNT(*) FROM news WHERE sentiment IS NOT NULL").fetchone()[0]
        without_sent = conn.execute("SELECT COUNT(*) FROM news WHERE sentiment IS NULL").fetchone()[0]
        print(f"Total news: {total}")
        print(f"  Avec sentiment: {with_sent} ({with_sent/total*100:.1f}%)" if total else "")
        print(f"  Sans sentiment: {without_sent} ({without_sent/total*100:.1f}%)" if total else "")
        print()

        # Distribution par tranche
        rows = conn.execute("""
            SELECT
                CASE
                    WHEN sentiment < -0.5 THEN 'tres negatif (< -0.5)'
                    WHEN sentiment < 0.0 THEN 'negatif (-0.5 a 0.0)'
                    WHEN sentiment = 0.0 THEN 'neutre (0.0)'
                    WHEN sentiment <= 0.5 THEN 'positif (0.0 a 0.5)'
                    ELSE 'tres positif (> 0.5)'
                END as tranche,
                COUNT(*) as nb
            FROM news
            WHERE sentiment IS NOT NULL
            GROUP BY tranche
            ORDER BY tranche
        """).fetchall()
        print("Distribution du sentiment:")
        for r in rows:
            print(f"  {r['tranche']:30} {r['nb']:5}")

        conn.close()
        return

    if args.dry_run:
        unscored = db.get_news_without_sentiment()
        print(f"News sans sentiment: {len(unscored)}")
        # Estimation cout
        estimated_tokens = len(unscored) * 200  # ~200 tokens par appel
        estimated_cost = estimated_tokens / 1_000_000 * 0.15  # $0.15/1M input tokens
        print(f"Cout estime: ~${estimated_cost:.3f}")
        return

    scorer = LLMSentimentScorer(db)
    result = scorer.score_all_unscored()

    print()
    print("=" * 40)
    print("  RESULTAT")
    print("=" * 40)
    print(f"  Total traitees: {result['total']}")
    print(f"  Scorees OK: {result['scored']}")
    print(f"  Erreurs: {result['errors']}")


if __name__ == "__main__":
    main()
