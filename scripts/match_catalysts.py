"""Script CLI pour matcher les trades avec leurs catalyseurs.

Usage:
    uv run python scripts/match_catalysts.py           # Matcher tous les trades
    uv run python scripts/match_catalysts.py --stats    # Stats seulement
"""

import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.analysis.catalyst_matcher import CatalystMatcher


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "trades.db")


def print_stats(db: Database, matcher: CatalystMatcher):
    """Affiche les statistiques detaillees des catalyseurs."""
    stats = matcher.get_stats()

    if stats["total_catalyseurs"] == 0:
        print("Aucun catalyseur en base. Lancez d'abord le matching sans --stats.")
        return

    # Stats globales
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row

    total_trades = stats["total_trades"]
    trades_avec = conn.execute(
        "SELECT COUNT(DISTINCT trade_id) FROM trade_catalyseurs"
    ).fetchone()[0]
    trades_sans = total_trades - trades_avec
    total_assoc = stats["total_catalyseurs"]
    score_moy = conn.execute(
        "SELECT AVG(score_pertinence) FROM trade_catalyseurs"
    ).fetchone()[0]

    print("=== CATALYST MATCHING ===")
    print(f"Trades analyses: {total_trades}")
    print(f"Trades avec catalyseurs: {trades_avec} ({100*trades_avec/total_trades:.1f}%)")
    print(f"Trades sans catalyseurs: {trades_sans} ({100*trades_sans/total_trades:.1f}%)")
    print(f"Total associations: {total_assoc}")
    print(f"Score moyen: {score_moy:.2f}")

    # Par source
    rows = conn.execute("""
        SELECT n.source_api, COUNT(*) as cnt
        FROM trade_catalyseurs tc
        JOIN news n ON tc.news_id = n.id
        GROUP BY n.source_api
        ORDER BY cnt DESC
    """).fetchall()
    print("\n=== PAR SOURCE ===")
    for r in rows:
        print(f"  {r['source_api']:30s}: {r['cnt']} associations")

    # Top 5 trades
    rows = conn.execute("""
        SELECT t.nom_action, t.date_achat, COUNT(*) as cnt,
               AVG(tc.score_pertinence) as score_moy
        FROM trade_catalyseurs tc
        JOIN trades_complets t ON tc.trade_id = t.id
        GROUP BY tc.trade_id
        ORDER BY cnt DESC
        LIMIT 5
    """).fetchall()
    print("\n=== TOP 5 TRADES (plus de catalyseurs) ===")
    for i, r in enumerate(rows, 1):
        print(f"  {i}. {r['nom_action']} (achat {r['date_achat'][:10]}): "
              f"{r['cnt']} catalyseurs, score moy {r['score_moy']:.2f}")

    # Gagnants vs perdants
    rows = conn.execute("""
        SELECT
            CASE WHEN t.rendement_brut_pct > 0 THEN 'gagnant' ELSE 'perdant' END as type,
            AVG(sub.cnt) as moy_catalyseurs,
            AVG(sub.score_moy) as moy_score
        FROM (
            SELECT tc.trade_id, COUNT(*) as cnt,
                   AVG(tc.score_pertinence) as score_moy
            FROM trade_catalyseurs tc
            GROUP BY tc.trade_id
        ) sub
        JOIN trades_complets t ON sub.trade_id = t.id
        WHERE t.statut = 'CLOTURE'
        GROUP BY type
    """).fetchall()
    print("\n=== GAGNANTS vs PERDANTS ===")
    for r in rows:
        print(f"  {r['type'].capitalize()}s: {r['moy_catalyseurs']:.1f} catalyseurs "
              f"en moyenne, score moy {r['moy_score']:.2f}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Matcher trades <-> catalyseurs")
    parser.add_argument("--stats", action="store_true", help="Afficher stats seulement")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()
    matcher = CatalystMatcher(db)

    if args.stats:
        print_stats(db, matcher)
        return

    print("Matching trades <-> catalyseurs...")
    result = matcher.match_all_trades()

    print(f"\n=== RESULTAT ===")
    print(f"Trades analyses: {result['total_trades']}")
    print(f"Trades avec catalyseurs: {result['trades_avec_catalyseurs']}")
    print(f"Total associations: {result['total_associations']}")
    print(f"Erreurs: {result['erreurs']}")

    print("\n--- Stats detaillees ---")
    print_stats(db, matcher)


if __name__ == "__main__":
    main()
