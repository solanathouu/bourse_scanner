"""Import batch des PDF d'avis d'exécution SG dans la base SQLite."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.extraction.pdf_parser import SGPDFParser
from src.extraction.trade_matcher import TradeMatcher
from loguru import logger


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(base_dir, "data", "pdfs")
    db_path = os.path.join(base_dir, "data", "trades.db")

    if not os.path.exists(pdf_dir):
        logger.error(f"Dossier PDF introuvable: {pdf_dir}")
        logger.info("Placez vos PDF dans data/pdfs/ et relancez.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        logger.error("Aucun PDF trouvé dans data/pdfs/")
        sys.exit(1)

    # 1. Init DB
    db = Database(db_path)
    db.init_db()

    # 2. Parser tous les PDF
    logger.info(f"=== PARSING DES PDF ===")
    parser = SGPDFParser()
    executions = parser.parse_directory(pdf_dir)

    if not executions:
        logger.warning("Aucune exécution extraite. Vérifiez vos PDF.")
        sys.exit(1)

    # 3. Insérer les exécutions en base
    db.insert_executions_batch(executions)
    logger.info(f"{len(executions)} exécutions insérées en base")

    # 4. Reconstruire les trades complets
    logger.info(f"=== RECONSTRUCTION DES TRADES ===")
    matcher = TradeMatcher()
    trades = matcher.match_trades(executions)

    # 5. Insérer les trades
    db.insert_trades_batch(trades)
    logger.info(f"{len(trades)} trades reconstitués et insérés en base")

    # 6. Résumé
    closed_trades = [t for t in trades if t["statut"] == "CLOTURE"]
    open_trades = [t for t in trades if t["statut"] == "OUVERT"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("RÉSUMÉ DE L'IMPORT")
    logger.info("=" * 60)
    logger.info(f"PDF parsés:            {len(executions)}")
    logger.info(f"  - Achats:            {sum(1 for e in executions if e['sens'] == 'ACHAT')}")
    logger.info(f"  - Ventes:            {sum(1 for e in executions if e['sens'] == 'VENTE')}")
    logger.info(f"Trades reconstitués:   {len(trades)}")
    logger.info(f"  - Clôturés:          {len(closed_trades)}")
    logger.info(f"  - Ouverts:           {len(open_trades)}")

    # Stats des ISIN
    isins = set(e.get("isin") for e in executions if e.get("isin"))
    logger.info(f"Actions différentes:   {len(isins)}")

    if closed_trades:
        avg_return = sum(t["rendement_brut_pct"] for t in closed_trades) / len(closed_trades)
        avg_duration = sum(t["duree_jours"] for t in closed_trades) / len(closed_trades)
        winners = sum(1 for t in closed_trades if t["rendement_brut_pct"] > 0)
        losers = len(closed_trades) - winners
        best = max(closed_trades, key=lambda t: t["rendement_brut_pct"])
        worst = min(closed_trades, key=lambda t: t["rendement_brut_pct"])

        logger.info("")
        logger.info("PERFORMANCE DES TRADES CLÔTURÉS:")
        logger.info(f"  Rendement moyen:     {avg_return:+.2f}%")
        logger.info(f"  Durée moyenne:       {avg_duration:.1f} jours")
        logger.info(f"  Gagnants:            {winners}/{len(closed_trades)} ({winners/len(closed_trades)*100:.0f}%)")
        logger.info(f"  Perdants:            {losers}/{len(closed_trades)} ({losers/len(closed_trades)*100:.0f}%)")
        logger.info(f"  Meilleur trade:      {best['nom_action']} {best['rendement_brut_pct']:+.2f}% ({best['duree_jours']:.0f}j)")
        logger.info(f"  Pire trade:          {worst['nom_action']} {worst['rendement_brut_pct']:+.2f}% ({worst['duree_jours']:.0f}j)")

    if open_trades:
        logger.info("")
        logger.info("POSITIONS OUVERTES:")
        for t in open_trades:
            logger.info(f"  {t['nom_action']:25s} {t['quantite']:>5}x @ {t['prix_achat']:.2f} (depuis {t['date_achat'][:10]})")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
