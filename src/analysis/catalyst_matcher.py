"""Correlation entre trades et catalyseurs (news autour de la date d'achat)."""

from datetime import datetime, timedelta

from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError


# Ponderations temporelles: distance_jours -> score de base
TEMPORAL_WEIGHTS = {
    0: 1.0,   # Jour d'achat
    -1: 0.8,  # Veille
    -2: 0.6,
    -3: 0.4,
    1: 0.7,   # Lendemain
}

TEXT_MATCH_BONUS = 0.2


class CatalystMatcher:
    """Associe les trades a leurs catalyseurs (news dans la fenetre temporelle)."""

    def __init__(self, db: Database, days_before: int = 3, days_after: int = 1):
        self.db = db
        self.days_before = days_before
        self.days_after = days_after
        self.ticker_mapper = TickerMapper()

    def match_trade(self, trade: dict) -> list[dict]:
        """Trouve les news catalyseurs pour un trade donne.

        Retourne une liste de dicts prets pour insert dans trade_catalyseurs.
        Retourne [] si ticker inconnu ou aucune news trouvee.
        """
        nom_action = trade["nom_action"]
        try:
            ticker = self.ticker_mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            logger.warning(f"Ticker inconnu pour '{nom_action}', skip")
            return []

        date_achat = datetime.strptime(trade["date_achat"][:10], "%Y-%m-%d")
        date_start = (date_achat - timedelta(days=self.days_before)).strftime("%Y-%m-%d")
        date_end = (date_achat + timedelta(days=self.days_after)).strftime("%Y-%m-%d")

        news_list = self.db.get_news_in_window(ticker, date_start, date_end)

        catalyseurs = []
        for news in news_list:
            news_date = datetime.strptime(news["published_at"][:10], "%Y-%m-%d")
            distance = (news_date - date_achat).days
            match_texte = self._check_text_match(
                nom_action, news["title"], news.get("description")
            )
            score = self._compute_score(distance, match_texte)
            if score > 0:
                catalyseurs.append({
                    "trade_id": trade["id"],
                    "news_id": news["id"],
                    "score_pertinence": score,
                    "distance_jours": distance,
                    "match_texte": 1 if match_texte else 0,
                })

        return catalyseurs

    def match_all_trades(self) -> dict:
        """Matche tous les trades. Vide la table avant, puis re-peuple.

        Retourne un resume: {total_trades, trades_avec_catalyseurs,
        total_associations, erreurs}.
        """
        self.db.clear_catalyseurs()
        trades = self.db.get_all_trades()
        total_associations = 0
        trades_avec = 0
        erreurs = 0

        for trade in trades:
            try:
                catalyseurs = self.match_trade(trade)
                if catalyseurs:
                    self.db.insert_catalyseurs_batch(catalyseurs)
                    trades_avec += 1
                    total_associations += len(catalyseurs)
            except Exception as e:
                logger.error(f"Erreur matching trade {trade['id']} "
                             f"({trade['nom_action']}): {e}")
                erreurs += 1

        result = {
            "total_trades": len(trades),
            "trades_avec_catalyseurs": trades_avec,
            "total_associations": total_associations,
            "erreurs": erreurs,
        }
        logger.info(f"Matching termine: {result}")
        return result

    def get_stats(self) -> dict:
        """Statistiques globales sur les catalyseurs."""
        total_catalyseurs = self.db.count_catalyseurs()
        total_trades = self.db.count_trades()
        return {
            "total_catalyseurs": total_catalyseurs,
            "total_trades": total_trades,
        }

    def _compute_score(self, distance_jours: int, match_texte: bool) -> float:
        """Calcule le score de pertinence.

        Score = ponderation temporelle + bonus match texte (cap a 1.0).
        Retourne 0.0 si la distance est hors fenetre.
        """
        base = TEMPORAL_WEIGHTS.get(distance_jours, 0.0)
        if base == 0.0:
            return 0.0
        bonus = TEXT_MATCH_BONUS if match_texte else 0.0
        return min(base + bonus, 1.0)

    def _check_text_match(self, nom_action: str, title: str, description: str | None) -> bool:
        """Verifie si le nom de l'action apparait dans le titre ou la description."""
        clean_name = nom_action.lstrip("* ").strip().lower()
        title_lower = title.lower() if title else ""
        desc_lower = description.lower() if description else ""
        return clean_name in title_lower or clean_name in desc_lower
