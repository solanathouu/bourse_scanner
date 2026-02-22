"""Collecte des news historiques via GNews (Google News RSS)."""

import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from gnews import GNews
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError

DAYS_BEFORE = 7
DAYS_AFTER = 3
DELAY_BETWEEN_REQUESTS = 3  # secondes


class NewsCollector:
    """Collecte les news historiques pour les actions tradees."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()

    def _parse_article(self, article: dict, ticker: str) -> dict:
        """Transforme un article GNews en dict pour la base."""
        published_at = ""
        raw_date = article.get("published date", "")
        if raw_date:
            try:
                dt = parsedate_to_datetime(raw_date)
                published_at = dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                published_at = raw_date

        publisher = article.get("publisher", {})
        source = publisher.get("title", "") if isinstance(publisher, dict) else ""

        return {
            "ticker": ticker,
            "title": article.get("title", ""),
            "source": source,
            "url": article.get("url", ""),
            "published_at": published_at,
            "description": article.get("description", ""),
        }

    def collect_for_action(
        self, nom_action: str, ticker: str, start: str, end: str
    ) -> int:
        """Collecte les news pour une action sur une periode.

        Args:
            nom_action: Nom de l'action pour la recherche (ex: "SANOFI")
            ticker: Ticker Yahoo pour le stockage (ex: "SAN.PA")
            start: Date debut YYYY-MM-DD
            end: Date fin YYYY-MM-DD

        Returns:
            Nombre de news inserees.
        """
        logger.info(f"Collecte news {nom_action} du {start} au {end}")

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")

        gn = GNews(
            language="fr",
            country="FR",
            start_date=(start_dt.year, start_dt.month, start_dt.day),
            end_date=(end_dt.year, end_dt.month, end_dt.day),
            max_results=100,
        )

        query = f"{nom_action} bourse action"
        articles = gn.get_news(query)

        if not articles:
            logger.warning(f"Aucune news pour {nom_action} ({start} -> {end})")
            return 0

        news_list = [self._parse_article(a, ticker) for a in articles]
        # Filtrer les articles sans URL (pas de deduplication possible)
        news_list = [n for n in news_list if n["url"]]

        self.db.insert_news_batch(news_list)
        logger.info(f"{len(news_list)} news inserees pour {nom_action}")
        return len(news_list)

    def compute_news_windows(self, trades: list[dict]) -> list[dict]:
        """Calcule les fenetres de recherche de news pour chaque trade.

        Pour chaque trade: start = date_achat - 7j, end = date_achat + 3j.
        Retourne une liste de fenetres (une par trade, pas par action).

        Returns:
            List[{"nom_action": str, "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}]
        """
        windows = []
        for trade in trades:
            date_achat = datetime.strptime(
                trade["date_achat"][:10], "%Y-%m-%d"
            )
            start = (date_achat - timedelta(days=DAYS_BEFORE)).strftime("%Y-%m-%d")
            end = (date_achat + timedelta(days=DAYS_AFTER)).strftime("%Y-%m-%d")
            name = trade["nom_action"].lstrip("* ").strip()

            windows.append({
                "nom_action": name,
                "start": start,
                "end": end,
            })

        return windows

    def _deduplicate_windows(self, windows: list[dict]) -> list[dict]:
        """Fusionne les fenetres qui se chevauchent pour la meme action."""
        by_action = {}
        for w in windows:
            name = w["nom_action"]
            if name not in by_action:
                by_action[name] = []
            by_action[name].append((w["start"], w["end"]))

        result = []
        for name, periods in by_action.items():
            periods.sort()
            merged = [periods[0]]
            for start, end in periods[1:]:
                prev_start, prev_end = merged[-1]
                if start <= prev_end:
                    merged[-1] = (prev_start, max(prev_end, end))
                else:
                    merged.append((start, end))
            for start, end in merged:
                result.append({"nom_action": name, "start": start, "end": end})

        return result

    def collect_all(self) -> dict:
        """Collecte les news pour toutes les actions tradees.

        Returns:
            Dict {"total_news": int, "errors": list}
        """
        trades = self.db.get_all_trades()
        windows = self.compute_news_windows(trades)
        windows = self._deduplicate_windows(windows)

        total = 0
        errors = []

        for window in windows:
            try:
                ticker = self.mapper.get_ticker(window["nom_action"])
                count = self.collect_for_action(
                    window["nom_action"], ticker,
                    window["start"], window["end"],
                )
                total += count
                time.sleep(DELAY_BETWEEN_REQUESTS)
            except TickerNotFoundError as e:
                errors.append({"action": window["nom_action"], "error": str(e)})
                logger.warning(f"Ticker inconnu: {window['nom_action']}")
            except Exception as e:
                errors.append({"action": window["nom_action"], "error": str(e)})
                logger.error(f"Erreur collecte news {window['nom_action']}: {e}")

        logger.info(f"Collecte news terminee: {total} news, {len(errors)} erreurs")
        return {"total_news": total, "errors": errors}
