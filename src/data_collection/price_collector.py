"""Collecte des prix OHLCV historiques via yfinance."""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError

DAYS_BEFORE_TRADE = 30


class PriceCollector:
    """Collecte les prix historiques pour les actions tradees."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()

    def collect_for_ticker(self, ticker: str, start: str, end: str) -> int:
        """Telecharge les prix OHLCV pour un ticker et une periode.

        Returns:
            Nombre de jours de prix inseres.
        """
        logger.info(f"Collecte prix {ticker} du {start} au {end}")
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            logger.warning(f"Aucun prix retourne pour {ticker}")
            return 0

        # yfinance 1.x retourne un MultiIndex (Price, Ticker) pour un seul ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        prices = []
        for date, row in df.iterrows():
            prices.append({
                "ticker": ticker,
                "date": date.strftime("%Y-%m-%d"),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
            })

        self.db.insert_prices_batch(prices)
        logger.info(f"{len(prices)} prix inseres pour {ticker}")
        return len(prices)

    def compute_date_ranges(self, trades: list[dict]) -> dict:
        """Calcule les plages de dates par action a partir des trades.

        Pour chaque action: start = min(date_achat) - 30 jours, end = max(date_vente).
        Si trade ouvert (date_vente=None), end = aujourd'hui.

        Returns:
            Dict {nom_action: {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}}
        """
        ranges = {}
        today = datetime.now().strftime("%Y-%m-%d")

        for trade in trades:
            name = trade["nom_action"].lstrip("* ").strip()
            date_achat = trade["date_achat"][:10]
            date_vente = trade["date_vente"][:10] if trade["date_vente"] else today

            if name not in ranges:
                ranges[name] = {"start": date_achat, "end": date_vente}
            else:
                if date_achat < ranges[name]["start"]:
                    ranges[name]["start"] = date_achat
                if date_vente > ranges[name]["end"]:
                    ranges[name]["end"] = date_vente

        # Reculer le start de DAYS_BEFORE_TRADE jours
        for name in ranges:
            start_dt = datetime.strptime(ranges[name]["start"], "%Y-%m-%d")
            ranges[name]["start"] = (
                start_dt - timedelta(days=DAYS_BEFORE_TRADE)
            ).strftime("%Y-%m-%d")

        return ranges

    def collect_all(self) -> dict:
        """Collecte les prix pour toutes les actions tradees.

        Returns:
            Dict {"total_prices": int, "errors": list}
        """
        trades = self.db.get_all_trades()
        ranges = self.compute_date_ranges(trades)

        total = 0
        errors = []

        for nom_action, period in ranges.items():
            try:
                ticker = self.mapper.get_ticker(nom_action)
                count = self.collect_for_ticker(
                    ticker, period["start"], period["end"]
                )
                total += count
            except TickerNotFoundError as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.warning(f"Ticker inconnu: {nom_action}")
            except Exception as e:
                errors.append({"action": nom_action, "error": str(e)})
                logger.error(f"Erreur collecte prix {nom_action}: {e}")

        logger.info(f"Collecte prix terminee: {total} prix, {len(errors)} erreurs")
        return {"total_prices": total, "errors": errors}
