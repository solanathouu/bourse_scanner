"""Collecte des donnees fondamentales via yfinance.

Recupere PER, PBR, consensus analyste, target price, prochaine date
d'earnings pour chaque action tradee par Nicolas.
"""

from datetime import datetime

import yfinance as yf
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper


RECOMMENDATION_MAP = {
    "strongBuy": 5, "buy": 4, "hold": 3,
    "underperform": 2, "sell": 2, "strongSell": 1,
}


class FundamentalCollector:
    """Collecte les donnees fondamentales pour chaque ticker."""

    def __init__(self, db: Database):
        self.db = db
        self.mapper = TickerMapper()

    def _extract_fundamentals(self, info: dict, ticker: str) -> dict:
        """Extrait les champs fondamentaux de yf.Ticker.get_info().

        Returns:
            Dict pret pour insert_fundamental.
        """
        today = datetime.now().strftime("%Y-%m-%d")

        # Dividend yield: yfinance retourne un ratio (0.035), on stocke en % (3.5)
        div_yield = info.get("dividendYield")
        if div_yield is not None:
            div_yield = round(div_yield * 100, 2)

        return {
            "ticker": ticker,
            "date": today,
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "market_cap": info.get("marketCap"),
            "dividend_yield": div_yield,
            "target_price": info.get("targetMeanPrice"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey"),
            "earnings_date": None,  # Traite separement
        }

    def _get_next_earnings(self, yf_ticker: yf.Ticker) -> str | None:
        """Recupere la prochaine date d'earnings."""
        try:
            cal = yf_ticker.get_calendar()
            if cal and "Earnings Date" in cal:
                dates = cal["Earnings Date"]
                if dates:
                    return str(dates[0])[:10]
        except Exception:
            pass
        return None

    def collect_for_ticker(self, ticker: str) -> dict | None:
        """Collecte les fondamentaux pour un ticker.

        Returns:
            Dict insere, ou None si echec.
        """
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.get_info()
        except Exception as e:
            logger.error(f"yfinance erreur pour {ticker}: {e}")
            return None

        if not info or info.get("regularMarketPrice") is None:
            logger.warning(f"Pas de donnees yfinance pour {ticker}")
            return None

        fundamental = self._extract_fundamentals(info, ticker)

        # Prochaine date d'earnings
        fundamental["earnings_date"] = self._get_next_earnings(yf_ticker)

        self.db.insert_fundamental(fundamental)
        logger.info(
            f"{ticker}: PE={fundamental['pe_ratio']}, "
            f"PB={fundamental['pb_ratio']}, "
            f"analysts={fundamental['analyst_count']}, "
            f"reco={fundamental['recommendation']}"
        )
        return fundamental

    def collect_all(self) -> dict:
        """Collecte les fondamentaux pour toutes les actions tradees.

        Returns:
            Dict {"total": int, "collected": int, "errors": int, "details": list}
        """
        mappings = self.mapper.get_all_mappings()
        tickers = list(set(mappings.values()))

        logger.info(f"Collecte fondamentaux pour {len(tickers)} tickers...")

        collected = 0
        errors = 0
        details = []

        for ticker in sorted(tickers):
            result = self.collect_for_ticker(ticker)
            if result:
                collected += 1
                details.append({"ticker": ticker, "pe": result["pe_ratio"],
                                "analysts": result["analyst_count"]})
            else:
                errors += 1

        summary = {
            "total": len(tickers),
            "collected": collected,
            "errors": errors,
            "details": details,
        }
        logger.info(f"Fondamentaux: {collected}/{len(tickers)} collectes, {errors} erreurs")
        return summary
