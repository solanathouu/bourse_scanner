"""Assemblage du vecteur de features pour chaque trade de Nicolas.

Combine les indicateurs techniques (au moment de l'achat), le type de
catalyseur, et le contexte personnel (historique de trading sur l'action).
"""

import pandas as pd
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError
from src.analysis.technical_indicators import TechnicalIndicators


# Colonnes de features techniques retournees par TechnicalIndicators
TECHNICAL_FEATURES = [
    "rsi_14", "macd_histogram", "bollinger_position",
    "range_position_10", "range_position_20",
    "range_amplitude_10", "range_amplitude_20",
    "volume_ratio_20", "atr_14_pct",
    "variation_1j", "variation_5j",
    "distance_sma20", "distance_sma50",
]

# Colonnes de features catalyseur (LLM-based)
CATALYST_FEATURES = [
    "catalyst_type", "catalyst_confidence", "news_sentiment",
    "has_clear_catalyst", "buy_reason_length",
]

# Colonnes de features contexte
CONTEXT_FEATURES = [
    "day_of_week", "nb_previous_trades",
    "previous_win_rate", "days_since_last_trade",
]


class FeatureEngine:
    """Assemble le vecteur de features pour chaque trade de Nicolas."""

    def __init__(self, db: Database):
        self.db = db
        self.tech = TechnicalIndicators()
        self.mapper = TickerMapper()
        # Cache des DataFrames de prix enrichis par ticker
        self._price_cache: dict[str, pd.DataFrame] = {}

    def _get_enriched_prices(self, ticker: str) -> pd.DataFrame | None:
        """Recupere et enrichit les prix pour un ticker (avec cache)."""
        if ticker in self._price_cache:
            return self._price_cache[ticker]

        prices = self.db.get_prices(ticker)
        if len(prices) < 20:
            logger.warning(f"Pas assez de prix pour {ticker}: {len(prices)} jours")
            return None

        df = pd.DataFrame(prices)
        enriched = self.tech.compute_all(df)
        self._price_cache[ticker] = enriched
        return enriched

    def _build_technical_features(self, ticker: str, date_achat: str) -> dict | None:
        """Construit les features techniques pour un trade a la date d'achat."""
        enriched = self._get_enriched_prices(ticker)
        if enriched is None:
            return None
        return self.tech.get_indicators_at_date(enriched, date_achat)

    def _build_catalyst_features(self, trade: dict) -> dict:
        """Construit les features catalyseur pour un trade.

        Utilise l'analyse LLM si disponible, sinon fallback TECHNICAL.
        """
        trade_id = trade["id"]
        analysis = self.db.get_trade_analysis(trade_id)

        if analysis:
            return {
                "catalyst_type": analysis["catalyst_type"],
                "catalyst_confidence": analysis["catalyst_confidence"],
                "news_sentiment": analysis["news_sentiment"] or 0.0,
                "has_clear_catalyst": 1 if analysis["primary_news_id"] else 0,
                "buy_reason_length": len(analysis["buy_reason"] or ""),
            }

        # Fallback: pas d'analyse LLM
        return {
            "catalyst_type": "TECHNICAL",
            "catalyst_confidence": 0.0,
            "news_sentiment": 0.0,
            "has_clear_catalyst": 0,
            "buy_reason_length": 0,
        }

    def _build_context_features(self, trade: dict, all_trades: list[dict]) -> dict:
        """Construit les features de contexte personnel.

        Calcule l'historique de Nicolas sur cette action AVANT ce trade.
        """
        from datetime import datetime

        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]

        # Trades precedents sur la meme action, avant ce trade
        previous = [
            t for t in all_trades
            if t["nom_action"] == nom_action
            and t["date_achat"][:10] < date_achat
            and t["statut"] == "CLOTURE"
        ]

        nb_previous = len(previous)
        if nb_previous > 0:
            wins = sum(1 for t in previous if t["rendement_brut_pct"] > 0)
            win_rate = wins / nb_previous
            last_trade_date = max(t["date_vente"][:10] for t in previous if t["date_vente"])
            last_dt = datetime.strptime(last_trade_date, "%Y-%m-%d")
            current_dt = datetime.strptime(date_achat, "%Y-%m-%d")
            days_since = (current_dt - last_dt).days
        else:
            win_rate = 0.0
            days_since = -1  # Pas de trade precedent

        # Jour de la semaine (0=lundi, 4=vendredi)
        day_of_week = datetime.strptime(date_achat, "%Y-%m-%d").weekday()

        return {
            "day_of_week": day_of_week,
            "nb_previous_trades": nb_previous,
            "previous_win_rate": round(win_rate, 4),
            "days_since_last_trade": days_since,
        }

    def build_trade_features(self, trade: dict, all_trades: list[dict] | None = None) -> dict | None:
        """Construit le vecteur de features complet pour UN trade.

        Args:
            trade: Dict du trade (de db.get_all_trades()).
            all_trades: Liste de tous les trades pour le contexte.
                        Si None, les recupere depuis la base.

        Returns:
            Dict avec ~25 features + target, ou None si pas assez de donnees.
        """
        if all_trades is None:
            all_trades = self.db.get_all_trades()

        nom_action = trade["nom_action"]
        date_achat = trade["date_achat"][:10]

        # Ticker Yahoo
        try:
            ticker = self.mapper.get_ticker(nom_action)
        except TickerNotFoundError:
            logger.warning(f"Ticker inconnu pour '{nom_action}', skip")
            return None

        # Features techniques
        tech_features = self._build_technical_features(ticker, date_achat)
        if tech_features is None:
            logger.warning(f"Pas de donnees techniques pour {nom_action} au {date_achat}")
            return None

        # Features catalyseur
        cat_features = self._build_catalyst_features(trade)

        # Features contexte
        ctx_features = self._build_context_features(trade, all_trades)

        # Target
        is_winner = 1 if trade["rendement_brut_pct"] > 0 else 0

        # Assembler
        features = {
            "trade_id": trade["id"],
            **tech_features,
            **cat_features,
            **ctx_features,
            "is_winner": is_winner,
        }

        return features

    def build_all_features(self) -> pd.DataFrame:
        """Construit la matrice de features pour tous les trades clotures.

        Retourne un DataFrame avec ~25 colonnes de features + target + trade_id.
        Les trades sans donnees suffisantes sont exclus (avec warning).
        """
        trades = self.db.get_all_trades()
        closed_trades = [t for t in trades if t["statut"] == "CLOTURE"]

        logger.info(f"Construction features pour {len(closed_trades)} trades clotures")

        rows = []
        skipped = 0

        for trade in closed_trades:
            features = self.build_trade_features(trade, all_trades=trades)
            if features is not None:
                rows.append(features)
            else:
                skipped += 1

        if skipped > 0:
            logger.warning(f"{skipped} trades exclus (donnees insuffisantes)")

        df = pd.DataFrame(rows)
        logger.info(f"Matrice de features: {len(df)} lignes x {len(df.columns)} colonnes")
        return df

    def get_feature_names(self) -> list[str]:
        """Liste ordonnee des noms de features (sans target ni trade_id)."""
        return TECHNICAL_FEATURES + CATALYST_FEATURES + CONTEXT_FEATURES
