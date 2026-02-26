"""Assemblage du vecteur de features pour chaque trade de Nicolas.

Combine les indicateurs techniques (au moment de l'achat), le type de
catalyseur, les donnees fondamentales, et le contexte personnel.
"""

from datetime import datetime

import pandas as pd
from loguru import logger

from src.core.database import Database
from src.data_collection.ticker_mapper import TickerMapper, TickerNotFoundError
from src.analysis.technical_indicators import TechnicalIndicators
from src.analysis.news_classifier import NewsClassifier


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

# Colonnes de features fondamentales
FUNDAMENTAL_FEATURES = [
    "pe_ratio", "pb_ratio", "target_upside_pct",
    "analyst_count", "days_to_earnings", "recommendation_score",
]

# Colonnes de features contexte
CONTEXT_FEATURES = [
    "day_of_week", "nb_previous_trades",
    "previous_win_rate", "days_since_last_trade",
]

RECOMMENDATION_SCORES = {
    "strongBuy": 5, "buy": 4, "hold": 3,
    "underperform": 2, "sell": 2, "strongSell": 1,
}


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

    def _build_fundamental_features(self, ticker: str, date_achat: str,
                                       prix_achat: float) -> dict:
        """Construit les features fondamentales pour un trade.

        Utilise les donnees fondamentales les plus recentes avant la date d'achat.
        Retourne des valeurs par defaut si aucune donnee disponible.
        """
        default = {
            "pe_ratio": 0.0, "pb_ratio": 0.0,
            "target_upside_pct": 0.0, "analyst_count": 0,
            "days_to_earnings": -1, "recommendation_score": 0,
        }

        fund = self.db.get_fundamental_at_date(ticker, date_achat)
        if fund is None:
            return default

        pe = fund.get("pe_ratio") or 0.0
        pb = fund.get("pb_ratio") or 0.0

        # Target upside: (target - prix_achat) / prix_achat * 100
        target = fund.get("target_price")
        target_upside = 0.0
        if target and prix_achat > 0:
            target_upside = round((target - prix_achat) / prix_achat * 100, 2)

        analysts = fund.get("analyst_count") or 0

        # Days to earnings
        days_to_earn = -1
        earnings_date = fund.get("earnings_date")
        if earnings_date:
            try:
                earn_dt = datetime.strptime(earnings_date[:10], "%Y-%m-%d")
                achat_dt = datetime.strptime(date_achat[:10], "%Y-%m-%d")
                days_to_earn = (earn_dt - achat_dt).days
            except ValueError:
                pass

        # Recommendation score
        reco = fund.get("recommendation") or ""
        reco_score = RECOMMENDATION_SCORES.get(reco, 0)

        return {
            "pe_ratio": pe,
            "pb_ratio": pb,
            "target_upside_pct": target_upside,
            "analyst_count": analysts,
            "days_to_earnings": days_to_earn,
            "recommendation_score": reco_score,
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

        # Features fondamentales
        fund_features = self._build_fundamental_features(
            ticker, date_achat, trade["prix_achat"]
        )

        # Features contexte
        ctx_features = self._build_context_features(trade, all_trades)

        # Target
        is_winner = 1 if trade["rendement_brut_pct"] > 0 else 0

        # Assembler
        features = {
            "trade_id": trade["id"],
            **tech_features,
            **cat_features,
            **fund_features,
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

    def build_realtime_features(self, ticker: str, current_price: float,
                                   date: str | None = None) -> dict | None:
        """Construit le vecteur de features pour un ticker en temps reel.

        Utilise les donnees en BDD (prix, news recentes, fondamentaux)
        pour assembler les memes features que build_trade_features().

        Args:
            ticker: Ticker Yahoo Finance (ex: SAN.PA).
            current_price: Prix actuel du ticker.
            date: Date au format YYYY-MM-DD. Si None, utilise aujourd'hui.

        Returns:
            Dict avec les features, ou None si pas assez de donnees.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        # Features techniques — fallback sur la derniere date disponible
        tech_features = self._build_technical_features(ticker, date)
        if tech_features is None:
            enriched = self._get_enriched_prices(ticker)
            if enriched is not None and not enriched.empty:
                last_date = enriched["date"].max()
                tech_features = self.tech.get_indicators_at_date(
                    enriched, last_date
                )
                if tech_features is not None:
                    logger.debug(
                        f"Fallback date technique {ticker}: "
                        f"{date} -> {last_date}"
                    )
                    date = last_date
        if tech_features is None:
            logger.warning(f"Pas de donnees techniques pour {ticker}")
            return None

        # Features catalyseur (news recentes)
        cat_features = self._build_realtime_catalyst_features(ticker, date)

        # Features fondamentales
        fund_features = self._build_fundamental_features(
            ticker, date, current_price
        )

        # Features contexte
        ctx_features = self._build_realtime_context_features(ticker, date)

        return {
            **tech_features,
            **cat_features,
            **fund_features,
            **ctx_features,
        }

    def _build_realtime_catalyst_features(self, ticker: str, date: str) -> dict:
        """Construit les features catalyseur a partir des news recentes en BDD.

        Regarde les news des 5 derniers jours, classifie avec NewsClassifier.
        """
        from datetime import timedelta

        dt = datetime.strptime(date, "%Y-%m-%d")
        date_start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")

        news_list = self.db.get_news_in_window(ticker, date_start, date)

        if not news_list:
            return {
                "catalyst_type": "TECHNICAL",
                "catalyst_confidence": 0.0,
                "news_sentiment": 0.0,
                "has_clear_catalyst": 0,
                "buy_reason_length": 0,
            }

        classifier = NewsClassifier()
        classified = classifier.classify_batch(news_list)

        # Trouver la news avec le catalyseur le plus prioritaire
        best = max(classified, key=lambda n: n.get("sentiment") or 0.0)
        cat_type = best.get("catalyst_type", "UNKNOWN")

        # Sentiment moyen des news
        sentiments = [n.get("sentiment") or 0.0 for n in news_list]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        has_catalyst = 1 if cat_type not in ("TECHNICAL", "UNKNOWN") else 0

        return {
            "catalyst_type": cat_type,
            "catalyst_confidence": 0.5 if has_catalyst else 0.0,
            "news_sentiment": round(avg_sentiment, 4),
            "has_clear_catalyst": has_catalyst,
            "buy_reason_length": len(best.get("title", "")),
        }

    def _build_realtime_context_features(self, ticker: str, date: str) -> dict:
        """Construit les features contexte pour le scoring temps reel.

        Utilise l'historique des trades de Nicolas sur ce ticker.
        """
        all_trades = self.db.get_all_trades()
        action_name = self.mapper.get_action_name(ticker)

        previous = [
            t for t in all_trades
            if t["nom_action"] == action_name
            and t["date_achat"][:10] < date
            and t["statut"] == "CLOTURE"
        ] if action_name else []

        nb_previous = len(previous)
        if nb_previous > 0:
            wins = sum(1 for t in previous if t["rendement_brut_pct"] > 0)
            win_rate = wins / nb_previous
            last_trade_date = max(
                t["date_vente"][:10] for t in previous if t["date_vente"]
            )
            last_dt = datetime.strptime(last_trade_date, "%Y-%m-%d")
            current_dt = datetime.strptime(date, "%Y-%m-%d")
            days_since = (current_dt - last_dt).days
        else:
            win_rate = 0.0
            days_since = -1

        day_of_week = datetime.strptime(date, "%Y-%m-%d").weekday()

        return {
            "day_of_week": day_of_week,
            "nb_previous_trades": nb_previous,
            "previous_win_rate": round(win_rate, 4),
            "days_since_last_trade": days_since,
        }

    def get_feature_names(self) -> list[str]:
        """Liste ordonnee des noms de features (sans target ni trade_id)."""
        return (TECHNICAL_FEATURES + CATALYST_FEATURES
                + FUNDAMENTAL_FEATURES + CONTEXT_FEATURES)
