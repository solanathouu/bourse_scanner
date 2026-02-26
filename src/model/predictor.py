"""Scoring temps reel des tickers de la watchlist.

Charge le modele XGBoost entraine et calcule un score
"Nicolas prendrait ce trade?" pour chaque ticker.
"""

import json

import pandas as pd
from loguru import logger

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine
from src.model.trainer import Trainer, CATALYST_TYPE_ENCODING


class Predictor:
    """Score les tickers en temps reel avec le modele de Nicolas."""

    def __init__(self, db: Database,
                 model_path: str = "data/models/nicolas_v1.joblib"):
        self.db = db
        self.engine = FeatureEngine(db)
        self.trainer = Trainer()
        self.trainer.load_model(model_path)

    def score_ticker(self, ticker: str, current_price: float,
                     date: str | None = None) -> dict | None:
        """Score un ticker et retourne un signal dict.

        Args:
            ticker: Ticker Yahoo (ex: SAN.PA).
            current_price: Prix actuel.
            date: Date YYYY-MM-DD (default: aujourd'hui).

        Returns:
            Dict signal ou None si pas assez de donnees.
        """
        from datetime import datetime

        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        features = self.engine.build_realtime_features(
            ticker, current_price, date
        )
        if features is None:
            return None

        # Construire le DataFrame pour la prediction
        X = self._build_feature_dataframe(features)
        if X is None:
            return None

        # Predire
        score = float(self.trainer.predict_proba(X)[0])

        # Construire le signal
        return {
            "ticker": ticker,
            "date": date,
            "score": round(score, 4),
            "catalyst_type": features.get("catalyst_type", "UNKNOWN"),
            "catalyst_news_title": None,
            "features": features,
            "features_json": json.dumps(
                {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in features.items()
                 if k != "catalyst_type"}
            ),
            "technical_summary": self._build_technical_summary(features),
        }

    def score_watchlist(self, watchlist: list[dict],
                        date: str | None = None) -> list[dict]:
        """Score tous les tickers non-ETF de la watchlist.

        Args:
            watchlist: Liste de dicts avec 'ticker', 'name', 'etf'.
            date: Date YYYY-MM-DD (default: aujourd'hui).

        Returns:
            Liste de signals tries par score decroissant.
        """
        signals = []

        for item in watchlist:
            if item.get("etf", False):
                logger.debug(f"Skip ETF: {item['name']}")
                continue

            ticker = item["ticker"]
            price = self._get_current_price(ticker)
            if price is None:
                logger.warning(f"Pas de prix pour {ticker}, skip")
                continue

            signal = self.score_ticker(ticker, price, date=date)
            if signal:
                signal["name"] = item["name"]
                signals.append(signal)

        signals.sort(key=lambda s: s["score"], reverse=True)
        logger.info(
            f"Watchlist scoree: {len(signals)} tickers, "
            f"top={signals[0]['score']:.2f} ({signals[0]['ticker']})"
            if signals else "Watchlist scoree: 0 tickers"
        )
        return signals

    def _get_current_price(self, ticker: str) -> float | None:
        """Recupere le dernier prix connu pour un ticker."""
        prices = self.db.get_prices(ticker)
        if not prices:
            return None
        return prices[-1]["close"]

    def _build_feature_dataframe(self, features: dict) -> pd.DataFrame | None:
        """Construit un DataFrame aligne sur les colonnes du modele."""
        # Encoder catalyst_type
        cat_type = features.get("catalyst_type", "UNKNOWN")
        encoded = CATALYST_TYPE_ENCODING.get(cat_type, 1)

        row = {**features, "catalyst_type": encoded}

        # Construire le DataFrame avec les colonnes du modele
        model_cols = self.trainer.feature_names
        df_row = {}
        for col in model_cols:
            val = row.get(col, 0)
            df_row[col] = val if val is not None else 0

        return pd.DataFrame([df_row])

    def _build_technical_summary(self, features: dict) -> str:
        """Construit un resume textuel des indicateurs techniques."""
        parts = []

        rsi = features.get("rsi_14")
        if rsi is not None:
            label = "survendu" if rsi < 30 else "suracheté" if rsi > 70 else "neutre"
            parts.append(f"RSI {rsi:.1f} ({label})")

        rp = features.get("range_position_20")
        if rp is not None:
            label = "bas" if rp < 0.3 else "haut" if rp > 0.7 else "milieu"
            parts.append(f"Range {rp:.2f} ({label})")

        return " | ".join(parts) if parts else "N/A"
