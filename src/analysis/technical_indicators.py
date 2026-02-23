"""Calcul des indicateurs techniques pour le style range trading de Nicolas.

Inclut les indicateurs standards (RSI, MACD, Bollinger, ATR) et les features
specifiques au range trading (range_position, range_amplitude).
"""

import pandas as pd
import numpy as np
import ta

from loguru import logger


class TechnicalIndicators:
    """Calcule les indicateurs techniques pour un ticker."""

    def compute_all(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute toutes les colonnes d'indicateurs au DataFrame de prix.

        Le DataFrame doit contenir: date, open, high, low, close, volume.
        Retourne une copie enrichie (ne modifie pas l'original).
        """
        df = prices_df.copy()

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"].astype(float)

        # --- Indicateurs standards via librairie ta ---

        # RSI(14)
        df["rsi_14"] = ta.momentum.RSIIndicator(
            close=close, window=14
        ).rsi()

        # MACD(12, 26, 9)
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd_histogram"] = macd.macd_diff()

        # Bollinger Bands(20, 2)
        bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband()
        bb_lower = bb.bollinger_lband()
        bb_range = bb_upper - bb_lower
        df["bollinger_position"] = np.where(
            bb_range > 0,
            (close - bb_lower) / bb_range,
            0.5,
        )

        # ATR(14)
        atr = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close, window=14
        ).average_true_range()
        df["atr_14_pct"] = (atr / close) * 100

        # SMA(20), SMA(50)
        sma_20 = ta.trend.SMAIndicator(close=close, window=20).sma_indicator()
        sma_50 = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()

        df["distance_sma20"] = ((close - sma_20) / sma_20) * 100
        df["distance_sma50"] = ((close - sma_50) / sma_50) * 100

        # --- Features range trading (specifiques au style Nicolas) ---

        # Range 10 jours
        range_high_10 = high.rolling(window=10, min_periods=10).max()
        range_low_10 = low.rolling(window=10, min_periods=10).min()
        range_span_10 = range_high_10 - range_low_10
        df["range_position_10"] = np.where(
            range_span_10 > 0,
            (close - range_low_10) / range_span_10,
            0.5,
        )
        df["range_amplitude_10"] = np.where(
            range_low_10 > 0,
            (range_span_10 / range_low_10) * 100,
            0.0,
        )

        # Range 20 jours
        range_high_20 = high.rolling(window=20, min_periods=20).max()
        range_low_20 = low.rolling(window=20, min_periods=20).min()
        range_span_20 = range_high_20 - range_low_20
        df["range_position_20"] = np.where(
            range_span_20 > 0,
            (close - range_low_20) / range_span_20,
            0.5,
        )
        df["range_amplitude_20"] = np.where(
            range_low_20 > 0,
            (range_span_20 / range_low_20) * 100,
            0.0,
        )

        # --- Features derivees ---

        # Volume ratio (volume / moyenne 20j)
        vol_ma_20 = volume.rolling(window=20, min_periods=1).mean()
        df["volume_ratio_20"] = np.where(
            vol_ma_20 > 0,
            volume / vol_ma_20,
            1.0,
        )

        # Variations
        df["variation_1j"] = close.pct_change(periods=1) * 100
        df["variation_5j"] = close.pct_change(periods=5) * 100

        return df

    def get_indicators_at_date(self, enriched_df: pd.DataFrame, date: str) -> dict | None:
        """Retourne les indicateurs techniques pour une date specifique.

        Args:
            enriched_df: DataFrame retourne par compute_all().
            date: Date au format 'YYYY-MM-DD'.

        Returns:
            Dict avec les 13 features techniques, ou None si date introuvable.
        """
        mask = enriched_df["date"] == date
        if not mask.any():
            return None

        row = enriched_df[mask].iloc[0]

        feature_cols = [
            "rsi_14", "macd_histogram", "bollinger_position",
            "range_position_10", "range_position_20",
            "range_amplitude_10", "range_amplitude_20",
            "volume_ratio_20", "atr_14_pct",
            "variation_1j", "variation_5j",
            "distance_sma20", "distance_sma50",
        ]

        result = {}
        for col in feature_cols:
            val = row[col]
            # Convertir numpy types en float Python
            result[col] = float(val) if pd.notna(val) else None

        return result
