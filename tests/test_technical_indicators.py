"""Tests pour le calcul des indicateurs techniques."""

import pandas as pd
import numpy as np
import pytest

from src.analysis.technical_indicators import TechnicalIndicators


def _make_price_df(n_days: int = 60) -> pd.DataFrame:
    """Cree un DataFrame de prix synthetiques simulant un range trading.

    Simule une action oscillant entre 10 et 12 (range de 20%).
    """
    np.random.seed(42)
    dates = pd.bdate_range("2025-06-01", periods=n_days)
    # Oscillation sinusoidale + bruit
    t = np.linspace(0, 4 * np.pi, n_days)
    base = 11 + np.sin(t)  # Oscille entre 10 et 12
    noise = np.random.normal(0, 0.1, n_days)
    close = base + noise

    return pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "open": close - np.random.uniform(0, 0.3, n_days),
        "high": close + np.random.uniform(0, 0.5, n_days),
        "low": close - np.random.uniform(0, 0.5, n_days),
        "close": close,
        "volume": np.random.randint(50000, 200000, n_days),
    })


class TestComputeAll:
    """Tests du calcul complet des indicateurs."""

    def setup_method(self):
        self.tech = TechnicalIndicators()
        self.df = _make_price_df(60)

    def test_compute_all_returns_dataframe(self):
        """compute_all retourne un DataFrame."""
        result = self.tech.compute_all(self.df)
        assert isinstance(result, pd.DataFrame)

    def test_compute_all_has_rsi(self):
        """Le resultat contient RSI(14)."""
        result = self.tech.compute_all(self.df)
        assert "rsi_14" in result.columns
        # RSI doit etre entre 0 et 100
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_compute_all_has_macd(self):
        """Le resultat contient MACD histogram."""
        result = self.tech.compute_all(self.df)
        assert "macd_histogram" in result.columns

    def test_compute_all_has_bollinger(self):
        """Le resultat contient bollinger_position."""
        result = self.tech.compute_all(self.df)
        assert "bollinger_position" in result.columns

    def test_compute_all_has_range_position(self):
        """Le resultat contient range_position_10 et range_position_20."""
        result = self.tech.compute_all(self.df)
        assert "range_position_10" in result.columns
        assert "range_position_20" in result.columns

    def test_range_position_between_0_and_1(self):
        """range_position est entre 0 (support) et 1 (resistance)."""
        result = self.tech.compute_all(self.df)
        valid = result["range_position_10"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_compute_all_has_range_amplitude(self):
        """Le resultat contient range_amplitude_10 et range_amplitude_20."""
        result = self.tech.compute_all(self.df)
        assert "range_amplitude_10" in result.columns
        assert "range_amplitude_20" in result.columns

    def test_range_amplitude_positive(self):
        """L'amplitude du range est positive."""
        result = self.tech.compute_all(self.df)
        valid = result["range_amplitude_10"].dropna()
        assert (valid >= 0).all()

    def test_compute_all_has_volume_ratio(self):
        """Le resultat contient volume_ratio_20."""
        result = self.tech.compute_all(self.df)
        assert "volume_ratio_20" in result.columns

    def test_compute_all_has_variations(self):
        """Le resultat contient variation_1j et variation_5j."""
        result = self.tech.compute_all(self.df)
        assert "variation_1j" in result.columns
        assert "variation_5j" in result.columns

    def test_compute_all_has_sma_distances(self):
        """Le resultat contient distance_sma20 et distance_sma50."""
        result = self.tech.compute_all(self.df)
        assert "distance_sma20" in result.columns
        assert "distance_sma50" in result.columns

    def test_compute_all_has_atr_pct(self):
        """Le resultat contient atr_14_pct."""
        result = self.tech.compute_all(self.df)
        assert "atr_14_pct" in result.columns

    def test_compute_all_preserves_original_columns(self):
        """Les colonnes originales sont preservees."""
        result = self.tech.compute_all(self.df)
        assert "close" in result.columns
        assert "date" in result.columns
        assert len(result) == len(self.df)


class TestGetIndicatorsAtDate:
    """Tests de recuperation des indicateurs a une date precise."""

    def setup_method(self):
        self.tech = TechnicalIndicators()
        self.df = _make_price_df(60)

    def test_get_at_date_returns_dict(self):
        """get_indicators_at_date retourne un dict."""
        enriched = self.tech.compute_all(self.df)
        # Prendre une date au milieu (apres warmup des indicateurs)
        target_date = self.df.iloc[50]["date"]
        result = self.tech.get_indicators_at_date(enriched, target_date)
        assert isinstance(result, dict)

    def test_get_at_date_has_all_features(self):
        """Le dict retourne contient toutes les features techniques."""
        enriched = self.tech.compute_all(self.df)
        target_date = self.df.iloc[50]["date"]
        result = self.tech.get_indicators_at_date(enriched, target_date)
        expected_keys = [
            "rsi_14", "macd_histogram", "bollinger_position",
            "range_position_10", "range_position_20",
            "range_amplitude_10", "range_amplitude_20",
            "volume_ratio_20", "atr_14_pct",
            "variation_1j", "variation_5j",
            "distance_sma20", "distance_sma50",
        ]
        for key in expected_keys:
            assert key in result, f"Cle manquante: {key}"

    def test_get_at_date_unknown_returns_none(self):
        """Date inconnue retourne None."""
        enriched = self.tech.compute_all(self.df)
        result = self.tech.get_indicators_at_date(enriched, "1999-01-01")
        assert result is None

    def test_short_dataframe_returns_nans(self):
        """Un DataFrame trop court (<50 jours) a des NaN mais ne crashe pas."""
        short_df = _make_price_df(15)
        enriched = self.tech.compute_all(short_df)
        # distance_sma50 sera NaN car pas assez de donnees
        assert enriched["distance_sma50"].isna().any()
