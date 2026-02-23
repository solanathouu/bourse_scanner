"""Tests pour l'entrainement du modele XGBoost."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.model.trainer import Trainer


def _make_feature_df(n_samples: int = 50) -> pd.DataFrame:
    """Cree un DataFrame de features synthetiques.

    Simule le style Nicolas: trades gagnants quand range_position < 0.3 + catalyseur.
    """
    np.random.seed(42)

    # Features techniques
    range_pos = np.random.uniform(0, 1, n_samples)
    rsi = np.random.uniform(20, 80, n_samples)
    macd_hist = np.random.normal(0, 0.5, n_samples)
    bollinger_pos = np.random.uniform(0, 1, n_samples)
    range_amp_10 = np.random.uniform(5, 25, n_samples)
    range_amp_20 = np.random.uniform(5, 30, n_samples)
    vol_ratio = np.random.uniform(0.5, 3, n_samples)
    atr_pct = np.random.uniform(1, 5, n_samples)
    var_1j = np.random.normal(0, 2, n_samples)
    var_5j = np.random.normal(0, 4, n_samples)
    dist_sma20 = np.random.normal(0, 3, n_samples)
    dist_sma50 = np.random.normal(0, 5, n_samples)

    # Features catalyseur
    cat_types = np.random.choice(
        ["EARNINGS", "FDA_REGULATORY", "UPGRADE", "CONTRACT", "TECHNICAL"],
        n_samples,
    )
    nb_cats = np.random.randint(0, 10, n_samples)
    best_score = np.random.uniform(0.3, 1.0, n_samples)
    has_match = np.random.randint(0, 2, n_samples)
    sentiment = np.random.uniform(-0.5, 0.8, n_samples)
    nb_sources = np.random.randint(0, 4, n_samples)

    # Features contexte
    dow = np.random.randint(0, 5, n_samples)
    nb_prev = np.random.randint(0, 15, n_samples)
    prev_wr = np.random.uniform(0.5, 1.0, n_samples)
    days_since = np.random.randint(-1, 60, n_samples)

    # Target: gagnant si range_position bas + catalyseur present
    prob = 1 / (1 + np.exp(3 * (range_pos - 0.4) - 0.5 * (nb_cats > 0).astype(float)))
    is_winner = (np.random.random(n_samples) < prob).astype(int)

    return pd.DataFrame({
        "trade_id": range(1, n_samples + 1),
        "range_position_10": range_pos,
        "range_position_20": range_pos + np.random.normal(0, 0.05, n_samples),
        "range_amplitude_10": range_amp_10,
        "range_amplitude_20": range_amp_20,
        "rsi_14": rsi,
        "macd_histogram": macd_hist,
        "bollinger_position": bollinger_pos,
        "volume_ratio_20": vol_ratio,
        "atr_14_pct": atr_pct,
        "variation_1j": var_1j,
        "variation_5j": var_5j,
        "distance_sma20": dist_sma20,
        "distance_sma50": dist_sma50,
        "catalyst_type": cat_types,
        "nb_catalysts": nb_cats,
        "best_catalyst_score": best_score,
        "has_text_match": has_match,
        "sentiment_avg": sentiment,
        "nb_news_sources": nb_sources,
        "day_of_week": dow,
        "nb_previous_trades": nb_prev,
        "previous_win_rate": prev_wr,
        "days_since_last_trade": days_since,
        "is_winner": is_winner,
    })


class TestPrepareData:
    """Tests de preparation des donnees."""

    def setup_method(self):
        self.trainer = Trainer()

    def test_prepare_data_splits_x_y(self):
        """prepare_data retourne X (features) et y (target)."""
        df = _make_feature_df(50)
        X, y = self.trainer.prepare_data(df)
        assert len(X) == 50
        assert len(y) == 50
        assert "is_winner" not in X.columns
        assert "trade_id" not in X.columns

    def test_prepare_data_encodes_catalyst_type(self):
        """catalyst_type est encode en numerique."""
        df = _make_feature_df(50)
        X, y = self.trainer.prepare_data(df)
        assert X["catalyst_type"].dtype in [np.int64, np.int32, np.float64, int]


class TestTrain:
    """Tests d'entrainement du modele."""

    def setup_method(self):
        self.trainer = Trainer()
        self.df = _make_feature_df(50)

    def test_train_returns_metrics(self):
        """train() retourne un dict avec des metriques."""
        X, y = self.trainer.prepare_data(self.df)
        metrics = self.trainer.train(X, y)
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0

    def test_model_is_set_after_train(self):
        """Le modele est accessible apres entrainement."""
        X, y = self.trainer.prepare_data(self.df)
        self.trainer.train(X, y)
        assert self.trainer.model is not None

    def test_predict_after_train(self):
        """Peut predire apres entrainement."""
        X, y = self.trainer.prepare_data(self.df)
        self.trainer.train(X, y)
        predictions = self.trainer.predict(X)
        assert len(predictions) == len(X)
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba_after_train(self):
        """Peut obtenir des probabilites apres entrainement."""
        X, y = self.trainer.prepare_data(self.df)
        self.trainer.train(X, y)
        probas = self.trainer.predict_proba(X)
        assert len(probas) == len(X)
        assert all(0 <= p <= 1 for p in probas)


class TestWalkForward:
    """Tests de validation walk-forward."""

    def setup_method(self):
        self.trainer = Trainer()

    def test_walk_forward_returns_metrics(self):
        """walk_forward_validate retourne des metriques."""
        df = _make_feature_df(50)
        # Simuler des dates pour le split
        dates = pd.bdate_range("2025-06-01", periods=50)
        df["date_achat"] = [d.strftime("%Y-%m-%d") for d in dates]

        results = self.trainer.walk_forward_validate(df, split_date="2025-07-25")
        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "baseline_accuracy" in results

    def test_walk_forward_has_predictions(self):
        """Les predictions detaillees sont retournees."""
        df = _make_feature_df(50)
        dates = pd.bdate_range("2025-06-01", periods=50)
        df["date_achat"] = [d.strftime("%Y-%m-%d") for d in dates]

        results = self.trainer.walk_forward_validate(df, split_date="2025-07-25")
        assert "predictions" in results
        assert len(results["predictions"]) > 0


class TestSaveLoad:
    """Tests de sauvegarde/chargement du modele."""

    def setup_method(self):
        self.trainer = Trainer()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.temp_dir.name, "test_model.joblib")

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_save_and_load(self):
        """Sauvegarde et recharge le modele."""
        df = _make_feature_df(50)
        X, y = self.trainer.prepare_data(df)
        self.trainer.train(X, y)
        self.trainer.save_model(self.model_path)

        # Charger dans un nouveau trainer
        trainer2 = Trainer()
        trainer2.load_model(self.model_path)
        preds = trainer2.predict(X)
        assert len(preds) == len(X)

    def test_save_without_train_raises(self):
        """Sauvegarder sans entrainer leve une erreur."""
        with pytest.raises(ValueError, match="pas entraine"):
            self.trainer.save_model(self.model_path)
