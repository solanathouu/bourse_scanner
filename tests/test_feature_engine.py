"""Tests pour le feature engine — assemblage du vecteur de features par trade."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine


def _seed_test_db(db: Database):
    """Peuple une base de test avec des donnees realistes."""
    # 2 trades SANOFI (1 gagnant, 1 perdant)
    db.insert_trades_batch([
        {
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-20",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        },
        {
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-08-15", "date_vente": "2025-08-25",
            "prix_achat": 100.0, "prix_vente": 97.0, "quantite": 10,
            "rendement_brut_pct": -3.0, "rendement_net_pct": -3.5,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        },
    ])

    # Prix synthetiques pour SAN.PA (60 jours avant + periode des trades)
    np.random.seed(42)
    dates = pd.bdate_range("2025-05-01", "2025-09-15")
    n = len(dates)
    t = np.linspace(0, 6 * np.pi, n)
    close = 97 + 5 * np.sin(t) + np.random.normal(0, 0.5, n)

    prices = []
    for i, d in enumerate(dates):
        prices.append({
            "ticker": "SAN.PA",
            "date": d.strftime("%Y-%m-%d"),
            "open": round(close[i] - 0.5, 4),
            "high": round(close[i] + 1.0, 4),
            "low": round(close[i] - 1.0, 4),
            "close": round(close[i], 4),
            "volume": int(100000 + np.random.randint(-20000, 20000)),
        })
    db.insert_prices_batch(prices)

    # News pour le trade 1
    db.insert_news_batch([
        {
            "ticker": "SAN.PA", "title": "Sanofi: resultats T2 solides",
            "source": "Reuters", "url": "https://ex.com/san1",
            "published_at": "2025-07-09", "description": "Bons resultats",
            "sentiment": 0.6, "source_api": "alpha_vantage",
        },
        {
            "ticker": "SAN.PA", "title": "FDA approves Sanofi new drug",
            "source": "Bloomberg", "url": "https://ex.com/san2",
            "published_at": "2025-07-10", "description": "Drug approved by FDA",
            "sentiment": 0.8, "source_api": "gnews",
        },
    ])
    # Catalyseurs pour trade 1
    db.insert_catalyseurs_batch([
        {"trade_id": 1, "news_id": 1, "score_pertinence": 0.9,
         "distance_jours": -1, "match_texte": 1},
        {"trade_id": 1, "news_id": 2, "score_pertinence": 1.0,
         "distance_jours": 0, "match_texte": 1},
    ])
    # Pas de catalyseurs pour trade 2


class TestBuildTradeFeatures:
    """Tests de construction des features pour un trade."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_returns_dict(self):
        """build_trade_features retourne un dict."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert isinstance(result, dict)

    def test_build_has_technical_features(self):
        """Le dict contient les features techniques."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "rsi_14" in result
        assert "range_position_10" in result
        assert "range_position_20" in result
        assert "bollinger_position" in result

    def test_build_has_catalyst_features(self):
        """Le dict contient les features catalyseur."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "catalyst_type" in result
        assert "nb_catalysts" in result
        assert "best_catalyst_score" in result
        assert "has_text_match" in result

    def test_build_has_context_features(self):
        """Le dict contient les features contexte."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "day_of_week" in result
        assert "nb_previous_trades" in result
        assert "previous_win_rate" in result

    def test_build_has_target(self):
        """Le dict contient la target is_winner."""
        trades = self.db.get_all_trades()
        # Trade 1: gagnant (+5.26%)
        result = self.engine.build_trade_features(trades[0])
        assert result["is_winner"] == 1
        # Trade 2: perdant (-3%)
        result = self.engine.build_trade_features(trades[1])
        assert result["is_winner"] == 0

    def test_trade_with_catalysts_has_type(self):
        """Un trade avec catalyseurs a un catalyst_type != TECHNICAL."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        # News "resultats T2" = EARNINGS, "FDA approves" = FDA_REGULATORY
        assert result["catalyst_type"] != "TECHNICAL"
        assert result["nb_catalysts"] == 2

    def test_trade_without_catalysts_is_technical(self):
        """Un trade sans catalyseurs a catalyst_type == TECHNICAL."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["nb_catalysts"] == 0

    def test_context_second_trade_has_history(self):
        """Le 2e trade SANOFI connait l'historique (1 trade precedent)."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["nb_previous_trades"] == 1
        assert result["previous_win_rate"] == 1.0  # Le 1er trade etait gagnant


class TestBuildAllFeatures:
    """Tests de construction de la matrice complete."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_all_returns_dataframe(self):
        """build_all_features retourne un DataFrame."""
        result = self.engine.build_all_features()
        assert isinstance(result, pd.DataFrame)

    def test_build_all_has_correct_rows(self):
        """Le DataFrame a autant de lignes que de trades clotures avec donnees."""
        result = self.engine.build_all_features()
        # 2 trades clotures, les deux devraient avoir assez de donnees
        assert len(result) == 2

    def test_build_all_has_target_column(self):
        """Le DataFrame a la colonne target."""
        result = self.engine.build_all_features()
        assert "is_winner" in result.columns

    def test_build_all_has_trade_id(self):
        """Le DataFrame a trade_id pour identifier chaque trade."""
        result = self.engine.build_all_features()
        assert "trade_id" in result.columns

    def test_feature_names_list(self):
        """get_feature_names retourne la liste des features (sans target)."""
        names = self.engine.get_feature_names()
        assert isinstance(names, list)
        assert "is_winner" not in names
        assert "trade_id" not in names
        assert "rsi_14" in names
        assert "catalyst_type" in names
