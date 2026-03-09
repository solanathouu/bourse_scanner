"""Tests pour le feature engine — assemblage du vecteur de features par trade."""

import json
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

    # Analyse LLM pour trade 1
    db.insert_trade_analysis({
        "trade_id": 1, "primary_news_id": 1,
        "catalyst_type": "EARNINGS",
        "catalyst_summary": "Nicolas a achete car resultats T2 solides",
        "catalyst_confidence": 0.85, "news_sentiment": 0.6,
        "buy_reason": "CA en hausse de 8% au T2",
        "sell_reason": "Objectif de +5% atteint",
        "trade_quality": "BON", "model_used": "gpt-4o-mini",
        "analyzed_at": "2026-02-24 19:00:00",
    })
    # Pas d'analyse LLM pour trade 2

    # Fondamentaux pour SANOFI
    db.insert_fundamental({
        "ticker": "SAN.PA", "date": "2025-06-01",
        "pe_ratio": 15.2, "pb_ratio": 2.1,
        "market_cap": 120000000000, "dividend_yield": 3.5,
        "target_price": 105.0, "analyst_count": 28,
        "recommendation": "buy", "earnings_date": "2025-07-25",
    })


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
        """Le dict contient les features catalyseur LLM."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "catalyst_type" in result
        assert "catalyst_confidence" in result
        assert "news_sentiment" in result
        assert "has_clear_catalyst" in result

    def test_build_has_context_features(self):
        """Le dict contient les features contexte."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "day_of_week" in result
        assert "nb_previous_trades" in result
        assert "previous_win_rate" in result

    def test_build_has_target(self):
        """Le dict contient la target is_winner (seuil 4.5%)."""
        trades = self.db.get_all_trades()
        # Trade 1: +5.26% >= 4.5% -> winner
        result = self.engine.build_trade_features(trades[0])
        assert result["is_winner"] == 1
        # Trade 2: -3% < 4.5% -> loser
        result = self.engine.build_trade_features(trades[1])
        assert result["is_winner"] == 0

    def test_trade_with_llm_analysis_uses_it(self):
        """Un trade avec analyse LLM utilise les features LLM."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert result["catalyst_type"] == "EARNINGS"
        assert result["catalyst_confidence"] == 0.85
        assert result["has_clear_catalyst"] == 1

    def test_trade_without_llm_falls_back(self):
        """Un trade sans analyse LLM utilise le fallback TECHNICAL."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["catalyst_type"] == "TECHNICAL"
        assert result["catalyst_confidence"] == 0.0
        assert result["has_clear_catalyst"] == 0

    def test_build_has_fundamental_features(self):
        """Le dict contient les features fondamentales."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert "pe_ratio" in result
        assert "pb_ratio" in result
        assert "target_upside_pct" in result
        assert "analyst_count" in result
        assert "days_to_earnings" in result
        assert "recommendation_score" in result

    def test_trade_with_fundamentals(self):
        """Un trade avec fondamentaux utilise les donnees reelles."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[0])
        assert result["pe_ratio"] == 15.2
        assert result["analyst_count"] == 28
        assert result["recommendation_score"] == 4  # "buy"
        # target_upside = (105 - 95) / 95 * 100 = 10.53%
        assert abs(result["target_upside_pct"] - 10.53) < 0.1

    def test_trade_without_fundamentals_defaults(self):
        """Un trade sans fondamentaux utilise les valeurs par defaut."""
        # Trade 2 date_achat=2025-08-15, fondamentaux date=2025-06-01 (avant)
        # mais on va tester un cas sans aucun fondamental
        # Creer un nouveau trade pour une action sans fondamentaux
        self.db.insert_trade_complet({
            "isin": "FR9999999999", "nom_action": "AIR LIQUIDE",
            "date_achat": "2025-01-01", "date_vente": "2025-01-10",
            "prix_achat": 150.0, "prix_vente": 155.0, "quantite": 5,
            "rendement_brut_pct": 3.33, "rendement_net_pct": 3.0,
            "duree_jours": 9, "frais_totaux": 5.0, "statut": "CLOTURE",
        })
        # AI.PA n'a pas de prix, donc build_trade_features retourne None
        # Testons _build_fundamental_features directement
        result = self.engine._build_fundamental_features("AI.PA", "2025-01-01", 150.0)
        assert result["pe_ratio"] == 0.0
        assert result["analyst_count"] == 0
        assert result["days_to_earnings"] == -1

    def test_context_second_trade_has_history(self):
        """Le 2e trade SANOFI connait l'historique (1 trade precedent)."""
        trades = self.db.get_all_trades()
        result = self.engine.build_trade_features(trades[1])
        assert result["nb_previous_trades"] == 1
        assert result["previous_win_rate"] == 1.0  # Le 1er trade etait gagnant


class TestRealtimeFeatures:
    """Tests pour build_realtime_features."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_returns_dict(self):
        """build_realtime_features retourne un dict."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert isinstance(result, dict)

    def test_has_technical_features(self):
        """Le dict contient les features techniques."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert "rsi_14" in result
        assert "bollinger_position" in result

    def test_has_catalyst_features(self):
        """Le dict contient les features catalyseur."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert "catalyst_type" in result
        assert "news_sentiment" in result

    def test_has_fundamental_features(self):
        """Le dict contient les features fondamentales."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert "pe_ratio" in result
        assert "analyst_count" in result

    def test_has_context_features(self):
        """Le dict contient les features contexte."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert "day_of_week" in result
        assert "nb_previous_trades" in result

    def test_no_target_column(self):
        """Les features temps reel n'ont pas is_winner ni trade_id."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert "is_winner" not in result
        assert "trade_id" not in result

    def test_no_data_returns_none(self):
        """Retourne None si pas assez de prix."""
        result = self.engine.build_realtime_features("UNKNOWN.PA", 50.0, "2025-08-15")
        assert result is None

    def test_with_recent_news(self):
        """Les news recentes alimentent les features catalyseur."""
        # Ajouter une news recente
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi resultats record",
            "source": "BFM", "url": "https://ex.com/new",
            "published_at": "2025-08-14", "description": "CA en hausse",
            "sentiment": 0.7, "source_api": "gnews",
        })
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert result["has_clear_catalyst"] == 1
        assert result["news_sentiment"] > 0


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
        assert "bid_ask_volume_ratio" in names


class TestBuildCombinedFeatures:
    """Tests pour build_combined_features (apprentissage continu)."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def _insert_review(self, signal_id, ticker, outcome, features_json,
                       signal_date="2026-03-01"):
        """Helper pour inserer un signal + review."""
        self.db.insert_signal({
            "ticker": ticker,
            "date": signal_date,
            "score": 0.80,
            "signal_price": 10.0,
            "sent_at": f"{signal_date} 10:00:00",
        })
        self.db.insert_signal_review({
            "signal_id": signal_id,
            "ticker": ticker,
            "signal_date": signal_date,
            "signal_price": 10.0,
            "review_date": "2026-03-04",
            "review_price": 10.5,
            "performance_pct": 5.0,
            "outcome": outcome,
            "failure_reason": None,
            "catalyst_type": "EARNINGS",
            "features_json": features_json,
            "reviewed_at": "2026-03-04 18:00:00",
        })

    def test_build_combined_includes_trades_and_reviews(self):
        """Le DataFrame combine contient trades historiques + reviews."""
        features = {"rsi_14": 40.0, "catalyst_type": "EARNINGS"}
        self._insert_review(1, "SAN.PA", "WIN", json.dumps(features))
        result = self.engine.build_combined_features()
        # 2 trades + 1 review = 3
        assert len(result) == 3

    def test_combined_target_threshold_4_5(self):
        """Trades < 4.5% rendement -> is_winner=0."""
        result = self.engine.build_combined_features()
        # Trade 1: +5.26% -> 1, Trade 2: -3% -> 0
        trade1 = result[result["trade_id"] == 1].iloc[0]
        trade2 = result[result["trade_id"] == 2].iloc[0]
        assert trade1["is_winner"] == 1
        assert trade2["is_winner"] == 0

    def test_combined_review_win_is_winner_1(self):
        """Review WIN -> is_winner=1."""
        features = {"rsi_14": 40.0}
        self._insert_review(1, "SAN.PA", "WIN", json.dumps(features))
        result = self.engine.build_combined_features()
        review_row = result[result["trade_id"] == -1].iloc[0]
        assert review_row["is_winner"] == 1

    def test_combined_review_loss_is_winner_0(self):
        """Review LOSS -> is_winner=0."""
        features = {"rsi_14": 60.0}
        self._insert_review(1, "SAN.PA", "LOSS", json.dumps(features),
                            signal_date="2026-03-02")
        result = self.engine.build_combined_features()
        review_row = result[result["trade_id"] == -1].iloc[0]
        assert review_row["is_winner"] == 0

    def test_combined_review_neutral_is_winner_0(self):
        """Review NEUTRAL -> is_winner=0."""
        features = {"rsi_14": 50.0}
        self._insert_review(1, "SAN.PA", "NEUTRAL", json.dumps(features),
                            signal_date="2026-03-03")
        result = self.engine.build_combined_features()
        review_row = result[result["trade_id"] == -1].iloc[0]
        assert review_row["is_winner"] == 0

    def test_combined_skips_null_features_json(self):
        """Reviews sans features_json sont ignorees."""
        self._insert_review(1, "SAN.PA", "WIN", None)
        result = self.engine.build_combined_features()
        # Seulement les 2 trades, pas la review sans features
        assert len(result) == 2

    def test_combined_has_date_achat_for_walk_forward(self):
        """Les reviews ont date_achat = signal_date pour le walk-forward."""
        features = {"rsi_14": 40.0}
        self._insert_review(1, "SAN.PA", "WIN", json.dumps(features),
                            signal_date="2026-03-05")
        result = self.engine.build_combined_features()
        review_row = result[result["trade_id"] == -1].iloc[0]
        assert review_row["date_achat"] == "2026-03-05"

    def test_combined_negative_trade_id_for_reviews(self):
        """Les reviews ont trade_id negatif (-signal_id)."""
        features = {"rsi_14": 40.0}
        self._insert_review(1, "SAN.PA", "WIN", json.dumps(features))
        result = self.engine.build_combined_features()
        review_ids = result[result["trade_id"] < 0]["trade_id"].tolist()
        assert len(review_ids) == 1
        assert review_ids[0] == -1


class TestOrderbookFeatures:
    """Tests pour les features du carnet d'ordres."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        _seed_test_db(self.db)
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_build_has_orderbook_features(self):
        """Les features temps reel contiennent les features orderbook."""
        result = self.engine.build_realtime_features("SAN.PA", 98.0, "2025-08-15")
        assert "bid_ask_volume_ratio" in result
        assert "spread_pct" in result
        assert "bid_depth_concentration" in result
        assert "bid_ask_order_ratio" in result

    def test_orderbook_features_default_no_data(self):
        """Sans snapshot orderbook, les features sont a 0.0."""
        result = self.engine._build_orderbook_features("SAN.PA")
        assert result["bid_ask_volume_ratio"] == 0.0
        assert result["spread_pct"] == 0.0
        assert result["bid_depth_concentration"] == 0.0
        assert result["bid_ask_order_ratio"] == 0.0

    def test_orderbook_features_with_data(self):
        """Avec un snapshot, les features sont calculees."""
        raw_data = {
            "bids": [
                {"price": 10.0, "quantity": 500, "orders": 3},
                {"price": 9.9, "quantity": 300, "orders": 2},
                {"price": 9.8, "quantity": 200, "orders": 1},
                {"price": 9.7, "quantity": 100, "orders": 1},
                {"price": 9.6, "quantity": 50, "orders": 1},
            ],
            "asks": [
                {"price": 10.1, "quantity": 400, "orders": 2},
                {"price": 10.2, "quantity": 200, "orders": 1},
                {"price": 10.3, "quantity": 100, "orders": 1},
            ],
        }
        self.db.insert_orderbook_snapshot({
            "ticker": "SAN.PA",
            "snapshot_time": "2025-08-15 10:00:00",
            "best_bid": 10.0,
            "best_ask": 10.1,
            "bid_volume_total": 1150,
            "ask_volume_total": 700,
            "bid_orders_total": 8,
            "ask_orders_total": 4,
            "spread_pct": 1.0,
            "bid_ask_volume_ratio": 1.6429,
            "raw_json": json.dumps(raw_data),
        })
        result = self.engine._build_orderbook_features("SAN.PA")
        assert result["bid_ask_volume_ratio"] == 1.6429
        assert result["spread_pct"] == 1.0
        assert result["bid_ask_order_ratio"] == 2.0  # 8/4
        assert result["bid_depth_concentration"] > 0
