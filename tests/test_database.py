"""Tests pour la couche base de données SQLite."""

import sqlite3
import os
import tempfile

import pytest

from src.core.database import Database


def test_database_creates_tables():
    """Vérifie que init_db crée toutes les tables nécessaires."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "executions" in tables
        assert "trades_complets" in tables
        assert "signal_reviews" in tables
        assert "filter_rules" in tables
        assert "model_versions" in tables


def test_database_insert_execution():
    """Vérifie qu'on peut insérer une exécution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        execution = {
            "date_execution": "2024-03-15",
            "heure_execution": "09:32:15",
            "sens": "ACHAT",
            "nom_action": "LVMH",
            "isin": "FR0000121014",
            "quantite": 50,
            "prix_unitaire": 842.30,
            "montant_brut": 42115.00,
            "commission": 4.95,
            "frais": 0.00,
            "montant_net": 42119.95,
            "fichier_source": "avis_001.pdf",
        }
        db.insert_execution(execution)

        results = db.get_all_executions()
        assert len(results) == 1
        assert results[0]["nom_action"] == "LVMH"
        assert results[0]["sens"] == "ACHAT"


def test_database_insert_trade_complet():
    """Vérifie qu'on peut insérer un trade complet."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        trade = {
            "isin": "FR0000121014",
            "nom_action": "LVMH",
            "date_achat": "2024-03-15 09:32:15",
            "date_vente": "2024-03-20 14:15:30",
            "prix_achat": 842.30,
            "prix_vente": 878.10,
            "quantite": 50,
            "rendement_brut_pct": 4.25,
            "rendement_net_pct": 4.13,
            "duree_jours": 5.2,
            "frais_totaux": 9.90,
            "statut": "CLOTURE",
        }
        db.insert_trade_complet(trade)

        results = db.get_all_trades()
        assert len(results) == 1
        assert results[0]["nom_action"] == "LVMH"
        assert results[0]["statut"] == "CLOTURE"


def test_database_insert_batch():
    """Vérifie l'insertion en batch de plusieurs exécutions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        executions = [
            {
                "date_execution": "2024-03-15",
                "heure_execution": "09:32:15",
                "sens": "ACHAT",
                "nom_action": "LVMH",
                "isin": "FR0000121014",
                "quantite": 50,
                "prix_unitaire": 842.30,
                "montant_brut": 42115.00,
                "commission": 4.95,
                "frais": 0.00,
                "montant_net": 42119.95,
                "fichier_source": "avis_001.pdf",
            },
            {
                "date_execution": "2024-03-16",
                "heure_execution": "10:00:00",
                "sens": "ACHAT",
                "nom_action": "BNP",
                "isin": "FR0000131104",
                "quantite": 100,
                "prix_unitaire": 60.00,
                "montant_brut": 6000.00,
                "commission": 4.95,
                "frais": 0.00,
                "montant_net": 6004.95,
                "fichier_source": "avis_002.pdf",
            },
        ]
        db.insert_executions_batch(executions)

        assert db.count_executions() == 2


def test_database_get_executions_by_isin():
    """Vérifie le filtrage par ISIN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        executions = [
            {
                "date_execution": "2024-03-15",
                "heure_execution": "09:00:00",
                "sens": "ACHAT",
                "nom_action": "LVMH",
                "isin": "FR0000121014",
                "quantite": 50,
                "prix_unitaire": 842.30,
                "montant_brut": 42115.00,
                "commission": 4.95,
                "frais": 0.00,
                "montant_net": 42119.95,
                "fichier_source": "avis_001.pdf",
            },
            {
                "date_execution": "2024-03-16",
                "heure_execution": "10:00:00",
                "sens": "ACHAT",
                "nom_action": "BNP",
                "isin": "FR0000131104",
                "quantite": 100,
                "prix_unitaire": 60.00,
                "montant_brut": 6000.00,
                "commission": 4.95,
                "frais": 0.00,
                "montant_net": 6004.95,
                "fichier_source": "avis_002.pdf",
            },
        ]
        db.insert_executions_batch(executions)

        lvmh = db.get_executions_by_isin("FR0000121014")
        assert len(lvmh) == 1
        assert lvmh[0]["nom_action"] == "LVMH"


def test_database_get_distinct_isins():
    """Vérifie la récupération des ISIN distincts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = Database(db_path)
        db.init_db()

        executions = [
            {
                "date_execution": "2024-03-15",
                "heure_execution": "09:00:00",
                "sens": "ACHAT",
                "nom_action": "LVMH",
                "isin": "FR0000121014",
                "quantite": 50,
                "prix_unitaire": 842.30,
                "montant_brut": 42115.00,
                "commission": 4.95,
                "frais": 0.00,
                "montant_net": 42119.95,
                "fichier_source": "avis_001.pdf",
            },
            {
                "date_execution": "2024-03-16",
                "heure_execution": "10:00:00",
                "sens": "ACHAT",
                "nom_action": "BNP",
                "isin": "FR0000131104",
                "quantite": 100,
                "prix_unitaire": 60.00,
                "montant_brut": 6000.00,
                "commission": 4.95,
                "frais": 0.00,
                "montant_net": 6004.95,
                "fichier_source": "avis_002.pdf",
            },
        ]
        db.insert_executions_batch(executions)

        isins = db.get_distinct_isins()
        assert len(isins) == 2
        assert "FR0000121014" in isins
        assert "FR0000131104" in isins


class TestPricesTable:
    """Tests pour la table prices."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_price(self):
        """Insere un prix et le recupere."""
        price = {
            "ticker": "SAN.PA",
            "date": "2025-06-15",
            "open": 95.50,
            "high": 96.80,
            "low": 95.10,
            "close": 96.20,
            "volume": 1234567,
        }
        self.db.insert_price(price)
        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 1
        assert prices[0]["close"] == 96.20

    def test_insert_prices_batch(self):
        """Insere plusieurs prix en batch."""
        prices = [
            {"ticker": "SAN.PA", "date": "2025-06-15", "open": 95.5,
             "high": 96.8, "low": 95.1, "close": 96.2, "volume": 100},
            {"ticker": "SAN.PA", "date": "2025-06-16", "open": 96.2,
             "high": 97.0, "low": 95.8, "close": 96.5, "volume": 200},
        ]
        self.db.insert_prices_batch(prices)
        result = self.db.get_prices("SAN.PA")
        assert len(result) == 2

    def test_insert_price_doublon_ignore(self):
        """Un doublon (meme ticker+date) est ignore silencieusement."""
        price = {"ticker": "SAN.PA", "date": "2025-06-15", "open": 95.5,
                 "high": 96.8, "low": 95.1, "close": 96.2, "volume": 100}
        self.db.insert_price(price)
        self.db.insert_price(price)  # doublon
        prices = self.db.get_prices("SAN.PA")
        assert len(prices) == 1


class TestNewsTable:
    """Tests pour la table news."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_news(self):
        """Insere une news et la recupere."""
        news = {
            "ticker": "SAN.PA",
            "title": "Sanofi: resultats T2 au-dessus des attentes",
            "source": "BFM Bourse",
            "url": "https://example.com/sanofi-t2",
            "published_at": "2025-06-15 08:30:00",
            "description": "Le laboratoire publie des resultats...",
        }
        self.db.insert_news(news)
        result = self.db.get_news("SAN.PA")
        assert len(result) == 1
        assert result[0]["title"] == "Sanofi: resultats T2 au-dessus des attentes"

    def test_insert_news_batch(self):
        """Insere plusieurs news en batch."""
        news_list = [
            {"ticker": "SAN.PA", "title": "News 1", "source": "BFM",
             "url": "https://example.com/1", "published_at": "2025-06-15 08:30:00",
             "description": "Desc 1"},
            {"ticker": "SAN.PA", "title": "News 2", "source": "Les Echos",
             "url": "https://example.com/2", "published_at": "2025-06-16 09:00:00",
             "description": "Desc 2"},
        ]
        self.db.insert_news_batch(news_list)
        result = self.db.get_news("SAN.PA")
        assert len(result) == 2

    def test_insert_news_doublon_url_ignore(self):
        """Un doublon (meme URL) est ignore silencieusement."""
        news = {"ticker": "SAN.PA", "title": "News 1", "source": "BFM",
                "url": "https://example.com/1", "published_at": "2025-06-15 08:30:00",
                "description": "Desc 1"}
        self.db.insert_news(news)
        self.db.insert_news(news)  # doublon
        result = self.db.get_news("SAN.PA")
        assert len(result) == 1


class TestCatalyseursTable:
    """Tests pour la table trade_catalyseurs."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        # Inserer un trade et une news pour les FK
        self.db.insert_trade_complet({
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-15",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 5, "frais_totaux": 2.5, "statut": "CLOTURE",
        })
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi annonce resultats",
            "source": "Reuters", "url": "https://example.com/sanofi-1",
            "published_at": "2025-07-09", "description": "Bons resultats",
            "sentiment": 0.5, "source_api": "gnews",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_catalyseur(self):
        """Insere un catalyseur et le recupere."""
        catalyseur = {
            "trade_id": 1, "news_id": 1,
            "score_pertinence": 0.8, "distance_jours": -1,
            "match_texte": 1,
        }
        self.db.insert_catalyseur(catalyseur)
        result = self.db.get_catalyseurs(trade_id=1)
        assert len(result) == 1
        assert result[0]["score_pertinence"] == 0.8
        assert result[0]["distance_jours"] == -1
        assert result[0]["match_texte"] == 1

    def test_insert_catalyseurs_batch(self):
        """Insere plusieurs catalyseurs en batch."""
        # Ajouter une 2e news
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "Sanofi FDA approval",
            "source": "Bloomberg", "url": "https://example.com/sanofi-2",
            "published_at": "2025-07-10", "description": "FDA approves drug",
            "sentiment": 0.8, "source_api": "alpha_vantage",
        })
        catalyseurs = [
            {"trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
             "distance_jours": -1, "match_texte": 1},
            {"trade_id": 1, "news_id": 2, "score_pertinence": 1.0,
             "distance_jours": 0, "match_texte": 1},
        ]
        self.db.insert_catalyseurs_batch(catalyseurs)
        result = self.db.get_catalyseurs(trade_id=1)
        assert len(result) == 2

    def test_insert_catalyseur_doublon_ignore(self):
        """Les doublons (meme trade_id+news_id) sont ignores."""
        cat = {"trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
               "distance_jours": -1, "match_texte": 1}
        self.db.insert_catalyseur(cat)
        self.db.insert_catalyseur(cat)  # doublon
        assert self.db.count_catalyseurs() == 1

    def test_count_catalyseurs(self):
        """Compte le nombre total de catalyseurs."""
        assert self.db.count_catalyseurs() == 0
        self.db.insert_catalyseur({
            "trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
            "distance_jours": -1, "match_texte": 1,
        })
        assert self.db.count_catalyseurs() == 1

    def test_clear_catalyseurs(self):
        """Vide la table trade_catalyseurs."""
        self.db.insert_catalyseur({
            "trade_id": 1, "news_id": 1, "score_pertinence": 0.8,
            "distance_jours": -1, "match_texte": 1,
        })
        self.db.clear_catalyseurs()
        assert self.db.count_catalyseurs() == 0

    def test_get_news_for_trade_window(self):
        """Recupere les news dans une fenetre temporelle pour un ticker."""
        result = self.db.get_news_in_window(
            ticker="SAN.PA",
            date_start="2025-07-07",
            date_end="2025-07-11",
        )
        assert len(result) == 1
        assert result[0]["title"] == "Sanofi annonce resultats"


class TestNewsSentimentUpdate:
    """Tests pour get_news_without_sentiment et update_news_sentiment."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_get_news_without_sentiment(self):
        """Recupere les news sans sentiment."""
        self.db.insert_news_batch([
            {"ticker": "SAN.PA", "title": "News 1", "source": "BFM",
             "url": "https://example.com/1", "published_at": "2025-06-15",
             "description": "Desc", "sentiment": None, "source_api": "gnews"},
            {"ticker": "SAN.PA", "title": "News 2", "source": "Reuters",
             "url": "https://example.com/2", "published_at": "2025-06-16",
             "description": "Desc", "sentiment": 0.5, "source_api": "alpha_vantage"},
        ])
        result = self.db.get_news_without_sentiment()
        assert len(result) == 1
        assert result[0]["title"] == "News 1"

    def test_update_news_sentiment(self):
        """Met a jour le sentiment d'une news."""
        self.db.insert_news({
            "ticker": "SAN.PA", "title": "News 1", "source": "BFM",
            "url": "https://example.com/1", "published_at": "2025-06-15",
            "description": "Desc",
        })
        news = self.db.get_news("SAN.PA")
        assert news[0]["sentiment"] is None

        self.db.update_news_sentiment(news[0]["id"], 0.75)

        news = self.db.get_news("SAN.PA")
        assert news[0]["sentiment"] == 0.75
        # Plus dans la liste sans sentiment
        assert len(self.db.get_news_without_sentiment()) == 0


class TestFundamentalsTable:
    """Tests pour la table fundamentals."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_fundamental(self):
        """Insere et recupere des fondamentaux."""
        fundamental = {
            "ticker": "SAN.PA", "date": "2026-02-24",
            "pe_ratio": 15.2, "pb_ratio": 2.1,
            "market_cap": 120000000000, "dividend_yield": 3.5,
            "target_price": 105.0, "analyst_count": 28,
            "recommendation": "buy", "earnings_date": "2026-04-25",
        }
        self.db.insert_fundamental(fundamental)
        result = self.db.get_fundamentals("SAN.PA")
        assert len(result) == 1
        assert result[0]["pe_ratio"] == 15.2
        assert result[0]["recommendation"] == "buy"

    def test_insert_fundamental_doublon_ignore(self):
        """Un doublon (meme ticker+date) est ignore."""
        f = {"ticker": "SAN.PA", "date": "2026-02-24",
             "pe_ratio": 15.0, "pb_ratio": None, "market_cap": None,
             "dividend_yield": None, "target_price": None,
             "analyst_count": None, "recommendation": None,
             "earnings_date": None}
        self.db.insert_fundamental(f)
        self.db.insert_fundamental(f)
        assert self.db.count_fundamentals() == 1

    def test_get_fundamental_at_date(self):
        """Recupere le fondamental le plus recent avant une date."""
        self.db.insert_fundamental({
            "ticker": "SAN.PA", "date": "2026-01-15",
            "pe_ratio": 14.0, "pb_ratio": None, "market_cap": None,
            "dividend_yield": None, "target_price": None,
            "analyst_count": None, "recommendation": None,
            "earnings_date": None,
        })
        self.db.insert_fundamental({
            "ticker": "SAN.PA", "date": "2026-02-15",
            "pe_ratio": 15.5, "pb_ratio": None, "market_cap": None,
            "dividend_yield": None, "target_price": None,
            "analyst_count": None, "recommendation": None,
            "earnings_date": None,
        })
        # Au 2026-02-20, le plus recent est celui du 2026-02-15
        result = self.db.get_fundamental_at_date("SAN.PA", "2026-02-20")
        assert result is not None
        assert result["pe_ratio"] == 15.5

    def test_get_fundamental_at_date_none(self):
        """Retourne None si aucun fondamental avant la date."""
        result = self.db.get_fundamental_at_date("SAN.PA", "2026-02-20")
        assert result is None

    def test_count_fundamentals(self):
        """Compte les fondamentaux."""
        assert self.db.count_fundamentals() == 0
        self.db.insert_fundamental({
            "ticker": "SAN.PA", "date": "2026-02-24",
            "pe_ratio": 15.0, "pb_ratio": None, "market_cap": None,
            "dividend_yield": None, "target_price": None,
            "analyst_count": None, "recommendation": None,
            "earnings_date": None,
        })
        assert self.db.count_fundamentals() == 1


class TestSignalsTable:
    """Tests pour la table signals."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_signal(self):
        """Insere un signal et le recupere."""
        signal = {
            "ticker": "SAN.PA", "date": "2026-02-26",
            "score": 0.82, "catalyst_type": "EARNINGS",
            "catalyst_news_title": "Sanofi: resultats T2",
            "features_json": '{"rsi_14": 38.2}',
            "sent_at": "2026-02-26 10:00:00",
        }
        self.db.insert_signal(signal)
        result = self.db.get_signals("SAN.PA")
        assert len(result) == 1
        assert result[0]["score"] == 0.82
        assert result[0]["catalyst_type"] == "EARNINGS"

    def test_get_latest_signal(self):
        """Recupere le signal le plus recent pour un ticker."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-02-25",
            "score": 0.70,
        })
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-02-26",
            "score": 0.85,
        })
        latest = self.db.get_latest_signal("SAN.PA")
        assert latest is not None
        assert latest["date"] == "2026-02-26"
        assert latest["score"] == 0.85

    def test_get_latest_signal_none(self):
        """Retourne None si aucun signal pour ce ticker."""
        result = self.db.get_latest_signal("UNKNOWN.PA")
        assert result is None

    def test_count_signals(self):
        """Compte les signaux."""
        assert self.db.count_signals() == 0
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-02-26", "score": 0.80,
        })
        assert self.db.count_signals() == 1

    def test_insert_signal_doublon_ignore(self):
        """Un doublon (meme ticker+date) est ignore."""
        signal = {"ticker": "SAN.PA", "date": "2026-02-26", "score": 0.80}
        self.db.insert_signal(signal)
        self.db.insert_signal(signal)
        assert self.db.count_signals() == 1


class TestSignalPriceMigration:
    """Tests pour la migration signal_price sur la table signals."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_signal_price_column_exists(self):
        """La colonne signal_price existe apres init_db."""
        conn = sqlite3.connect(self.db_path)
        columns = [row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()]
        conn.close()
        assert "signal_price" in columns

    def test_insert_signal_with_price(self):
        """Insere un signal avec signal_price."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 96.50,
        })
        result = self.db.get_signals("SAN.PA")
        assert len(result) == 1
        assert result[0]["signal_price"] == 96.50

    def test_insert_signal_without_price(self):
        """Insere un signal sans signal_price (default None)."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01", "score": 0.80,
        })
        result = self.db.get_signals("SAN.PA")
        assert result[0]["signal_price"] is None


class TestSignalReviews:
    """Tests CRUD pour la table signal_reviews."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        # Insert signals to reference
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-02-26",
            "score": 0.82, "signal_price": 96.50,
            "sent_at": "2026-02-26 10:00:00",
        })
        self.db.insert_signal({
            "ticker": "MC.PA", "date": "2026-02-25",
            "score": 0.78, "signal_price": 850.0,
            "sent_at": "2026-02-25 11:00:00",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_review(self):
        """Insere et recupere une review."""
        review = {
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-02-26", "signal_price": 96.50,
            "review_date": "2026-03-01", "review_price": 100.0,
            "performance_pct": 3.63, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": "EARNINGS",
            "features_json": '{"rsi_14": 38.2}',
            "reviewed_at": "2026-03-01 18:00:00",
        }
        self.db.insert_signal_review(review)
        result = self.db.get_signal_reviews()
        assert len(result) == 1
        assert result[0]["ticker"] == "SAN.PA"
        assert result[0]["performance_pct"] == 3.63
        assert result[0]["outcome"] == "WIN"

    def test_insert_review_doublon_ignore(self):
        """Un doublon (meme signal_id) est ignore."""
        review = {
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-02-26", "signal_price": 96.50,
            "review_date": "2026-03-01", "review_price": 100.0,
            "performance_pct": 3.63, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": "EARNINGS",
            "features_json": None, "reviewed_at": "2026-03-01 18:00:00",
        }
        self.db.insert_signal_review(review)
        self.db.insert_signal_review(review)
        result = self.db.get_signal_reviews()
        assert len(result) == 1

    def test_get_reviews_by_ticker(self):
        """Filtre les reviews par ticker."""
        self.db.insert_signal_review({
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-02-26", "signal_price": 96.50,
            "review_date": "2026-03-01", "review_price": 100.0,
            "performance_pct": 3.63, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": None,
            "features_json": None, "reviewed_at": "2026-03-01 18:00:00",
        })
        self.db.insert_signal_review({
            "signal_id": 2, "ticker": "MC.PA",
            "signal_date": "2026-02-25", "signal_price": 850.0,
            "review_date": "2026-02-28", "review_price": 840.0,
            "performance_pct": -1.18, "outcome": "LOSS",
            "failure_reason": "market_downturn", "catalyst_type": None,
            "features_json": None, "reviewed_at": "2026-02-28 18:00:00",
        })
        san = self.db.get_signal_reviews("SAN.PA")
        assert len(san) == 1
        assert san[0]["ticker"] == "SAN.PA"
        all_reviews = self.db.get_signal_reviews()
        assert len(all_reviews) == 2

    def test_get_pending_signal_reviews(self):
        """Recupere les signaux envoyes il y a 3+ jours sans review."""
        pending = self.db.get_pending_signal_reviews("2026-03-01")
        # signal du 2026-02-26 (5 jours ago) + signal du 2026-02-25 (4 jours ago)
        assert len(pending) == 2
        # Trop tot: seulement 1 jour apres
        pending_early = self.db.get_pending_signal_reviews("2026-02-27")
        assert len(pending_early) == 0

    def test_get_pending_excludes_reviewed(self):
        """Les signaux deja reviewes ne sont pas dans pending."""
        self.db.insert_signal_review({
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-02-26", "signal_price": 96.50,
            "review_date": "2026-03-01", "review_price": 100.0,
            "performance_pct": 3.63, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": None,
            "features_json": None, "reviewed_at": "2026-03-01 18:00:00",
        })
        pending = self.db.get_pending_signal_reviews("2026-03-01")
        assert len(pending) == 1  # only MC.PA signal left
        assert pending[0]["ticker"] == "MC.PA"

    def test_get_reviews_in_period(self):
        """Recupere les reviews dans une periode donnee."""
        self.db.insert_signal_review({
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-02-26", "signal_price": 96.50,
            "review_date": "2026-03-01", "review_price": 100.0,
            "performance_pct": 3.63, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": None,
            "features_json": None, "reviewed_at": "2026-03-01 18:00:00",
        })
        result = self.db.get_reviews_in_period("2026-02-20", "2026-02-28")
        assert len(result) == 1
        result_empty = self.db.get_reviews_in_period("2026-03-01", "2026-03-10")
        assert len(result_empty) == 0

    def test_get_review_stats(self):
        """Retourne les stats des reviews."""
        self.db.insert_signal_review({
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-02-26", "signal_price": 96.50,
            "review_date": "2026-03-01", "review_price": 100.0,
            "performance_pct": 3.63, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": None,
            "features_json": None, "reviewed_at": "2026-03-01 18:00:00",
        })
        self.db.insert_signal_review({
            "signal_id": 2, "ticker": "MC.PA",
            "signal_date": "2026-02-25", "signal_price": 850.0,
            "review_date": "2026-02-28", "review_price": 840.0,
            "performance_pct": -1.18, "outcome": "LOSS",
            "failure_reason": "market_downturn", "catalyst_type": None,
            "features_json": None, "reviewed_at": "2026-02-28 18:00:00",
        })
        stats = self.db.get_review_stats()
        assert stats["total"] == 2
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["neutrals"] == 0

    def test_get_review_stats_empty(self):
        """Stats vides quand aucune review."""
        stats = self.db.get_review_stats()
        assert stats == {"total": 0, "wins": 0, "losses": 0, "neutrals": 0}


class TestFilterRules:
    """Tests CRUD pour la table filter_rules."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_rule(self):
        """Insere et recupere une regle de filtrage."""
        rule = {
            "rule_type": "exclude_catalyst",
            "rule_json": '{"catalyst_type": "UNKNOWN", "min_samples": 5}',
            "win_rate": 0.30,
            "sample_size": 10,
            "created_at": "2026-03-01 18:00:00",
            "active": 1,
        }
        self.db.insert_filter_rule(rule)
        result = self.db.get_active_filter_rules()
        assert len(result) == 1
        assert result[0]["rule_type"] == "exclude_catalyst"
        assert result[0]["win_rate"] == 0.30

    def test_deactivate_rule(self):
        """Desactive une regle de filtrage."""
        self.db.insert_filter_rule({
            "rule_type": "exclude_catalyst",
            "rule_json": '{"catalyst_type": "UNKNOWN"}',
            "created_at": "2026-03-01 18:00:00",
        })
        rules = self.db.get_active_filter_rules()
        assert len(rules) == 1
        self.db.deactivate_filter_rule(rules[0]["id"])
        rules = self.db.get_active_filter_rules()
        assert len(rules) == 0

    def test_multiple_rules_active_filter(self):
        """Seules les regles actives sont retournees."""
        self.db.insert_filter_rule({
            "rule_type": "exclude_catalyst",
            "rule_json": '{"catalyst_type": "UNKNOWN"}',
            "created_at": "2026-03-01 18:00:00",
        })
        self.db.insert_filter_rule({
            "rule_type": "min_score",
            "rule_json": '{"min_score": 0.80}',
            "created_at": "2026-03-01 18:00:00",
            "active": 0,  # inactive
        })
        active = self.db.get_active_filter_rules()
        assert len(active) == 1
        assert active[0]["rule_type"] == "exclude_catalyst"

    def test_insert_rule_defaults(self):
        """Les valeurs par defaut (win_rate, sample_size, active) fonctionnent."""
        self.db.insert_filter_rule({
            "rule_type": "test",
            "rule_json": "{}",
            "created_at": "2026-03-01 18:00:00",
        })
        result = self.db.get_active_filter_rules()
        assert result[0]["win_rate"] is None
        assert result[0]["sample_size"] is None
        assert result[0]["active"] == 1


class TestModelVersions:
    """Tests CRUD pour la table model_versions."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_version(self):
        """Insere et recupere une version de modele."""
        version = {
            "version": "v2.0", "file_path": "data/models/nicolas_v2.joblib",
            "trained_at": "2026-03-01 18:00:00", "training_signals": 50,
            "accuracy": 0.85, "precision_score": 0.82, "recall": 0.90,
            "f1": 0.86, "is_active": 1, "notes": "Retrained with feedback",
        }
        self.db.insert_model_version(version)
        result = self.db.get_all_model_versions()
        assert len(result) == 1
        assert result[0]["version"] == "v2.0"
        assert result[0]["accuracy"] == 0.85
        assert result[0]["is_active"] == 1

    def test_get_active_model_version(self):
        """Recupere la version active."""
        self.db.insert_model_version({
            "version": "v1.0", "file_path": "data/models/v1.joblib",
            "trained_at": "2026-02-01 18:00:00", "is_active": 0,
        })
        self.db.insert_model_version({
            "version": "v2.0", "file_path": "data/models/v2.joblib",
            "trained_at": "2026-03-01 18:00:00", "is_active": 1,
        })
        active = self.db.get_active_model_version()
        assert active is not None
        assert active["version"] == "v2.0"

    def test_get_active_model_version_none(self):
        """Retourne None si aucun modele actif."""
        result = self.db.get_active_model_version()
        assert result is None

    def test_set_active_model(self):
        """Active un modele et desactive les autres."""
        self.db.insert_model_version({
            "version": "v1.0", "file_path": "data/models/v1.joblib",
            "trained_at": "2026-02-01 18:00:00", "is_active": 1,
        })
        self.db.insert_model_version({
            "version": "v2.0", "file_path": "data/models/v2.joblib",
            "trained_at": "2026-03-01 18:00:00", "is_active": 0,
        })
        # Activate v2
        self.db.set_active_model(2)
        active = self.db.get_active_model_version()
        assert active["version"] == "v2.0"
        # v1 should be inactive now
        all_versions = self.db.get_all_model_versions()
        assert all_versions[0]["is_active"] == 0
        assert all_versions[1]["is_active"] == 1

    def test_get_all_model_versions_ordered(self):
        """Recupere toutes les versions triees par id."""
        self.db.insert_model_version({
            "version": "v1.0", "file_path": "v1.joblib",
            "trained_at": "2026-02-01",
        })
        self.db.insert_model_version({
            "version": "v2.0", "file_path": "v2.joblib",
            "trained_at": "2026-03-01",
        })
        result = self.db.get_all_model_versions()
        assert len(result) == 2
        assert result[0]["version"] == "v1.0"
        assert result[1]["version"] == "v2.0"

    def test_insert_version_defaults(self):
        """Les valeurs par defaut fonctionnent correctement."""
        self.db.insert_model_version({
            "version": "v1.0", "file_path": "v1.joblib",
            "trained_at": "2026-02-01",
        })
        result = self.db.get_all_model_versions()
        assert result[0]["training_signals"] == 0
        assert result[0]["accuracy"] is None
        assert result[0]["is_active"] == 0
        assert result[0]["notes"] is None


class TestTradeAnalysesLLM:
    """Tests CRUD pour la table trade_analyses_llm."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        # Insert a trade to reference
        self.db.insert_trade_complet({
            "isin": "FR0000120578", "nom_action": "SANOFI",
            "date_achat": "2025-07-10", "date_vente": "2025-07-20",
            "prix_achat": 95.0, "prix_vente": 100.0, "quantite": 10,
            "rendement_brut_pct": 5.26, "rendement_net_pct": 5.0,
            "duree_jours": 10, "frais_totaux": 2.5, "statut": "CLOTURE",
        })

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_insert_and_get_analysis(self):
        """Insert et recupere une analyse LLM."""
        analysis = {
            "trade_id": 1,
            "primary_news_id": None,
            "catalyst_type": "EARNINGS",
            "catalyst_summary": "Nicolas a achete car resultats T2 solides",
            "catalyst_confidence": 0.85,
            "news_sentiment": 0.6,
            "buy_reason": "Resultats T2 au-dessus des attentes",
            "sell_reason": "Objectif de +5% atteint",
            "trade_quality": "BON",
            "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        }
        self.db.insert_trade_analysis(analysis)
        result = self.db.get_trade_analysis(1)
        assert result is not None
        assert result["catalyst_type"] == "EARNINGS"
        assert result["catalyst_confidence"] == 0.85
        assert result["buy_reason"] == "Resultats T2 au-dessus des attentes"

    def test_get_analysis_missing(self):
        """Retourne None si le trade n'a pas d'analyse."""
        result = self.db.get_trade_analysis(999)
        assert result is None

    def test_upsert_analysis(self):
        """Insert ou replace une analyse existante."""
        analysis1 = {
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "UNKNOWN", "catalyst_summary": "test",
            "catalyst_confidence": 0.5, "news_sentiment": 0.0,
            "buy_reason": "v1", "sell_reason": "v1",
            "trade_quality": "MOYEN", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        }
        self.db.insert_trade_analysis(analysis1)
        # Update with new data
        analysis2 = {**analysis1, "catalyst_type": "EARNINGS",
                      "catalyst_confidence": 0.9, "buy_reason": "v2"}
        self.db.insert_trade_analysis(analysis2)
        result = self.db.get_trade_analysis(1)
        assert result["catalyst_type"] == "EARNINGS"
        assert result["buy_reason"] == "v2"

    def test_count_analyses(self):
        """Compte les analyses."""
        assert self.db.count_trade_analyses() == 0
        self.db.insert_trade_analysis({
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "EARNINGS", "catalyst_summary": "test",
            "catalyst_confidence": 0.5, "news_sentiment": 0.0,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        })
        assert self.db.count_trade_analyses() == 1

    def test_get_all_analyses(self):
        """Recupere toutes les analyses."""
        self.db.insert_trade_analysis({
            "trade_id": 1, "primary_news_id": None,
            "catalyst_type": "EARNINGS", "catalyst_summary": "test",
            "catalyst_confidence": 0.85, "news_sentiment": 0.6,
            "buy_reason": "x", "sell_reason": "x",
            "trade_quality": "BON", "model_used": "gpt-4o-mini",
            "analyzed_at": "2026-02-24 19:00:00",
        })
        results = self.db.get_all_trade_analyses()
        assert len(results) == 1
        assert results[0]["trade_id"] == 1
