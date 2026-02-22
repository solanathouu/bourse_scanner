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
