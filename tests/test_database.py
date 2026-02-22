"""Tests pour la couche base de données SQLite."""

import sqlite3
import os
import tempfile

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
