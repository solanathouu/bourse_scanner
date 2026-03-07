"""Tests pour ModelRetrainer."""

import os
import tempfile

import pytest

from src.core.database import Database
from src.feedback.model_retrainer import ModelRetrainer


class TestShouldRetrain:
    """Tests de la logique should_retrain."""

    def setup_method(self):
        """Cree une base temporaire avec tables initialisees."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()

    def test_not_enough_reviews_returns_false(self):
        """Pas assez de reviews -> False."""
        retrainer = ModelRetrainer(self.db, min_reviews_for_retrain=50)
        assert retrainer.should_retrain() is False

    def test_enough_reviews_returns_true(self):
        """50 reviews accumulees sans modele actif -> True."""
        # Insert 50 signals + 50 reviews
        for i in range(50):
            self.db.insert_signal({
                "ticker": f"TEST{i}.PA",
                "date": f"2026-01-{(i % 28) + 1:02d}",
                "score": 0.8,
                "signal_price": 10.0,
            })
            self.db.insert_signal_review({
                "signal_id": i + 1,
                "ticker": f"TEST{i}.PA",
                "signal_date": f"2026-01-{(i % 28) + 1:02d}",
                "signal_price": 10.0,
                "review_date": f"2026-02-{(i % 28) + 1:02d}",
                "review_price": 10.5,
                "performance_pct": 5.0,
                "outcome": "WIN",
                "failure_reason": None,
                "catalyst_type": "EARNINGS",
                "features_json": None,
                "reviewed_at": "2026-02-15 10:00:00",
            })

        retrainer = ModelRetrainer(self.db, min_reviews_for_retrain=50)
        assert retrainer.should_retrain() is True

    def test_reviews_below_threshold_returns_false(self):
        """10 reviews, seuil 50 -> False."""
        for i in range(10):
            self.db.insert_signal({
                "ticker": f"TEST{i}.PA",
                "date": f"2026-01-{(i % 28) + 1:02d}",
                "score": 0.8,
                "signal_price": 10.0,
            })
            self.db.insert_signal_review({
                "signal_id": i + 1,
                "ticker": f"TEST{i}.PA",
                "signal_date": f"2026-01-{(i % 28) + 1:02d}",
                "signal_price": 10.0,
                "review_date": f"2026-02-{(i % 28) + 1:02d}",
                "review_price": 10.5,
                "performance_pct": 5.0,
                "outcome": "WIN",
                "failure_reason": None,
                "catalyst_type": "EARNINGS",
                "features_json": None,
                "reviewed_at": "2026-02-15 10:00:00",
            })

        retrainer = ModelRetrainer(self.db, min_reviews_for_retrain=50)
        assert retrainer.should_retrain() is False

    def test_with_active_model_subtracts_training_signals(self):
        """Modele actif avec training_signals=40, 50 reviews -> only 10 new -> False."""
        for i in range(50):
            self.db.insert_signal({
                "ticker": f"TEST{i}.PA",
                "date": f"2026-01-{(i % 28) + 1:02d}",
                "score": 0.8,
                "signal_price": 10.0,
            })
            self.db.insert_signal_review({
                "signal_id": i + 1,
                "ticker": f"TEST{i}.PA",
                "signal_date": f"2026-01-{(i % 28) + 1:02d}",
                "signal_price": 10.0,
                "review_date": f"2026-02-{(i % 28) + 1:02d}",
                "review_price": 10.5,
                "performance_pct": 5.0,
                "outcome": "WIN",
                "failure_reason": None,
                "catalyst_type": "EARNINGS",
                "features_json": None,
                "reviewed_at": "2026-02-15 10:00:00",
            })

        # Insert active model with training_signals=40
        self.db.insert_model_version({
            "version": "v1",
            "file_path": "data/models/nicolas_v1.joblib",
            "trained_at": "2026-01-01 00:00:00",
            "training_signals": 40,
            "is_active": 1,
        })

        retrainer = ModelRetrainer(self.db, min_reviews_for_retrain=50)
        # 50 total - 40 training_signals = 10 new reviews < 50 threshold
        assert retrainer.should_retrain() is False


class TestBackupModel:
    """Tests de la sauvegarde de backup."""

    def setup_method(self):
        """Cree un fichier modele factice."""
        self.tmpdir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.tmpdir, "nicolas_v1.joblib")
        with open(self.model_path, "w") as f:
            f.write("fake model data")
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.retrainer = ModelRetrainer(self.db)

    def test_backup_creates_copy(self):
        """Le backup cree un fichier avec 'backup' dans le nom."""
        backup_path = self.retrainer._backup_model(self.model_path)
        assert os.path.exists(backup_path)
        assert "backup" in os.path.basename(backup_path)

    def test_original_still_exists(self):
        """Le fichier original existe toujours apres le backup."""
        self.retrainer._backup_model(self.model_path)
        assert os.path.exists(self.model_path)

    def test_backup_path_ends_with_joblib(self):
        """Le chemin du backup se termine par .joblib."""
        backup_path = self.retrainer._backup_model(self.model_path)
        assert backup_path.endswith(".joblib")

    def test_backup_file_not_found(self):
        """FileNotFoundError si le modele n'existe pas."""
        with pytest.raises(FileNotFoundError, match="Modele non trouve"):
            self.retrainer._backup_model(
                os.path.join(self.tmpdir, "inexistant.joblib")
            )

    def test_backup_content_matches_original(self):
        """Le contenu du backup est identique a l'original."""
        backup_path = self.retrainer._backup_model(self.model_path)
        with open(self.model_path) as f:
            original = f.read()
        with open(backup_path) as f:
            backup = f.read()
        assert original == backup


class TestCompareModels:
    """Tests de _is_new_model_better (sans BDD)."""

    def setup_method(self):
        """Cree une instance sans passer par __init__."""
        self.retrainer = ModelRetrainer.__new__(ModelRetrainer)

    def test_new_model_better(self):
        """F1 superieur de >= 0.01 -> True."""
        old = {"f1": 0.70, "accuracy": 0.80, "precision": 0.75, "recall": 0.65}
        new = {"f1": 0.72, "accuracy": 0.82, "precision": 0.77, "recall": 0.67}
        assert self.retrainer._is_new_model_better(old, new) is True

    def test_new_model_worse(self):
        """F1 inferieur -> False."""
        old = {"f1": 0.75, "accuracy": 0.80, "precision": 0.75, "recall": 0.75}
        new = {"f1": 0.70, "accuracy": 0.78, "precision": 0.72, "recall": 0.68}
        assert self.retrainer._is_new_model_better(old, new) is False

    def test_marginal_improvement_not_enough(self):
        """F1 diff < 0.01 -> False (pas assez d'amelioration)."""
        old = {"f1": 0.750}
        new = {"f1": 0.755}
        assert self.retrainer._is_new_model_better(old, new) is False

    def test_exact_threshold(self):
        """F1 diff == 0.01 -> True (seuil exact)."""
        old = {"f1": 0.70}
        new = {"f1": 0.71}
        assert self.retrainer._is_new_model_better(old, new) is True

    def test_missing_f1_defaults_zero(self):
        """Metriques manquantes traitees comme 0."""
        old = {}
        new = {"f1": 0.05}
        assert self.retrainer._is_new_model_better(old, new) is True


class TestNextVersion:
    """Tests de la logique de versioning."""

    def setup_method(self):
        """Cree une base temporaire."""
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = Database(self.db_path)
        self.db.init_db()
        self.retrainer = ModelRetrainer(self.db)

    def test_no_versions_returns_v2(self):
        """Aucune version en BDD -> v2."""
        assert self.retrainer._next_version() == "v2"

    def test_v1_exists_returns_v2(self):
        """v1 existe -> v2."""
        self.db.insert_model_version({
            "version": "v1",
            "file_path": "data/models/nicolas_v1.joblib",
            "trained_at": "2026-01-01 00:00:00",
        })
        assert self.retrainer._next_version() == "v2"

    def test_v1_and_v2_exist_returns_v3(self):
        """v1 et v2 existent -> v3."""
        self.db.insert_model_version({
            "version": "v1",
            "file_path": "data/models/nicolas_v1.joblib",
            "trained_at": "2026-01-01 00:00:00",
        })
        self.db.insert_model_version({
            "version": "v2",
            "file_path": "data/models/nicolas_v2.joblib",
            "trained_at": "2026-02-01 00:00:00",
        })
        assert self.retrainer._next_version() == "v3"

    def test_non_sequential_versions(self):
        """v1 et v5 existent -> v6."""
        self.db.insert_model_version({
            "version": "v1",
            "file_path": "data/models/nicolas_v1.joblib",
            "trained_at": "2026-01-01 00:00:00",
        })
        self.db.insert_model_version({
            "version": "v5",
            "file_path": "data/models/nicolas_v5.joblib",
            "trained_at": "2026-02-01 00:00:00",
        })
        assert self.retrainer._next_version() == "v6"


class TestFormatReport:
    """Tests du formatage du rapport."""

    def setup_method(self):
        """Cree une instance sans BDD."""
        self.retrainer = ModelRetrainer.__new__(ModelRetrainer)

    def test_deployed_report_contains_version(self):
        """Rapport deploye contient 'deploye' et la version."""
        result = {
            "deployed": True,
            "version": "v3",
            "old_metrics": {"accuracy": 0.80, "precision": 0.75, "recall": 0.70, "f1": 0.72},
            "new_metrics": {"accuracy": 0.85, "precision": 0.80, "recall": 0.75, "f1": 0.77},
            "backup_path": "/tmp/backup.joblib",
        }
        report = self.retrainer.format_retrain_report(result)
        assert "deploye" in report.lower()
        assert "v3" in report

    def test_not_deployed_report_contains_conserve(self):
        """Rapport non deploye contient 'conserve'."""
        result = {
            "deployed": False,
            "old_metrics": {"accuracy": 0.80, "precision": 0.75, "recall": 0.70, "f1": 0.72},
            "new_metrics": {"accuracy": 0.79, "precision": 0.74, "recall": 0.69, "f1": 0.71},
            "backup_path": "/tmp/backup.joblib",
        }
        report = self.retrainer.format_retrain_report(result)
        assert "conserve" in report.lower()

    def test_report_contains_metrics(self):
        """Le rapport contient les metriques comparees."""
        result = {
            "deployed": True,
            "version": "v2",
            "old_metrics": {"accuracy": 0.80, "precision": 0.75, "recall": 0.70, "f1": 0.72},
            "new_metrics": {"accuracy": 0.85, "precision": 0.80, "recall": 0.75, "f1": 0.77},
            "backup_path": "/tmp/backup.joblib",
        }
        report = self.retrainer.format_retrain_report(result)
        assert "Accuracy" in report
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report

    def test_report_contains_backup_path(self):
        """Le rapport contient le chemin du backup."""
        result = {
            "deployed": False,
            "old_metrics": {"accuracy": 0.80, "precision": 0.75, "recall": 0.70, "f1": 0.72},
            "new_metrics": {"accuracy": 0.79, "precision": 0.74, "recall": 0.69, "f1": 0.71},
            "backup_path": "/tmp/my_backup.joblib",
        }
        report = self.retrainer.format_retrain_report(result)
        assert "/tmp/my_backup.joblib" in report
