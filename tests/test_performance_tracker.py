"""Tests pour PerformanceTracker — seuil adaptatif, regles de filtrage, summaries."""

import json
import tempfile
import os
import pytest

from src.core.database import Database
from src.feedback.performance_tracker import PerformanceTracker


def _make_db():
    """Cree une base temporaire initialisee."""
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test_tracker.db")
    db = Database(db_path)
    db.init_db()
    return db


def _seed_reviews(db, reviews_data):
    """Helper: insere des signaux + reviews de test.

    reviews_data = list of (ticker, outcome, perf, cat_type, score)
    """
    for i, (ticker, outcome, perf, cat_type, score) in enumerate(reviews_data):
        date = f"2026-03-{(i % 28) + 1:02d}"
        db.insert_signal({
            "ticker": ticker,
            "date": date,
            "score": score,
            "signal_price": 10.0,
            "sent_at": f"{date} 10:00:00",
        })
        signals = db.get_signals(ticker)
        signal = [s for s in signals if s["date"] == date][0]
        db.insert_signal_review({
            "signal_id": signal["id"],
            "ticker": ticker,
            "signal_date": date,
            "signal_price": 10.0,
            "review_date": f"2026-03-{(i % 28) + 4:02d}",
            "review_price": 10.0 * (1 + perf / 100),
            "performance_pct": perf,
            "outcome": outcome,
            "failure_reason": None,
            "catalyst_type": cat_type,
            "features_json": None,
            "reviewed_at": f"2026-03-{(i % 28) + 4:02d} 18:00:00",
        })


class TestWinRateByCatalyst:
    """Tests du calcul de win rate par type de catalyseur."""

    def setup_method(self):
        self.db = _make_db()
        self.tracker = PerformanceTracker(self.db)

    def test_empty_reviews(self):
        """Aucune review retourne dict vide."""
        result = self.tracker.win_rate_by_catalyst()
        assert result == {}

    def test_single_catalyst_type(self):
        """Un seul type de catalyseur."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "WIN", 3.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
        ])
        result = self.tracker.win_rate_by_catalyst()
        assert "EARNINGS" in result
        assert result["EARNINGS"]["wins"] == 2
        assert result["EARNINGS"]["total"] == 3
        assert abs(result["EARNINGS"]["win_rate"] - 2 / 3) < 0.01

    def test_multiple_catalyst_types(self):
        """Plusieurs types de catalyseurs."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
            ("AI.PA", "WIN", 3.0, "EARNINGS", 0.78),
            ("OR.PA", "WIN", 4.0, "UPGRADE", 0.82),
            ("SAN.PA", "LOSS", -1.0, "CONTRAT", 0.76),
        ])
        result = self.tracker.win_rate_by_catalyst()
        assert result["EARNINGS"]["win_rate"] == 1.0
        assert result["UPGRADE"]["win_rate"] == 0.5
        assert result["CONTRAT"]["win_rate"] == 0.0

    def test_unknown_catalyst(self):
        """Catalyseur None est classe comme UNKNOWN."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, None, 0.85),
        ])
        result = self.tracker.win_rate_by_catalyst()
        assert "UNKNOWN" in result
        assert result["UNKNOWN"]["wins"] == 1


class TestAdaptiveThreshold:
    """Tests du calcul de seuil adaptatif."""

    def setup_method(self):
        self.db = _make_db()
        self.tracker = PerformanceTracker(self.db, base_threshold=0.75, min_samples=5)

    def test_not_enough_samples_returns_base(self):
        """Moins de min_samples retourne le seuil de base."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "EARNINGS", 0.80),
        ])
        assert self.tracker.compute_adaptive_threshold() == 0.75

    def test_good_win_rate_no_change(self):
        """Win rate >= 60% ne change pas le seuil."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "WIN", 3.0, "EARNINGS", 0.80),
            ("AI.PA", "WIN", 4.0, "EARNINGS", 0.78),
            ("OR.PA", "WIN", 2.0, "UPGRADE", 0.82),
            ("SAN.PA", "LOSS", -1.0, "CONTRAT", 0.76),
        ])
        # 4/5 = 80% win rate
        assert self.tracker.compute_adaptive_threshold() == 0.75

    def test_medium_win_rate_small_increase(self):
        """Win rate 30-40% augmente le seuil de 0.01."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "WIN", 3.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
            ("OR.PA", "LOSS", -1.0, "UPGRADE", 0.82),
            ("SAN.PA", "LOSS", -3.0, "CONTRAT", 0.76),
        ])
        # 2/5 = 40% win rate -> >= 40% -> unchanged
        assert self.tracker.compute_adaptive_threshold() == 0.75

    def test_low_win_rate_increase(self):
        """Win rate < 30% augmente le seuil de 0.02."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
            ("OR.PA", "LOSS", -1.0, "UPGRADE", 0.82),
            ("SAN.PA", "LOSS", -3.0, "CONTRAT", 0.76),
        ])
        # 1/5 = 20% win rate -> +0.02
        assert self.tracker.compute_adaptive_threshold() == 0.77

    def test_threshold_capped_at_base_plus_010(self):
        """Le seuil ne depasse jamais base + 0.10."""
        tracker = PerformanceTracker(self.db, base_threshold=0.75, min_samples=5)
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
            ("OR.PA", "LOSS", -1.0, "UPGRADE", 0.82),
            ("SAN.PA", "LOSS", -3.0, "CONTRAT", 0.76),
        ])
        # 0/5 = 0% -> 0.75 + 0.02 = 0.77, cap at 0.85, so 0.77
        assert tracker.compute_adaptive_threshold() == 0.77

    def test_35_percent_win_rate_threshold(self):
        """Win rate 30-40% augmente le seuil de 0.01."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "WIN", 3.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
            ("OR.PA", "LOSS", -1.0, "UPGRADE", 0.82),
            ("SAN.PA", "LOSS", -3.0, "CONTRAT", 0.76),
            ("RNO.PA", "LOSS", -2.0, "CONTRAT", 0.76),
        ])
        # 2/6 = 33% -> < 40%, >= 30% -> +0.01
        assert self.tracker.compute_adaptive_threshold() == 0.76


class TestFilterRuleGeneration:
    """Tests de la generation de stats catalyseur."""

    def setup_method(self):
        self.db = _make_db()
        self.tracker = PerformanceTracker(self.db, min_samples=3)

    def test_generates_stats_for_catalyst(self):
        """Genere CATALYST_STATS pour chaque type avec assez de samples."""
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "RUMOR", 0.85),
            ("BN.PA", "LOSS", -2.0, "RUMOR", 0.80),
            ("AI.PA", "LOSS", -2.0, "RUMOR", 0.78),
            ("OR.PA", "WIN", 4.0, "EARNINGS", 0.82),
            ("SAN.PA", "WIN", 3.0, "EARNINGS", 0.76),
            ("RNO.PA", "WIN", 2.0, "EARNINGS", 0.76),
        ])
        rules = self.tracker.generate_filter_rules()
        stats_rules = [r for r in rules if r["rule_type"] == "CATALYST_STATS"]
        assert len(stats_rules) == 2
        types = {json.loads(r["rule_json"])["catalyst_type"] for r in stats_rules}
        assert "RUMOR" in types
        assert "EARNINGS" in types

    def test_no_stats_if_not_enough_samples(self):
        """Pas de stats si moins de min_samples."""
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "RUMOR", 0.85),
            ("BN.PA", "LOSS", -2.0, "RUMOR", 0.80),
        ])
        rules = self.tracker.generate_filter_rules()
        stats_rules = [r for r in rules if r["rule_type"] == "CATALYST_STATS"]
        assert len(stats_rules) == 0

    def test_never_generates_exclude_rules(self):
        """Ne genere JAMAIS de regles EXCLUDE_CATALYST."""
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "RUMOR", 0.85),
            ("BN.PA", "LOSS", -2.0, "RUMOR", 0.80),
            ("AI.PA", "LOSS", -2.0, "RUMOR", 0.78),
        ])
        rules = self.tracker.generate_filter_rules()
        exclude_rules = [r for r in rules if r["rule_type"] == "EXCLUDE_CATALYST"]
        assert len(exclude_rules) == 0

    def test_deactivates_old_rules(self):
        """Les anciennes regles sont desactivees avant d'en creer de nouvelles."""
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "RUMOR", 0.85),
            ("BN.PA", "LOSS", -2.0, "RUMOR", 0.80),
            ("AI.PA", "LOSS", -2.0, "RUMOR", 0.78),
        ])
        self.tracker.generate_filter_rules()
        old_rules = self.db.get_active_filter_rules()
        assert len(old_rules) > 0

        # Generate again
        self.tracker.generate_filter_rules()
        all_rules = self.db.get_active_filter_rules()
        # Only new batch active
        assert len(all_rules) == 1
        assert all_rules[0]["rule_type"] == "CATALYST_STATS"

    def test_stats_contain_wins_and_total(self):
        """Les stats contiennent wins et total."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
        ])
        rules = self.tracker.generate_filter_rules()
        data = json.loads(rules[0]["rule_json"])
        assert data["catalyst_type"] == "EARNINGS"
        assert data["wins"] == 1
        assert data["total"] == 3

    def test_rules_persisted_in_db(self):
        """Les regles generees sont bien en base."""
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "RUMOR", 0.85),
            ("BN.PA", "LOSS", -2.0, "RUMOR", 0.80),
            ("AI.PA", "LOSS", -2.0, "RUMOR", 0.78),
        ])
        self.tracker.generate_filter_rules()
        db_rules = self.db.get_active_filter_rules()
        rule_types = [r["rule_type"] for r in db_rules]
        assert "CATALYST_STATS" in rule_types
        assert "EXCLUDE_CATALYST" not in rule_types


class TestGetCatalystStats:
    """Tests de la recuperation des stats catalyseur."""

    def setup_method(self):
        self.db = _make_db()
        self.tracker = PerformanceTracker(self.db, min_samples=3)

    def test_empty_stats(self):
        """Sans regles, retourne dict vide."""
        assert self.tracker.get_catalyst_stats() == {}

    def test_returns_stats_from_rules(self):
        """Retourne les stats depuis les regles CATALYST_STATS."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "EARNINGS", 0.80),
            ("AI.PA", "LOSS", -2.0, "EARNINGS", 0.78),
        ])
        self.tracker.generate_filter_rules()
        stats = self.tracker.get_catalyst_stats()
        assert "EARNINGS" in stats
        assert stats["EARNINGS"]["wins"] == 1
        assert stats["EARNINGS"]["total"] == 3


class TestDailySummary:
    """Tests du resume quotidien."""

    def setup_method(self):
        self.db = _make_db()
        self.tracker = PerformanceTracker(self.db)

    def test_empty_reviews(self):
        """Aucune review affiche message vide."""
        result = self.tracker.get_daily_summary([])
        assert "Aucun signal" in result

    def test_contains_ticker_names(self):
        """Le summary contient les noms des tickers."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
        ])
        reviews = self.db.get_signal_reviews()
        result = self.tracker.get_daily_summary(reviews)
        assert "MC.PA" in result
        assert "BN.PA" in result

    def test_contains_outcomes(self):
        """Le summary contient les outcomes."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
        ])
        reviews = self.db.get_signal_reviews()
        result = self.tracker.get_daily_summary(reviews)
        assert "OK" in result  # WIN icon
        assert "X" in result   # LOSS icon

    def test_contains_win_rate(self):
        """Le summary contient le win rate global."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
        ])
        reviews = self.db.get_signal_reviews()
        result = self.tracker.get_daily_summary(reviews)
        assert "Win rate global" in result

    def test_contains_threshold(self):
        """Le summary contient le seuil adaptatif."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
        ])
        reviews = self.db.get_signal_reviews()
        result = self.tracker.get_daily_summary(reviews)
        assert "Seuil actuel" in result

    def test_neutral_outcome_display(self):
        """NEUTRAL affiche un tiret."""
        _seed_reviews(self.db, [
            ("MC.PA", "NEUTRAL", 0.5, "EARNINGS", 0.85),
        ])
        reviews = self.db.get_signal_reviews()
        result = self.tracker.get_daily_summary(reviews)
        assert "[-]" in result


class TestWeeklySummary:
    """Tests du bilan hebdomadaire."""

    def setup_method(self):
        self.db = _make_db()
        self.tracker = PerformanceTracker(self.db)

    def test_empty_week(self):
        """Semaine sans review affiche message vide."""
        result = self.tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "Aucune review" in result

    def test_contains_signal_count(self):
        """Le bilan contient le nombre de signaux."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
            ("AI.PA", "NEUTRAL", 0.5, "CONTRAT", 0.78),
        ])
        result = self.tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "Signaux: 3" in result

    def test_contains_win_loss_neutral_counts(self):
        """Le bilan contient WIN, LOSS, NEUTRAL decomposition."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
            ("AI.PA", "NEUTRAL", 0.5, "CONTRAT", 0.78),
        ])
        result = self.tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "WIN: 1" in result
        assert "LOSS: 1" in result
        assert "NEUTRAL: 1" in result

    def test_contains_perf_metrics(self):
        """Le bilan contient les metriques de performance."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("BN.PA", "LOSS", -2.0, "UPGRADE", 0.80),
        ])
        result = self.tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "Win rate" in result
        assert "Perf moyenne" in result
        assert "Meilleur" in result
        assert "Pire" in result

    def test_contains_adaptive_threshold(self):
        """Le bilan contient le seuil adaptatif."""
        _seed_reviews(self.db, [
            ("MC.PA", "WIN", 5.0, "EARNINGS", 0.85),
        ])
        result = self.tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "Seuil adaptatif" in result

    def test_contains_date_range(self):
        """Le bilan contient la plage de dates."""
        result = self.tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "2026-03-01" in result
        assert "2026-03-07" in result

    def test_shows_catalyst_stats(self):
        """Le bilan affiche les stats par catalyseur."""
        _seed_reviews(self.db, [
            ("MC.PA", "LOSS", -5.0, "RUMOR", 0.85),
            ("BN.PA", "LOSS", -2.0, "RUMOR", 0.80),
            ("AI.PA", "LOSS", -2.0, "RUMOR", 0.78),
            ("OR.PA", "WIN", 4.0, "EARNINGS", 0.82),
            ("SAN.PA", "WIN", 3.0, "EARNINGS", 0.76),
        ])
        tracker = PerformanceTracker(self.db, min_samples=3)
        tracker.generate_filter_rules()
        result = tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "Win rate par catalyseur" in result
        assert "RUMOR" in result
