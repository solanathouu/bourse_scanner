"""Tests pour le module SignalReviewer."""

import json
import os
import tempfile

import pytest

from src.core.database import Database
from src.feedback.signal_reviewer import SignalReviewer


class TestOutcomeClassification:
    """Tests de classification des outcomes."""

    def setup_method(self):
        """Cree un reviewer sans DB (test methodes pures)."""
        self.reviewer = SignalReviewer.__new__(SignalReviewer)
        self.reviewer.win_threshold = 4.5

    def test_win_at_threshold(self):
        """Performance exactement au seuil = WIN."""
        assert self.reviewer._classify_outcome(4.5) == "WIN"

    def test_win_above_threshold(self):
        """Performance au-dessus du seuil = WIN."""
        assert self.reviewer._classify_outcome(10.0) == "WIN"

    def test_loss_negative(self):
        """Performance negative = LOSS."""
        assert self.reviewer._classify_outcome(-2.5) == "LOSS"

    def test_loss_slightly_negative(self):
        """Performance legerement negative = LOSS."""
        assert self.reviewer._classify_outcome(-0.01) == "LOSS"

    def test_neutral_zero(self):
        """Performance a zero = NEUTRAL."""
        assert self.reviewer._classify_outcome(0.0) == "NEUTRAL"

    def test_neutral_between_zero_and_threshold(self):
        """Performance entre 0 et seuil = NEUTRAL."""
        assert self.reviewer._classify_outcome(2.0) == "NEUTRAL"

    def test_neutral_just_below_threshold(self):
        """Performance juste sous le seuil = NEUTRAL."""
        assert self.reviewer._classify_outcome(4.49) == "NEUTRAL"


class TestPerformanceCalculation:
    """Tests du calcul de performance."""

    def setup_method(self):
        """Cree un reviewer sans DB."""
        self.reviewer = SignalReviewer.__new__(SignalReviewer)
        self.reviewer.win_threshold = 4.5

    def test_positive_performance(self):
        """Calcul correct d'une performance positive."""
        perf = self.reviewer._calculate_performance(100.0, 110.0)
        assert perf == 10.0

    def test_negative_performance(self):
        """Calcul correct d'une performance negative."""
        perf = self.reviewer._calculate_performance(100.0, 95.0)
        assert perf == -5.0

    def test_zero_performance(self):
        """Performance nulle quand prix identiques."""
        perf = self.reviewer._calculate_performance(50.0, 50.0)
        assert perf == 0.0

    def test_zero_signal_price(self):
        """Signal price a zero retourne 0.0 (pas de division par zero)."""
        perf = self.reviewer._calculate_performance(0.0, 50.0)
        assert perf == 0.0

    def test_performance_rounding(self):
        """Performance arrondie a 2 decimales."""
        perf = self.reviewer._calculate_performance(33.33, 34.44)
        assert perf == round((34.44 - 33.33) / 33.33 * 100, 2)


class TestFailureAnalysis:
    """Tests de l'analyse des echecs."""

    def setup_method(self):
        """Cree un reviewer sans DB."""
        self.reviewer = SignalReviewer.__new__(SignalReviewer)
        self.reviewer.win_threshold = 4.5

    def test_win_returns_none(self):
        """Pas d'analyse d'echec si performance positive."""
        signal = {"catalyst_type": "EARNINGS"}
        result = self.reviewer._analyze_failure(signal, 5.0)
        assert result is None

    def test_neutral_returns_none(self):
        """Pas d'analyse d'echec si performance nulle."""
        signal = {"catalyst_type": "EARNINGS"}
        result = self.reviewer._analyze_failure(signal, 0.0)
        assert result is None

    def test_loss_generates_reason(self):
        """Performance negative genere une raison d'echec."""
        signal = {"catalyst_type": "EARNINGS"}
        result = self.reviewer._analyze_failure(signal, -3.0)
        assert result is not None
        assert "EARNINGS" in result

    def test_loss_with_high_rsi(self):
        """RSI eleve mentionne dans l'analyse."""
        features = json.dumps({"rsi_14": 72.5})
        signal = {"catalyst_type": "UPGRADE", "features_json": features}
        result = self.reviewer._analyze_failure(signal, -2.0)
        assert "RSI" in result
        assert "surachat" in result

    def test_loss_with_low_volume(self):
        """Volume faible mentionne dans l'analyse."""
        features = json.dumps({"volume_ratio_20": 0.3})
        signal = {"catalyst_type": "CONTRAT", "features_json": features}
        result = self.reviewer._analyze_failure(signal, -1.5)
        assert "Volume faible" in result

    def test_loss_with_low_sentiment(self):
        """Sentiment faible mentionne dans l'analyse."""
        features = json.dumps({"news_sentiment": 0.1})
        signal = {"catalyst_type": "NEWS", "features_json": features}
        result = self.reviewer._analyze_failure(signal, -4.0)
        assert "Sentiment faible" in result

    def test_loss_unknown_catalyst(self):
        """Catalyst type absent utilise UNKNOWN."""
        signal = {}
        result = self.reviewer._analyze_failure(signal, -1.0)
        assert "UNKNOWN" in result

    def test_loss_invalid_features_json(self):
        """features_json invalide ne cause pas de crash."""
        signal = {"catalyst_type": "EARNINGS", "features_json": "not json"}
        result = self.reviewer._analyze_failure(signal, -2.0)
        assert result is not None
        assert "EARNINGS" in result

    def test_loss_multiple_factors(self):
        """Plusieurs facteurs combines avec separateur."""
        features = json.dumps({
            "rsi_14": 75.0,
            "volume_ratio_20": 0.2,
            "news_sentiment": 0.05,
        })
        signal = {"catalyst_type": "EARNINGS", "features_json": features}
        result = self.reviewer._analyze_failure(signal, -5.0)
        assert " | " in result
        parts = result.split(" | ")
        assert len(parts) == 4  # catalyst + RSI + volume + sentiment


class TestReviewPending:
    """Tests d'integration avec base de donnees temporaire."""

    def setup_method(self):
        """Cree une BDD temporaire avec donnees de test."""
        self.tmpdir = tempfile.TemporaryDirectory()
        db_path = os.path.join(self.tmpdir.name, "test.db")
        self.db = Database(db_path)
        self.db.init_db()
        self.reviewer = SignalReviewer(self.db, win_threshold=4.5)

    def teardown_method(self):
        """Nettoie la BDD temporaire."""
        self.tmpdir.cleanup()

    def _insert_signal(self, ticker: str, date: str, score: float,
                       signal_price: float = 100.0, sent_at: str = None):
        """Helper: insere un signal en base."""
        if sent_at is None:
            sent_at = f"{date} 10:00:00"
        self.db.insert_signal({
            "ticker": ticker,
            "date": date,
            "score": score,
            "catalyst_type": "EARNINGS",
            "catalyst_news_title": "Test news",
            "features_json": json.dumps({"rsi_14": 45.0}),
            "sent_at": sent_at,
            "signal_price": signal_price,
        })

    def _insert_price(self, ticker: str, date: str, close: float,
                      high: float | None = None):
        """Helper: insere un prix en base. high defaut = close."""
        self.db.insert_price({
            "ticker": ticker,
            "date": date,
            "open": close,
            "high": high if high is not None else close,
            "low": close - 1,
            "close": close,
            "volume": 10000,
        })

    def test_review_signal_with_price_at_j3(self):
        """Signal + prix a J+3 avec TP touche (high=105 >= +4.5%) = WIN."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        self._insert_price("MC.PA", "2026-03-04", close=103.0, high=105.0)

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        review = reviews[0]
        assert review["ticker"] == "MC.PA"
        assert review["signal_price"] == 100.0
        assert review["performance_pct"] == 4.5  # TP touche
        assert review["outcome"] == "WIN"
        assert review["signal_date"] == "2026-03-01"
        assert review["review_date"] == "2026-03-04"

    def test_already_reviewed_not_reviewed_again(self):
        """Signal deja reviewe n'est pas reviewe une 2e fois."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        self._insert_price("MC.PA", "2026-03-04", 105.0)

        reviews1 = self.reviewer.review_pending("2026-03-05")
        assert len(reviews1) == 1

        reviews2 = self.reviewer.review_pending("2026-03-06")
        assert len(reviews2) == 0

    def test_no_price_at_j3_skipped(self):
        """Signal sans prix autour de J+3 est skippe."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        # Pas de prix insere pour J+2 a J+5

        reviews = self.reviewer.review_pending("2026-03-05")
        assert len(reviews) == 0

    def test_signal_without_signal_price_skipped(self):
        """Signal sans signal_price est skippe."""
        self.db.insert_signal({
            "ticker": "MC.PA",
            "date": "2026-03-01",
            "score": 0.85,
            "sent_at": "2026-03-01 10:00:00",
            "signal_price": None,
        })
        self._insert_price("MC.PA", "2026-03-04", 105.0)

        reviews = self.reviewer.review_pending("2026-03-05")
        assert len(reviews) == 0

    def test_review_loss_signal(self):
        """Signal en perte = outcome LOSS avec failure_reason."""
        self._insert_signal("AI.PA", "2026-03-01", 0.80, signal_price=50.0)
        self._insert_price("AI.PA", "2026-03-04", 47.0)  # -6%

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        review = reviews[0]
        assert review["outcome"] == "LOSS"
        assert review["performance_pct"] == -6.0
        assert review["failure_reason"] is not None
        assert "EARNINGS" in review["failure_reason"]

    def test_review_neutral_signal(self):
        """Signal avec petite hausse = NEUTRAL."""
        self._insert_signal("SAN.PA", "2026-03-01", 0.78, signal_price=80.0)
        self._insert_price("SAN.PA", "2026-03-04", 82.0)  # +2.5%

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        assert reviews[0]["outcome"] == "NEUTRAL"
        assert reviews[0]["failure_reason"] is None

    def test_review_price_in_weekend_window(self):
        """Prix au J+4 (lundi) pris en compte si J+3 tombe un weekend."""
        # Signal le vendredi, J+3 = lundi -> prix au J+4 aussi acceptable
        self._insert_signal("MC.PA", "2026-03-06", 0.85, signal_price=100.0)
        # Pas de prix au J+3 (dimanche 2026-03-09), mais prix au J+4 (lundi)
        self._insert_price("MC.PA", "2026-03-10", 104.0)  # J+4

        reviews = self.reviewer.review_pending("2026-03-10")

        assert len(reviews) == 1
        assert reviews[0]["review_price"] == 104.0

    def test_review_multiple_signals(self):
        """Plusieurs signaux en attente reviews en un seul appel."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        self._insert_signal("AI.PA", "2026-03-01", 0.80, signal_price=50.0)
        self._insert_price("MC.PA", "2026-03-04", 110.0)
        self._insert_price("AI.PA", "2026-03-04", 48.0)

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 2
        tickers = {r["ticker"] for r in reviews}
        assert tickers == {"MC.PA", "AI.PA"}

    def test_signal_too_recent_not_reviewed(self):
        """Signal emis il y a moins de 3 jours n'est pas reviewe."""
        self._insert_signal("MC.PA", "2026-03-04", 0.85, signal_price=100.0)
        self._insert_price("MC.PA", "2026-03-07", 105.0)

        # current_date = 2026-03-05, signal du 04 -> seulement 1 jour
        reviews = self.reviewer.review_pending("2026-03-05")
        assert len(reviews) == 0

    def test_empty_pending_returns_empty(self):
        """Aucun signal en attente retourne une liste vide."""
        reviews = self.reviewer.review_pending("2026-03-10")
        assert reviews == []

    def test_review_stored_in_database(self):
        """La review est bien persistee dans la BDD."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        # Close a +3% (pas WIN seul) mais high a +5% = TP touche
        self._insert_price("MC.PA", "2026-03-04", close=103.0, high=105.0)

        self.reviewer.review_pending("2026-03-05")

        stored = self.db.get_signal_reviews("MC.PA")
        assert len(stored) == 1
        assert stored[0]["performance_pct"] == 4.5  # TP touche
        assert stored[0]["outcome"] == "WIN"

    def test_tp_hit_at_j1_then_drop_at_j3(self):
        """TP touche a J+1 (high >= +4.5%) puis rechute a J+3 = WIN."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        # J+1: high touche 105 (+5%) mais close a 102
        self._insert_price("MC.PA", "2026-03-02", close=102.0, high=105.0)
        # J+2: redescend
        self._insert_price("MC.PA", "2026-03-03", close=99.0, high=100.0)
        # J+3: en dessous du signal -> serait LOSS sans TP
        self._insert_price("MC.PA", "2026-03-04", close=97.0, high=98.0)

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        assert reviews[0]["outcome"] == "WIN"
        assert reviews[0]["performance_pct"] == 4.5  # perf = seuil TP

    def test_tp_not_hit_uses_close_j3(self):
        """TP pas touche -> utilise le close a J+3 normalement."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        # Highs ne touchent jamais +4.5%
        self._insert_price("MC.PA", "2026-03-02", close=101.0, high=103.0)
        self._insert_price("MC.PA", "2026-03-03", close=100.5, high=102.0)
        self._insert_price("MC.PA", "2026-03-04", close=102.0, high=103.5)

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        assert reviews[0]["outcome"] == "NEUTRAL"  # +2% < 4.5%
        assert reviews[0]["performance_pct"] == 2.0

    def test_tp_hit_same_day(self):
        """TP touche le jour meme du signal = WIN."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        # J+0: high a 106 (+6%)
        self._insert_price("MC.PA", "2026-03-01", close=103.0, high=106.0)
        self._insert_price("MC.PA", "2026-03-04", close=95.0, high=96.0)

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        assert reviews[0]["outcome"] == "WIN"
        assert reviews[0]["performance_pct"] == 4.5

    def test_tp_hit_exactly_at_threshold(self):
        """High exactement a +4.5% = WIN."""
        self._insert_signal("MC.PA", "2026-03-01", 0.85, signal_price=100.0)
        self._insert_price("MC.PA", "2026-03-02", close=101.0, high=104.5)
        self._insert_price("MC.PA", "2026-03-04", close=99.0, high=100.0)

        reviews = self.reviewer.review_pending("2026-03-05")

        assert len(reviews) == 1
        assert reviews[0]["outcome"] == "WIN"
