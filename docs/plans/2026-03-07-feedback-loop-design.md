# Feedback Loop Auto-Apprenant — Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Systeme de feedback loop qui review les signaux a J+3, apprend des erreurs, ajuste le seuil et les filtres, et re-entraine le modele quand assez de donnees.

**Architecture:** Nouveau module `src/feedback/` (niveau 4) avec 3 fichiers : SignalReviewer (review J+3), PerformanceTracker (stats + regles adaptatives), ModelRetrainer (retrain avec backup/validation). Le script `run_scanner.py` orchestre les nouveaux jobs APScheduler. Le SignalFilter existant charge les regles dynamiques.

**Tech Stack:** Python 3.13, SQLite, XGBoost, yfinance, python-telegram-bot, APScheduler, loguru

---

## Task 1: Migration BDD — nouvelles tables + colonne signal_price

**Files:**
- Modify: `src/core/database.py`
- Test: `tests/test_database.py`

### Step 1: Write failing tests for new tables and methods

Add to `tests/test_database.py`:

```python
# --- Signal Reviews ---

class TestSignalReviews:
    """Tests pour la table signal_reviews."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_insert_signal_review(self):
        # Insert a signal first
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50, "sent_at": "2026-03-01 10:00:00",
        })
        signals = self.db.get_signals("SAN.PA")
        signal_id = signals[0]["id"]

        review = {
            "signal_id": signal_id, "ticker": "SAN.PA",
            "signal_date": "2026-03-01", "signal_price": 95.50,
            "review_date": "2026-03-04", "review_price": 100.30,
            "performance_pct": 5.03, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": "EARNINGS",
            "features_json": '{"rsi_14": 35.0}', "reviewed_at": "2026-03-04 18:00:00",
        }
        self.db.insert_signal_review(review)
        reviews = self.db.get_signal_reviews()
        assert len(reviews) == 1
        assert reviews[0]["outcome"] == "WIN"
        assert reviews[0]["performance_pct"] == 5.03

    def test_get_pending_reviews(self):
        """Signaux envoyes il y a 3+ jours sans review."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50, "sent_at": "2026-03-01 10:00:00",
        })
        self.db.insert_signal({
            "ticker": "DBV.PA", "date": "2026-03-04",
            "score": 0.80, "signal_price": 3.20, "sent_at": "2026-03-04 10:00:00",
        })
        # Only SAN.PA should be pending (3+ days ago relative to 2026-03-04)
        pending = self.db.get_pending_signal_reviews("2026-03-04")
        assert len(pending) == 1
        assert pending[0]["ticker"] == "SAN.PA"

    def test_get_reviews_in_period(self):
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50, "sent_at": "2026-03-01 10:00:00",
        })
        signals = self.db.get_signals("SAN.PA")
        self.db.insert_signal_review({
            "signal_id": signals[0]["id"], "ticker": "SAN.PA",
            "signal_date": "2026-03-01", "signal_price": 95.50,
            "review_date": "2026-03-04", "review_price": 100.30,
            "performance_pct": 5.03, "outcome": "WIN",
            "failure_reason": None, "catalyst_type": "EARNINGS",
            "features_json": None, "reviewed_at": "2026-03-04 18:00:00",
        })
        reviews = self.db.get_reviews_in_period("2026-03-01", "2026-03-07")
        assert len(reviews) == 1

    def test_signal_price_stored(self):
        """Verifie que signal_price est stocke dans signals."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50, "sent_at": "2026-03-01 10:00:00",
        })
        signal = self.db.get_latest_signal("SAN.PA")
        assert signal["signal_price"] == 95.50

    def test_get_signal_review_stats(self):
        """Stats globales des reviews."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50, "sent_at": "2026-03-01 10:00:00",
        })
        signals = self.db.get_signals("SAN.PA")
        for outcome, perf in [("WIN", 5.0), ("LOSS", -2.0), ("NEUTRAL", 2.0)]:
            self.db.insert_signal({
                "ticker": f"T{outcome}.PA", "date": "2026-03-01",
                "score": 0.80, "signal_price": 10.0,
                "sent_at": "2026-03-01 10:00:00",
            })
            s = self.db.get_signals(f"T{outcome}.PA")
            self.db.insert_signal_review({
                "signal_id": s[0]["id"], "ticker": f"T{outcome}.PA",
                "signal_date": "2026-03-01", "signal_price": 10.0,
                "review_date": "2026-03-04", "review_price": 10.0 + perf/10,
                "performance_pct": perf, "outcome": outcome,
                "failure_reason": None, "catalyst_type": "EARNINGS",
                "features_json": None, "reviewed_at": "2026-03-04 18:00:00",
            })
        stats = self.db.get_review_stats()
        assert stats["total"] == 3
        assert stats["wins"] == 1
        assert stats["losses"] == 1


class TestFilterRules:
    """Tests pour la table filter_rules."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_insert_and_get_active_rules(self):
        self.db.insert_filter_rule({
            "rule_type": "EXCLUDE_CATALYST",
            "rule_json": '{"catalyst_type": "TECHNICAL", "condition": "volume_ratio_20 < 0.5"}',
            "win_rate": 0.20, "sample_size": 10,
            "created_at": "2026-03-04 18:30:00", "active": 1,
        })
        rules = self.db.get_active_filter_rules()
        assert len(rules) == 1
        assert rules[0]["rule_type"] == "EXCLUDE_CATALYST"

    def test_deactivate_rule(self):
        self.db.insert_filter_rule({
            "rule_type": "EXCLUDE_CATALYST",
            "rule_json": '{"catalyst_type": "TECHNICAL"}',
            "win_rate": 0.20, "sample_size": 10,
            "created_at": "2026-03-04 18:30:00", "active": 1,
        })
        rules = self.db.get_active_filter_rules()
        self.db.deactivate_filter_rule(rules[0]["id"])
        assert len(self.db.get_active_filter_rules()) == 0


class TestModelVersions:
    """Tests pour la table model_versions."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_insert_and_get_active_model(self):
        self.db.insert_model_version({
            "version": "v1", "file_path": "data/models/nicolas_v1.joblib",
            "trained_at": "2026-02-20 10:00:00",
            "training_signals": 0, "accuracy": 0.85,
            "precision_score": 0.90, "recall": 0.80, "f1": 0.85,
            "is_active": 1, "notes": "Initial model",
        })
        active = self.db.get_active_model_version()
        assert active["version"] == "v1"
        assert active["is_active"] == 1

    def test_switch_active_model(self):
        self.db.insert_model_version({
            "version": "v1", "file_path": "data/models/nicolas_v1.joblib",
            "trained_at": "2026-02-20 10:00:00",
            "training_signals": 0, "accuracy": 0.85,
            "precision_score": 0.90, "recall": 0.80, "f1": 0.85,
            "is_active": 1, "notes": "Initial",
        })
        self.db.insert_model_version({
            "version": "v2", "file_path": "data/models/nicolas_v2.joblib",
            "trained_at": "2026-03-07 10:00:00",
            "training_signals": 50, "accuracy": 0.88,
            "precision_score": 0.92, "recall": 0.85, "f1": 0.88,
            "is_active": 0, "notes": "Retrained",
        })
        versions = self.db.get_all_model_versions()
        v2_id = [v for v in versions if v["version"] == "v2"][0]["id"]
        self.db.set_active_model(v2_id)
        active = self.db.get_active_model_version()
        assert active["version"] == "v2"
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_database.py -v -k "SignalReviews or FilterRules or ModelVersions"`
Expected: FAIL (methods not defined)

### Step 3: Implement DB migration and new methods

In `src/core/database.py`:

1. Add `signal_price` column migration in `init_db()` (after signals table creation)
2. Add 3 new tables: `signal_reviews`, `filter_rules`, `model_versions`
3. Add CRUD methods:
   - `insert_signal_review(review)`, `get_signal_reviews(ticker=None)`, `get_pending_signal_reviews(current_date)`, `get_reviews_in_period(start, end)`, `get_review_stats()`
   - `insert_filter_rule(rule)`, `get_active_filter_rules()`, `deactivate_filter_rule(rule_id)`
   - `insert_model_version(version)`, `get_active_model_version()`, `get_all_model_versions()`, `set_active_model(version_id)`

**Migration for signal_price:**
```python
def _migrate_signals_columns(self, conn):
    existing = [row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()]
    if "signal_price" not in existing:
        conn.execute("ALTER TABLE signals ADD COLUMN signal_price REAL")
        logger.info("Migration: colonne 'signal_price' ajoutee a la table signals")
    conn.commit()
```

**signal_reviews table:**
```sql
CREATE TABLE IF NOT EXISTS signal_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL UNIQUE,
    ticker TEXT NOT NULL,
    signal_date TEXT NOT NULL,
    signal_price REAL NOT NULL,
    review_date TEXT NOT NULL,
    review_price REAL NOT NULL,
    performance_pct REAL NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('WIN', 'LOSS', 'NEUTRAL')),
    failure_reason TEXT,
    catalyst_type TEXT,
    features_json TEXT,
    reviewed_at TEXT NOT NULL,
    FOREIGN KEY (signal_id) REFERENCES signals(id)
);
```

**filter_rules table:**
```sql
CREATE TABLE IF NOT EXISTS filter_rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_type TEXT NOT NULL,
    rule_json TEXT NOT NULL,
    win_rate REAL,
    sample_size INTEGER,
    created_at TEXT NOT NULL,
    active INTEGER DEFAULT 1
);
```

**model_versions table:**
```sql
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    training_signals INTEGER DEFAULT 0,
    accuracy REAL,
    precision_score REAL,
    recall REAL,
    f1 REAL,
    is_active INTEGER DEFAULT 0,
    notes TEXT
);
```

**Key methods:**

`get_pending_signal_reviews(current_date)`:
```python
def get_pending_signal_reviews(self, current_date: str) -> list[dict]:
    """Signaux envoyes il y a 3+ jours sans review."""
    conn = self._connect()
    rows = conn.execute("""
        SELECT s.* FROM signals s
        LEFT JOIN signal_reviews sr ON sr.signal_id = s.id
        WHERE sr.id IS NULL
          AND s.sent_at IS NOT NULL
          AND julianday(?) - julianday(s.date) >= 3
        ORDER BY s.date
    """, (current_date,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]
```

`get_review_stats()`:
```python
def get_review_stats(self) -> dict:
    conn = self._connect()
    rows = conn.execute("SELECT outcome, COUNT(*) as cnt FROM signal_reviews GROUP BY outcome").fetchall()
    conn.close()
    stats = {"total": 0, "wins": 0, "losses": 0, "neutrals": 0}
    for row in rows:
        stats["total"] += row["cnt"]
        if row["outcome"] == "WIN": stats["wins"] = row["cnt"]
        elif row["outcome"] == "LOSS": stats["losses"] = row["cnt"]
        elif row["outcome"] == "NEUTRAL": stats["neutrals"] = row["cnt"]
    return stats
```

`set_active_model(version_id)`:
```python
def set_active_model(self, version_id: int):
    conn = self._connect()
    conn.execute("UPDATE model_versions SET is_active = 0")
    conn.execute("UPDATE model_versions SET is_active = 1 WHERE id = ?", (version_id,))
    conn.commit()
    conn.close()
```

Also update `insert_signal` to include `signal_price`:
```python
def insert_signal(self, signal: dict):
    conn = self._connect()
    conn.execute("""
        INSERT OR IGNORE INTO signals
            (ticker, date, score, catalyst_type,
             catalyst_news_title, features_json, sent_at, signal_price)
        VALUES
            (:ticker, :date, :score, :catalyst_type,
             :catalyst_news_title, :features_json, :sent_at, :signal_price)
    """, {
        "catalyst_type": None, "catalyst_news_title": None,
        "features_json": None, "sent_at": None, "signal_price": None,
        **signal,
    })
    conn.commit()
    conn.close()
```

### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_database.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add src/core/database.py tests/test_database.py
git commit -m "feat(db): add signal_reviews, filter_rules, model_versions tables + signal_price migration"
```

---

## Task 2: SignalReviewer — Review des signaux a J+3

**Files:**
- Create: `src/feedback/__init__.py`
- Create: `src/feedback/signal_reviewer.py`
- Create: `tests/test_signal_reviewer.py`

### Step 1: Write failing tests

```python
"""Tests pour SignalReviewer."""

import os
import tempfile
import json

import pytest

from src.core.database import Database
from src.feedback.signal_reviewer import SignalReviewer


class TestOutcomeClassification:
    """Tests de classification WIN/LOSS/NEUTRAL."""

    def test_win_above_threshold(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        reviewer.win_threshold = 4.5
        assert reviewer._classify_outcome(5.0) == "WIN"
        assert reviewer._classify_outcome(4.5) == "WIN"

    def test_loss_negative(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        reviewer.win_threshold = 4.5
        assert reviewer._classify_outcome(-1.0) == "LOSS"
        assert reviewer._classify_outcome(-0.1) == "LOSS"

    def test_neutral_between(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        reviewer.win_threshold = 4.5
        assert reviewer._classify_outcome(0.0) == "NEUTRAL"
        assert reviewer._classify_outcome(3.0) == "NEUTRAL"
        assert reviewer._classify_outcome(4.4) == "NEUTRAL"


class TestPerformanceCalculation:
    """Tests du calcul de performance."""

    def test_positive_performance(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        perf = reviewer._calculate_performance(100.0, 105.0)
        assert perf == 5.0

    def test_negative_performance(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        perf = reviewer._calculate_performance(100.0, 97.0)
        assert perf == -3.0

    def test_zero_signal_price(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        perf = reviewer._calculate_performance(0.0, 105.0)
        assert perf == 0.0


class TestFailureAnalysis:
    """Tests de l'analyse d'echec."""

    def test_loss_generates_reason(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        signal = {
            "ticker": "SAN.PA", "score": 0.82,
            "catalyst_type": "EARNINGS",
            "features_json": json.dumps({
                "rsi_14": 65.0, "volume_ratio_20": 0.4,
                "news_sentiment": 0.1,
            }),
        }
        reason = reviewer._analyze_failure(signal, -3.0)
        assert reason is not None
        assert len(reason) > 0

    def test_win_no_failure_reason(self):
        reviewer = SignalReviewer.__new__(SignalReviewer)
        signal = {"ticker": "SAN.PA", "score": 0.85,
                  "catalyst_type": "EARNINGS", "features_json": "{}"}
        reason = reviewer._analyze_failure(signal, 5.0)
        assert reason is None


class TestReviewPending:
    """Tests integration de la review avec BDD."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_review_with_price_in_db(self):
        """Review quand le prix J+3 est en BDD."""
        # Signal emis le 01/03
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50,
            "catalyst_type": "EARNINGS", "features_json": "{}",
            "sent_at": "2026-03-01 10:00:00",
        })
        # Prix J+3 en BDD
        self.db.insert_price({
            "ticker": "SAN.PA", "date": "2026-03-04",
            "open": 99.0, "high": 101.0, "low": 98.5,
            "close": 100.30, "volume": 1000000,
        })

        reviewer = SignalReviewer(self.db)
        results = reviewer.review_pending("2026-03-04")
        assert len(results) == 1
        assert results[0]["outcome"] == "WIN"
        assert results[0]["performance_pct"] == pytest.approx(5.03, abs=0.1)

    def test_no_pending_if_already_reviewed(self):
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50,
            "sent_at": "2026-03-01 10:00:00",
        })
        self.db.insert_price({
            "ticker": "SAN.PA", "date": "2026-03-04",
            "open": 99.0, "high": 101.0, "low": 98.5,
            "close": 100.30, "volume": 1000000,
        })
        reviewer = SignalReviewer(self.db)
        reviewer.review_pending("2026-03-04")
        # Second call: nothing to review
        results = reviewer.review_pending("2026-03-04")
        assert len(results) == 0

    def test_skip_if_no_price(self):
        """Skip le signal si pas de prix dispo a J+3."""
        self.db.insert_signal({
            "ticker": "SAN.PA", "date": "2026-03-01",
            "score": 0.85, "signal_price": 95.50,
            "sent_at": "2026-03-01 10:00:00",
        })
        reviewer = SignalReviewer(self.db)
        results = reviewer.review_pending("2026-03-04")
        assert len(results) == 0
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_signal_reviewer.py -v`
Expected: FAIL (module not found)

### Step 3: Implement SignalReviewer

**`src/feedback/__init__.py`**: empty file

**`src/feedback/signal_reviewer.py`**:
```python
"""Review des signaux emis a J+3.

Compare le prix au moment du signal avec le prix 3 jours apres
pour evaluer la qualite de chaque call.
"""

import json
from datetime import datetime, timedelta

from loguru import logger

from src.core.database import Database


class SignalReviewer:
    """Review les signaux a J+3 et enregistre les resultats."""

    def __init__(self, db: Database, win_threshold: float = 4.5):
        self.db = db
        self.win_threshold = win_threshold

    def review_pending(self, current_date: str | None = None) -> list[dict]:
        """Review tous les signaux en attente (emis il y a 3+ jours).

        Args:
            current_date: Date actuelle YYYY-MM-DD. Si None, aujourd'hui.

        Returns:
            Liste des reviews effectuees.
        """
        if current_date is None:
            current_date = datetime.now().strftime("%Y-%m-%d")

        pending = self.db.get_pending_signal_reviews(current_date)
        if not pending:
            logger.info("Aucun signal en attente de review")
            return []

        logger.info(f"Review de {len(pending)} signaux en attente")
        reviews = []

        for signal in pending:
            review = self._review_signal(signal, current_date)
            if review:
                self.db.insert_signal_review(review)
                reviews.append(review)
                logger.info(
                    f"Review {signal['ticker']}: {review['outcome']} "
                    f"({review['performance_pct']:+.2f}%)"
                )

        logger.info(f"{len(reviews)}/{len(pending)} signaux reviewes")
        return reviews

    def _review_signal(self, signal: dict, current_date: str) -> dict | None:
        """Review un signal individuel."""
        ticker = signal["ticker"]
        signal_price = signal.get("signal_price")

        if not signal_price or signal_price <= 0:
            logger.warning(f"Signal {ticker} sans prix, skip review")
            return None

        # Trouver le prix a J+3
        review_price = self._get_review_price(ticker, signal["date"])
        if review_price is None:
            logger.warning(f"Pas de prix J+3 pour {ticker}, skip review")
            return None

        # Calculer performance
        perf = self._calculate_performance(signal_price, review_price)
        outcome = self._classify_outcome(perf)

        # Analyser les echecs
        failure_reason = self._analyze_failure(signal, perf)

        # Calculer la date de review (signal_date + 3 jours)
        signal_dt = datetime.strptime(signal["date"], "%Y-%m-%d")
        review_date = (signal_dt + timedelta(days=3)).strftime("%Y-%m-%d")

        return {
            "signal_id": signal["id"],
            "ticker": ticker,
            "signal_date": signal["date"],
            "signal_price": signal_price,
            "review_date": review_date,
            "review_price": review_price,
            "performance_pct": round(perf, 2),
            "outcome": outcome,
            "failure_reason": failure_reason,
            "catalyst_type": signal.get("catalyst_type"),
            "features_json": signal.get("features_json"),
            "reviewed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _get_review_price(self, ticker: str, signal_date: str) -> float | None:
        """Recupere le prix de cloture a J+3 (ou le plus proche apres)."""
        signal_dt = datetime.strptime(signal_date, "%Y-%m-%d")
        target_date = (signal_dt + timedelta(days=3)).strftime("%Y-%m-%d")
        # Chercher le prix le plus proche dans une fenetre J+2 a J+5
        # (pour gerer les weekends)
        date_start = (signal_dt + timedelta(days=2)).strftime("%Y-%m-%d")
        date_end = (signal_dt + timedelta(days=5)).strftime("%Y-%m-%d")

        prices = self.db.get_prices(ticker)
        candidates = [
            p for p in prices
            if date_start <= p["date"] <= date_end
        ]
        if not candidates:
            return None

        # Prendre le prix le plus proche de J+3
        candidates.sort(key=lambda p: abs(
            (datetime.strptime(p["date"], "%Y-%m-%d") -
             datetime.strptime(target_date, "%Y-%m-%d")).days
        ))
        return candidates[0]["close"]

    def _calculate_performance(self, signal_price: float,
                                review_price: float) -> float:
        """Calcule la performance en pourcentage."""
        if signal_price <= 0:
            return 0.0
        return round((review_price - signal_price) / signal_price * 100, 2)

    def _classify_outcome(self, performance_pct: float) -> str:
        """Classifie le resultat: WIN (>=4.5%), LOSS (<0%), NEUTRAL."""
        if performance_pct >= self.win_threshold:
            return "WIN"
        elif performance_pct < 0:
            return "LOSS"
        return "NEUTRAL"

    def _analyze_failure(self, signal: dict, performance_pct: float) -> str | None:
        """Analyse pourquoi un signal a echoue. None si pas un echec."""
        if performance_pct >= 0:
            return None

        parts = []

        # Type de catalyseur
        cat_type = signal.get("catalyst_type", "UNKNOWN")
        parts.append(f"Catalyseur {cat_type} non confirme par le marche")

        # Analyser les features si disponibles
        features_json = signal.get("features_json")
        if features_json:
            try:
                features = json.loads(features_json)
                rsi = features.get("rsi_14")
                if rsi and rsi > 60:
                    parts.append(f"RSI eleve ({rsi:.0f}) = surachat potentiel")
                vol = features.get("volume_ratio_20")
                if vol and vol < 0.5:
                    parts.append(f"Volume faible ({vol:.2f}x) = pas de conviction")
                sentiment = features.get("news_sentiment")
                if sentiment is not None and sentiment < 0.2:
                    parts.append(f"Sentiment faible ({sentiment:.2f})")
            except (json.JSONDecodeError, TypeError):
                pass

        return " | ".join(parts)
```

### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_signal_reviewer.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add src/feedback/__init__.py src/feedback/signal_reviewer.py tests/test_signal_reviewer.py
git commit -m "feat(feedback): add SignalReviewer — review signals at J+3"
```

---

## Task 3: PerformanceTracker — Stats + regles adaptatives

**Files:**
- Create: `src/feedback/performance_tracker.py`
- Create: `tests/test_performance_tracker.py`

### Step 1: Write failing tests

```python
"""Tests pour PerformanceTracker."""

import os
import json
import tempfile

import pytest

from src.core.database import Database
from src.feedback.performance_tracker import PerformanceTracker


def _seed_reviews(db, reviews_data):
    """Helper: insere des signaux + reviews de test."""
    for i, (ticker, outcome, perf, cat_type, score) in enumerate(reviews_data):
        db.insert_signal({
            "ticker": ticker, "date": f"2026-03-{01+i:02d}",
            "score": score, "signal_price": 10.0,
            "sent_at": f"2026-03-{01+i:02d} 10:00:00",
        })
        signals = db.get_signals(ticker)
        signal = [s for s in signals if s["date"] == f"2026-03-{01+i:02d}"][0]
        db.insert_signal_review({
            "signal_id": signal["id"], "ticker": ticker,
            "signal_date": f"2026-03-{01+i:02d}", "signal_price": 10.0,
            "review_date": f"2026-03-{04+i:02d}",
            "review_price": 10.0 * (1 + perf / 100),
            "performance_pct": perf, "outcome": outcome,
            "failure_reason": None, "catalyst_type": cat_type,
            "features_json": None,
            "reviewed_at": f"2026-03-{04+i:02d} 18:00:00",
        })


class TestWinRateByCatalyst:
    """Tests du win rate par type de catalyseur."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_win_rate_by_catalyst(self):
        _seed_reviews(self.db, [
            ("A.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("B.PA", "WIN", 6.0, "EARNINGS", 0.90),
            ("C.PA", "LOSS", -2.0, "EARNINGS", 0.80),
            ("D.PA", "LOSS", -3.0, "TECHNICAL", 0.78),
            ("E.PA", "LOSS", -1.0, "TECHNICAL", 0.76),
        ])
        tracker = PerformanceTracker(self.db)
        rates = tracker.win_rate_by_catalyst()
        assert rates["EARNINGS"]["win_rate"] == pytest.approx(2/3, abs=0.01)
        assert rates["TECHNICAL"]["win_rate"] == 0.0


class TestAdaptiveThreshold:
    """Tests de l'ajustement du seuil."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_threshold_increases_on_low_win_rate(self):
        """Seuil monte si win rate < 50%."""
        _seed_reviews(self.db, [
            ("A.PA", "LOSS", -2.0, "EARNINGS", 0.80),
            ("B.PA", "LOSS", -3.0, "TECHNICAL", 0.78),
            ("C.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("D.PA", "LOSS", -1.0, "TECHNICAL", 0.76),
            ("E.PA", "LOSS", -4.0, "UPGRADE", 0.82),
        ])
        tracker = PerformanceTracker(self.db, base_threshold=0.75)
        new_threshold = tracker.compute_adaptive_threshold()
        assert new_threshold > 0.75

    def test_threshold_stable_on_good_win_rate(self):
        """Seuil stable si win rate >= 60%."""
        _seed_reviews(self.db, [
            ("A.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("B.PA", "WIN", 6.0, "UPGRADE", 0.90),
            ("C.PA", "WIN", 4.5, "EARNINGS", 0.82),
            ("D.PA", "NEUTRAL", 2.0, "TECHNICAL", 0.78),
            ("E.PA", "LOSS", -1.0, "TECHNICAL", 0.76),
        ])
        tracker = PerformanceTracker(self.db, base_threshold=0.75)
        new_threshold = tracker.compute_adaptive_threshold()
        assert new_threshold == 0.75  # No increase needed


class TestFilterRuleGeneration:
    """Tests de la generation de regles."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_generates_exclude_rule_for_bad_catalyst(self):
        """Genere une regle d'exclusion si un catalyseur a < 30% win rate."""
        _seed_reviews(self.db, [
            ("A.PA", "LOSS", -2.0, "TECHNICAL", 0.80),
            ("B.PA", "LOSS", -3.0, "TECHNICAL", 0.78),
            ("C.PA", "LOSS", -1.0, "TECHNICAL", 0.76),
            ("D.PA", "LOSS", -4.0, "TECHNICAL", 0.82),
            ("E.PA", "LOSS", -2.0, "TECHNICAL", 0.79),
            ("F.PA", "WIN", 5.0, "EARNINGS", 0.85),
        ])
        tracker = PerformanceTracker(self.db)
        rules = tracker.generate_filter_rules()
        # Should generate a rule to exclude TECHNICAL
        cat_rules = [r for r in rules if r["rule_type"] == "EXCLUDE_CATALYST"]
        assert len(cat_rules) >= 1
        rule_data = json.loads(cat_rules[0]["rule_json"])
        assert rule_data["catalyst_type"] == "TECHNICAL"

    def test_no_rule_if_not_enough_samples(self):
        """Pas de regle si < 5 samples pour un catalyseur."""
        _seed_reviews(self.db, [
            ("A.PA", "LOSS", -2.0, "TECHNICAL", 0.80),
            ("B.PA", "LOSS", -3.0, "TECHNICAL", 0.78),
        ])
        tracker = PerformanceTracker(self.db, min_samples=5)
        rules = tracker.generate_filter_rules()
        assert len(rules) == 0


class TestDailySummary:
    """Tests du resume quotidien."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_daily_summary_format(self):
        _seed_reviews(self.db, [
            ("SAN.PA", "WIN", 5.2, "EARNINGS", 0.85),
            ("ATO.PA", "LOSS", -2.3, "UPGRADE", 0.80),
        ])
        tracker = PerformanceTracker(self.db)
        summary = tracker.get_daily_summary([
            {"ticker": "SAN.PA", "outcome": "WIN", "performance_pct": 5.2,
             "catalyst_type": "EARNINGS"},
            {"ticker": "ATO.PA", "outcome": "LOSS", "performance_pct": -2.3,
             "catalyst_type": "UPGRADE", "failure_reason": "Catalyseur non confirme"},
        ])
        assert "SAN.PA" in summary
        assert "ATO.PA" in summary
        assert "WIN" in summary or "5.2" in summary


class TestWeeklySummary:
    """Tests du resume hebdomadaire."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_weekly_summary_format(self):
        _seed_reviews(self.db, [
            ("SAN.PA", "WIN", 5.0, "EARNINGS", 0.85),
            ("DBV.PA", "LOSS", -2.0, "TECHNICAL", 0.78),
            ("MAU.PA", "NEUTRAL", 2.0, "UPGRADE", 0.80),
        ])
        tracker = PerformanceTracker(self.db)
        summary = tracker.get_weekly_summary("2026-03-01", "2026-03-07")
        assert "3" in summary  # 3 signals
        assert "WIN" in summary or "1" in summary
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_performance_tracker.py -v`
Expected: FAIL (module not found)

### Step 3: Implement PerformanceTracker

**`src/feedback/performance_tracker.py`**:
```python
"""Tracking des performances et generation de regles adaptatives.

Analyse les reviews pour detecter les patterns d'echec,
ajuster le seuil de score, et generer des regles de filtrage.
"""

import json
from datetime import datetime

from loguru import logger

from src.core.database import Database


class PerformanceTracker:
    """Analyse les performances et genere des regles de filtrage."""

    def __init__(self, db: Database, base_threshold: float = 0.75,
                 min_samples: int = 5):
        self.db = db
        self.base_threshold = base_threshold
        self.min_samples = min_samples

    def win_rate_by_catalyst(self) -> dict:
        """Calcule le win rate par type de catalyseur."""
        reviews = self.db.get_signal_reviews()
        by_cat = {}

        for r in reviews:
            cat = r.get("catalyst_type") or "UNKNOWN"
            if cat not in by_cat:
                by_cat[cat] = {"wins": 0, "total": 0}
            by_cat[cat]["total"] += 1
            if r["outcome"] == "WIN":
                by_cat[cat]["wins"] += 1

        for cat, data in by_cat.items():
            data["win_rate"] = data["wins"] / data["total"] if data["total"] > 0 else 0.0

        return by_cat

    def compute_adaptive_threshold(self) -> float:
        """Calcule le seuil adaptatif base sur le win rate global.

        - Win rate < 40% -> seuil + 0.04
        - Win rate 40-50% -> seuil + 0.02
        - Win rate >= 60% -> seuil inchange
        - Max threshold: 0.95
        """
        stats = self.db.get_review_stats()
        total = stats["total"]

        if total < self.min_samples:
            return self.base_threshold

        win_rate = stats["wins"] / total if total > 0 else 0.0

        threshold = self.base_threshold
        if win_rate < 0.40:
            threshold += 0.04
        elif win_rate < 0.50:
            threshold += 0.02

        threshold = min(threshold, 0.95)

        logger.info(
            f"Seuil adaptatif: {threshold:.2f} "
            f"(win_rate={win_rate:.1%}, base={self.base_threshold})"
        )
        return round(threshold, 2)

    def generate_filter_rules(self) -> list[dict]:
        """Genere des regles de filtrage basees sur les patterns d'echec.

        Regle EXCLUDE_CATALYST si un type a < 30% win rate sur 5+ samples.
        """
        rates = self.win_rate_by_catalyst()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_rules = []

        # Desactiver les anciennes regles auto-generees
        existing = self.db.get_active_filter_rules()
        for rule in existing:
            self.db.deactivate_filter_rule(rule["id"])

        for cat, data in rates.items():
            if data["total"] < self.min_samples:
                continue
            if data["win_rate"] < 0.30:
                rule = {
                    "rule_type": "EXCLUDE_CATALYST",
                    "rule_json": json.dumps({"catalyst_type": cat}),
                    "win_rate": round(data["win_rate"], 4),
                    "sample_size": data["total"],
                    "created_at": now,
                    "active": 1,
                }
                self.db.insert_filter_rule(rule)
                new_rules.append(rule)
                logger.info(
                    f"Regle generee: exclure {cat} "
                    f"(win_rate={data['win_rate']:.1%}, n={data['total']})"
                )

        # Regle MAX_SIGNALS_PER_DAY
        rule_max = {
            "rule_type": "MAX_SIGNALS_PER_DAY",
            "rule_json": json.dumps({"max": 2}),
            "win_rate": None,
            "sample_size": None,
            "created_at": now,
            "active": 1,
        }
        self.db.insert_filter_rule(rule_max)
        new_rules.append(rule_max)

        return new_rules

    def get_daily_summary(self, reviews: list[dict]) -> str:
        """Formate un resume quotidien des reviews en HTML."""
        if not reviews:
            return "<b>Review J+3</b>\nAucun signal a reviewer aujourd'hui."

        date = reviews[0].get("review_date", "")
        lines = [f"<b>Review J+3 -- {date}</b>", ""]

        for r in reviews:
            ticker = r["ticker"]
            perf = r["performance_pct"]
            outcome = r["outcome"]
            cat = r.get("catalyst_type", "")

            if outcome == "WIN":
                icon = "OK"
                detail = f"{cat} confirme"
            elif outcome == "LOSS":
                icon = "X"
                detail = r.get("failure_reason", "Echec")
            else:
                icon = "-"
                detail = "Performance insuffisante"

            lines.append(f"[{icon}] {ticker}: {perf:+.1f}% -- {detail}")

        # Stats globales
        stats = self.db.get_review_stats()
        total = stats["total"]
        if total > 0:
            wr = stats["wins"] / total
            lines.append("")
            lines.append(f"Win rate global: {wr:.0%} ({stats['wins']}/{total})")

        threshold = self.compute_adaptive_threshold()
        lines.append(f"Seuil actuel: {threshold:.2f}")

        return "\n".join(lines)

    def get_weekly_summary(self, date_start: str, date_end: str) -> str:
        """Formate un resume hebdomadaire en HTML."""
        reviews = self.db.get_reviews_in_period(date_start, date_end)

        lines = [f"<b>Bilan semaine -- {date_start} au {date_end}</b>", ""]

        if not reviews:
            lines.append("Aucune review cette semaine.")
            return "\n".join(lines)

        wins = [r for r in reviews if r["outcome"] == "WIN"]
        losses = [r for r in reviews if r["outcome"] == "LOSS"]
        neutrals = [r for r in reviews if r["outcome"] == "NEUTRAL"]
        perfs = [r["performance_pct"] for r in reviews]

        lines.append(
            f"Signaux: {len(reviews)} | WIN: {len(wins)} | "
            f"LOSS: {len(losses)} | NEUTRAL: {len(neutrals)}"
        )

        if perfs:
            avg_perf = sum(perfs) / len(perfs)
            best = max(reviews, key=lambda r: r["performance_pct"])
            worst = min(reviews, key=lambda r: r["performance_pct"])
            wr = len(wins) / len(reviews)

            lines.append(f"Win rate: {wr:.0%} | Perf moyenne: {avg_perf:+.1f}%")
            lines.append(
                f"Meilleur: {best['ticker']} {best['performance_pct']:+.1f}% | "
                f"Pire: {worst['ticker']} {worst['performance_pct']:+.1f}%"
            )

        # Regles actives
        rules = self.db.get_active_filter_rules()
        if rules:
            lines.append("")
            lines.append("Regles actives:")
            for rule in rules:
                data = json.loads(rule["rule_json"])
                if rule["rule_type"] == "EXCLUDE_CATALYST":
                    cat = data["catalyst_type"]
                    lines.append(
                        f"  - Exclure {cat} (win_rate={rule['win_rate']:.0%})"
                    )
                elif rule["rule_type"] == "MAX_SIGNALS_PER_DAY":
                    lines.append(f"  - Max {data['max']} signaux/jour")

        # Seuil
        threshold = self.compute_adaptive_threshold()
        lines.append(f"\nSeuil adaptatif: {threshold:.2f}")

        # Modele
        active_model = self.db.get_active_model_version()
        if active_model:
            lines.append(f"Modele actif: {active_model['version']}")

        return "\n".join(lines)
```

### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_performance_tracker.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add src/feedback/performance_tracker.py tests/test_performance_tracker.py
git commit -m "feat(feedback): add PerformanceTracker — adaptive threshold + filter rules"
```

---

## Task 4: ModelRetrainer — Re-entrainement avec validation

**Files:**
- Create: `src/feedback/model_retrainer.py`
- Create: `tests/test_model_retrainer.py`

### Step 1: Write failing tests

```python
"""Tests pour ModelRetrainer."""

import os
import tempfile
import shutil

import pytest

from src.core.database import Database
from src.feedback.model_retrainer import ModelRetrainer


class TestShouldRetrain:
    """Tests de la condition de re-entrainement."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()

    def teardown_method(self):
        self.tmp.cleanup()

    def test_should_not_retrain_if_not_enough_reviews(self):
        retrainer = ModelRetrainer(self.db, min_reviews_for_retrain=50)
        assert retrainer.should_retrain() is False

    def test_should_retrain_after_threshold(self):
        # Insert 50 reviews
        for i in range(50):
            self.db.insert_signal({
                "ticker": f"T{i}.PA", "date": f"2026-03-{(i % 28) + 1:02d}",
                "score": 0.80, "signal_price": 10.0,
                "sent_at": f"2026-03-{(i % 28) + 1:02d} 10:00:00",
            })
            signals = self.db.get_signals(f"T{i}.PA")
            self.db.insert_signal_review({
                "signal_id": signals[0]["id"], "ticker": f"T{i}.PA",
                "signal_date": f"2026-03-{(i % 28) + 1:02d}",
                "signal_price": 10.0,
                "review_date": f"2026-03-{(i % 28) + 1:02d}",
                "review_price": 10.5,
                "performance_pct": 5.0, "outcome": "WIN",
                "failure_reason": None, "catalyst_type": "EARNINGS",
                "features_json": None,
                "reviewed_at": "2026-03-07 18:00:00",
            })
        retrainer = ModelRetrainer(self.db, min_reviews_for_retrain=50)
        assert retrainer.should_retrain() is True


class TestBackupModel:
    """Tests de la sauvegarde du modele."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()
        # Create a fake model file
        self.model_dir = os.path.join(self.tmp.name, "models")
        os.makedirs(self.model_dir)
        self.model_path = os.path.join(self.model_dir, "nicolas_v1.joblib")
        with open(self.model_path, "w") as f:
            f.write("fake model")

    def teardown_method(self):
        self.tmp.cleanup()

    def test_backup_creates_copy(self):
        retrainer = ModelRetrainer(self.db)
        backup_path = retrainer._backup_model(self.model_path)
        assert os.path.exists(backup_path)
        assert "backup" in backup_path
        assert os.path.exists(self.model_path)  # Original untouched

    def test_backup_path_format(self):
        retrainer = ModelRetrainer(self.db)
        backup_path = retrainer._backup_model(self.model_path)
        assert backup_path.endswith(".joblib")
        assert "nicolas_v1_backup_" in backup_path


class TestCompareModels:
    """Tests de la comparaison de modeles."""

    def test_new_model_better(self):
        retrainer = ModelRetrainer.__new__(ModelRetrainer)
        old = {"accuracy": 0.82, "precision": 0.85, "recall": 0.80, "f1": 0.82}
        new = {"accuracy": 0.88, "precision": 0.90, "recall": 0.85, "f1": 0.87}
        assert retrainer._is_new_model_better(old, new) is True

    def test_new_model_worse(self):
        retrainer = ModelRetrainer.__new__(ModelRetrainer)
        old = {"accuracy": 0.88, "precision": 0.90, "recall": 0.85, "f1": 0.87}
        new = {"accuracy": 0.82, "precision": 0.85, "recall": 0.80, "f1": 0.82}
        assert retrainer._is_new_model_better(old, new) is False

    def test_marginal_improvement_not_enough(self):
        """Le nouveau doit etre > 1% meilleur en f1."""
        retrainer = ModelRetrainer.__new__(ModelRetrainer)
        old = {"accuracy": 0.85, "precision": 0.85, "recall": 0.85, "f1": 0.850}
        new = {"accuracy": 0.855, "precision": 0.855, "recall": 0.855, "f1": 0.855}
        assert retrainer._is_new_model_better(old, new) is False
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_model_retrainer.py -v`
Expected: FAIL (module not found)

### Step 3: Implement ModelRetrainer

**`src/feedback/model_retrainer.py`**:
```python
"""Re-entrainement du modele avec backup et validation.

Ne remplace le modele actif que si le nouveau est strictement meilleur.
Conserve un backup de l'ancien modele pour rollback.
"""

import os
import shutil
from datetime import datetime

from loguru import logger

from src.core.database import Database


class ModelRetrainer:
    """Gere le re-entrainement du modele avec validation."""

    def __init__(self, db: Database, min_reviews_for_retrain: int = 50):
        self.db = db
        self.min_reviews = min_reviews_for_retrain

    def should_retrain(self) -> bool:
        """Verifie si assez de reviews pour justifier un re-entrainement."""
        stats = self.db.get_review_stats()
        total = stats["total"]

        # Verifier si on a assez de nouvelles reviews depuis le dernier retrain
        active_model = self.db.get_active_model_version()
        last_training_signals = 0
        if active_model:
            last_training_signals = active_model.get("training_signals", 0)

        new_reviews = total - last_training_signals
        if new_reviews >= self.min_reviews:
            logger.info(f"Re-entrainement possible: {new_reviews} nouvelles reviews")
            return True

        logger.debug(
            f"Pas assez de reviews: {new_reviews}/{self.min_reviews}"
        )
        return False

    def retrain_with_validation(self, current_model_path: str,
                                  new_model_dir: str = "data/models") -> dict:
        """Re-entraine le modele, compare, et deploie si meilleur.

        Args:
            current_model_path: Chemin du modele actif.
            new_model_dir: Dossier pour le nouveau modele.

        Returns:
            Dict avec le resultat: deployed, metrics, etc.
        """
        from src.analysis.feature_engine import FeatureEngine
        from src.model.trainer import Trainer
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
        )

        # 1. Backup
        backup_path = self._backup_model(current_model_path)
        logger.info(f"Backup: {backup_path}")

        # 2. Charger l'ancien modele et ses metriques
        old_trainer = Trainer()
        old_trainer.load_model(current_model_path)

        # 3. Construire les features (donnees historiques)
        engine = FeatureEngine(self.db)
        features_df = engine.build_all_features()

        if len(features_df) < 20:
            logger.warning("Pas assez de donnees pour re-entrainer")
            return {"deployed": False, "reason": "not_enough_data"}

        # 4. Walk-forward validation ancien modele
        old_results = old_trainer.walk_forward_validate(features_df)
        old_metrics = {
            "accuracy": old_results.get("accuracy", 0),
            "precision": old_results.get("precision", 0),
            "recall": old_results.get("recall", 0),
            "f1": old_results.get("f1", 0),
        }

        # 5. Entrainer nouveau modele (full data)
        new_trainer = Trainer()
        X, y = new_trainer.prepare_data(features_df)
        new_trainer.train(X, y)

        # Evaluer via walk-forward
        new_results = new_trainer.walk_forward_validate(features_df)
        new_metrics = {
            "accuracy": new_results.get("accuracy", 0),
            "precision": new_results.get("precision", 0),
            "recall": new_results.get("recall", 0),
            "f1": new_results.get("f1", 0),
        }

        # 6. Comparer
        is_better = self._is_new_model_better(old_metrics, new_metrics)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = self.db.get_review_stats()

        result = {
            "deployed": is_better,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
            "backup_path": backup_path,
        }

        if is_better:
            # Deploy
            version = self._next_version()
            new_path = os.path.join(new_model_dir, f"nicolas_{version}.joblib")
            new_trainer.save_model(new_path)

            self.db.insert_model_version({
                "version": version,
                "file_path": new_path,
                "trained_at": now,
                "training_signals": stats["total"],
                "accuracy": new_metrics["accuracy"],
                "precision_score": new_metrics["precision"],
                "recall": new_metrics["recall"],
                "f1": new_metrics["f1"],
                "is_active": 0,
                "notes": f"Retrained, f1 {old_metrics['f1']:.3f} -> {new_metrics['f1']:.3f}",
            })
            # Switch active
            versions = self.db.get_all_model_versions()
            new_version = [v for v in versions if v["version"] == version][0]
            self.db.set_active_model(new_version["id"])

            result["new_path"] = new_path
            result["version"] = version
            logger.info(
                f"Nouveau modele deploye: {version} "
                f"(f1: {old_metrics['f1']:.3f} -> {new_metrics['f1']:.3f})"
            )
        else:
            logger.info(
                f"Ancien modele conserve "
                f"(new f1={new_metrics['f1']:.3f} vs old f1={old_metrics['f1']:.3f})"
            )

        return result

    def _backup_model(self, model_path: str) -> str:
        """Cree un backup du modele actif."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modele non trouve: {model_path}")

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(model_path)
        backup_path = f"{base}_backup_{date_str}{ext}"
        shutil.copy2(model_path, backup_path)
        logger.info(f"Backup modele: {backup_path}")
        return backup_path

    def _is_new_model_better(self, old_metrics: dict, new_metrics: dict) -> bool:
        """Compare deux modeles. Le nouveau doit avoir >= 1% de f1 en plus."""
        old_f1 = old_metrics.get("f1", 0)
        new_f1 = new_metrics.get("f1", 0)
        return (new_f1 - old_f1) >= 0.01

    def _next_version(self) -> str:
        """Determine le prochain numero de version."""
        versions = self.db.get_all_model_versions()
        if not versions:
            return "v2"
        max_num = 1
        for v in versions:
            try:
                num = int(v["version"].replace("v", ""))
                max_num = max(max_num, num)
            except ValueError:
                pass
        return f"v{max_num + 1}"

    def format_retrain_report(self, result: dict) -> str:
        """Formate le rapport de re-entrainement en HTML."""
        old = result["old_metrics"]
        new = result["new_metrics"]

        lines = ["<b>Re-entrainement du modele</b>", ""]

        if result["deployed"]:
            lines.append(
                f"Nouveau modele deploye: {result.get('version', '?')}"
            )
        else:
            lines.append("Ancien modele conserve (nouveau pas assez meilleur)")

        lines.append("")
        lines.append("Metriques comparees:")
        lines.append(f"  Accuracy: {old['accuracy']:.1%} -> {new['accuracy']:.1%}")
        lines.append(f"  Precision: {old['precision']:.1%} -> {new['precision']:.1%}")
        lines.append(f"  Recall: {old['recall']:.1%} -> {new['recall']:.1%}")
        lines.append(f"  F1: {old['f1']:.1%} -> {new['f1']:.1%}")
        lines.append("")
        lines.append(f"Backup: {result.get('backup_path', 'N/A')}")

        return "\n".join(lines)
```

### Step 4: Run tests to verify they pass

Run: `uv run pytest tests/test_model_retrainer.py -v`
Expected: ALL PASS

### Step 5: Commit

```bash
git add src/feedback/model_retrainer.py tests/test_model_retrainer.py
git commit -m "feat(feedback): add ModelRetrainer — safe retrain with backup + validation"
```

---

## Task 5: Modifier SignalFilter — filtrage adaptatif

**Files:**
- Modify: `src/alerts/signal_filter.py`
- Modify: `tests/test_signal_filter.py`

### Step 1: Write new tests

Add to `tests/test_signal_filter.py`:

```python
class TestAdaptiveFiltering:
    """Tests du filtrage adaptatif avec regles dynamiques."""

    def setup_method(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "test.db"))
        self.db.init_db()
        self.config = {"threshold": 0.75, "cooldown_hours": 24}
        self.sf = SignalFilter(self.db, self.config)

    def teardown_method(self):
        self.tmp.cleanup()

    def test_exclude_catalyst_rule(self):
        """Regle EXCLUDE_CATALYST bloque les signaux du type exclu."""
        self.db.insert_filter_rule({
            "rule_type": "EXCLUDE_CATALYST",
            "rule_json": '{"catalyst_type": "TECHNICAL"}',
            "win_rate": 0.20, "sample_size": 10,
            "created_at": "2026-03-07", "active": 1,
        })
        signals = [
            {"ticker": "SAN.PA", "score": 0.85, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.82, "catalyst_type": "TECHNICAL"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 1
        assert filtered[0]["ticker"] == "SAN.PA"

    def test_max_signals_per_day_rule(self):
        """Regle MAX_SIGNALS_PER_DAY limite le nombre de signaux."""
        self.db.insert_filter_rule({
            "rule_type": "MAX_SIGNALS_PER_DAY",
            "rule_json": '{"max": 2}',
            "win_rate": None, "sample_size": None,
            "created_at": "2026-03-07", "active": 1,
        })
        signals = [
            {"ticker": "SAN.PA", "score": 0.90, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.88, "catalyst_type": "UPGRADE"},
            {"ticker": "MAU.PA", "score": 0.85, "catalyst_type": "EARNINGS"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 2

    def test_adaptive_threshold(self):
        """Le seuil adaptatif est utilise si present."""
        self.db.insert_filter_rule({
            "rule_type": "ADAPTIVE_THRESHOLD",
            "rule_json": '{"threshold": 0.85}',
            "win_rate": None, "sample_size": None,
            "created_at": "2026-03-07", "active": 1,
        })
        signals = [
            {"ticker": "SAN.PA", "score": 0.90, "catalyst_type": "EARNINGS"},
            {"ticker": "DBV.PA", "score": 0.82, "catalyst_type": "UPGRADE"},
        ]
        filtered = self.sf.filter_signals(signals)
        assert len(filtered) == 1
        assert filtered[0]["ticker"] == "SAN.PA"
```

### Step 2: Run tests to verify they fail

Run: `uv run pytest tests/test_signal_filter.py -v -k "Adaptive"`
Expected: FAIL

### Step 3: Modify SignalFilter

Update `src/alerts/signal_filter.py` to:

1. Load active filter rules from DB at the start of `filter_signals()`
2. Apply EXCLUDE_CATALYST rules
3. Apply ADAPTIVE_THRESHOLD rule (overrides config threshold)
4. Apply MAX_SIGNALS_PER_DAY rule
5. Keep existing cooldown logic

Key changes to `filter_signals()`:
```python
def filter_signals(self, signals: list[dict]) -> list[dict]:
    # Load dynamic rules
    rules = self.db.get_active_filter_rules()
    exclude_catalysts = self._get_excluded_catalysts(rules)
    max_per_day = self._get_max_per_day(rules)
    adaptive_threshold = self._get_adaptive_threshold(rules)

    effective_threshold = adaptive_threshold or self.threshold

    filtered = []
    for signal in signals:
        if signal.get("catalyst_type") in exclude_catalysts:
            logger.debug(f"Signal {signal['ticker']} exclu: catalyst {signal['catalyst_type']}")
            continue
        if signal.get("score", 0) < effective_threshold:
            continue
        if not self._passes_cooldown(signal):
            continue
        filtered.append(signal)

    # Apply max per day
    if max_per_day and len(filtered) > max_per_day:
        filtered = filtered[:max_per_day]

    logger.info(
        f"SignalFilter: {len(filtered)}/{len(signals)} retenus "
        f"(seuil={effective_threshold})"
    )
    return filtered
```

### Step 4: Run ALL signal_filter tests

Run: `uv run pytest tests/test_signal_filter.py -v`
Expected: ALL PASS (old + new)

### Step 5: Commit

```bash
git add src/alerts/signal_filter.py tests/test_signal_filter.py
git commit -m "feat(filter): adaptive filtering with dynamic rules from feedback loop"
```

---

## Task 6: Stocker signal_price lors de l'emission

**Files:**
- Modify: `src/alerts/signal_filter.py` (method `record_signal`)
- Modify: `scripts/run_scanner.py` (pass current_price)

### Step 1: Update record_signal to include signal_price

In `signal_filter.py`, `record_signal()`:
```python
def record_signal(self, signal: dict):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    self.db.insert_signal({
        "ticker": signal["ticker"],
        "date": signal["date"],
        "score": signal["score"],
        "signal_price": signal.get("current_price"),
        "catalyst_type": signal.get("catalyst_type"),
        "catalyst_news_title": signal.get("catalyst_news_title"),
        "features_json": signal.get("features_json"),
        "sent_at": now,
    })
```

### Step 2: Run all tests

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

### Step 3: Commit

```bash
git add src/alerts/signal_filter.py
git commit -m "fix(signals): store signal_price when recording signals for J+3 review"
```

---

## Task 7: Script run_feedback.py

**Files:**
- Create: `scripts/run_feedback.py`

### Step 1: Create the script

```python
"""Pipeline de feedback — review, regles adaptatives, re-entrainement.

Usage:
    uv run python scripts/run_feedback.py              # Review + update rules
    uv run python scripts/run_feedback.py --stats       # Stats globales
    uv run python scripts/run_feedback.py --retrain     # Forcer re-entrainement
    uv run python scripts/run_feedback.py --dry-run     # Review sans Telegram
    uv run python scripts/run_feedback.py --weekly      # Envoyer resume hebdo
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.database import Database
from src.feedback.signal_reviewer import SignalReviewer
from src.feedback.performance_tracker import PerformanceTracker
from src.feedback.model_retrainer import ModelRetrainer

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trades.db")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_telegram():
    from src.alerts.telegram_bot import TelegramBot
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return TelegramBot(token, chat_id)
    return None


def run_review(dry_run=False):
    """Review les signaux en attente + mise a jour des regles."""
    db = Database(DB_PATH)
    db.init_db()
    config = load_config()
    base_threshold = config.get("scoring", {}).get("threshold", 0.75)

    # 1. Review J+3
    reviewer = SignalReviewer(db, win_threshold=4.5)
    reviews = reviewer.review_pending()

    if reviews:
        # 2. Envoyer recap Telegram
        tracker = PerformanceTracker(db, base_threshold=base_threshold)
        summary = tracker.get_daily_summary(reviews)

        if dry_run:
            print(f"\n[DRY-RUN] Review quotidienne:\n{summary}\n")
        else:
            telegram = get_telegram()
            if telegram:
                telegram.send_alert_sync(summary)

    # 3. Update regles
    tracker = PerformanceTracker(db, base_threshold=base_threshold)
    rules = tracker.generate_filter_rules()

    # 4. Mettre a jour le seuil adaptatif
    threshold = tracker.compute_adaptive_threshold()
    if threshold != base_threshold:
        import json
        db.insert_filter_rule({
            "rule_type": "ADAPTIVE_THRESHOLD",
            "rule_json": json.dumps({"threshold": threshold}),
            "win_rate": None, "sample_size": None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": 1,
        })

    print(f"Review: {len(reviews)} signaux reviewes")
    print(f"Regles: {len(rules)} regles actives")
    print(f"Seuil adaptatif: {threshold:.2f}")


def run_stats():
    """Affiche les stats globales."""
    db = Database(DB_PATH)
    db.init_db()

    stats = db.get_review_stats()
    print(f"\n=== Stats Feedback Loop ===")
    print(f"Total reviews: {stats['total']}")
    print(f"  WIN: {stats['wins']}")
    print(f"  LOSS: {stats['losses']}")
    print(f"  NEUTRAL: {stats['neutrals']}")

    if stats['total'] > 0:
        wr = stats['wins'] / stats['total']
        print(f"Win rate: {wr:.1%}")

    rules = db.get_active_filter_rules()
    print(f"\nRegles actives: {len(rules)}")
    for r in rules:
        print(f"  - {r['rule_type']}: {r['rule_json']}")

    active_model = db.get_active_model_version()
    if active_model:
        print(f"\nModele actif: {active_model['version']}")
        print(f"  Accuracy: {active_model['accuracy']:.1%}")
        print(f"  F1: {active_model['f1']:.1%}")


def run_retrain(dry_run=False):
    """Force un re-entrainement."""
    db = Database(DB_PATH)
    db.init_db()
    config = load_config()
    model_path = config.get("scoring", {}).get(
        "model_path", "data/models/nicolas_v1.joblib"
    )

    retrainer = ModelRetrainer(db, min_reviews_for_retrain=0)
    result = retrainer.retrain_with_validation(model_path)
    report = retrainer.format_retrain_report(result)

    if dry_run:
        print(f"\n[DRY-RUN] Rapport re-entrainement:\n{report}\n")
    else:
        telegram = get_telegram()
        if telegram:
            telegram.send_alert_sync(report)

    print(report)


def run_weekly(dry_run=False):
    """Envoie le resume hebdomadaire."""
    db = Database(DB_PATH)
    db.init_db()
    config = load_config()
    base_threshold = config.get("scoring", {}).get("threshold", 0.75)

    today = datetime.now()
    week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end = today.strftime("%Y-%m-%d")

    tracker = PerformanceTracker(db, base_threshold=base_threshold)
    summary = tracker.get_weekly_summary(week_start, week_end)

    if dry_run:
        print(f"\n[DRY-RUN] Resume hebdo:\n{summary}\n")
    else:
        telegram = get_telegram()
        if telegram:
            telegram.send_alert_sync(summary)

    print(summary)


def main():
    parser = argparse.ArgumentParser(description="PEA Scanner — feedback loop")
    parser.add_argument("--stats", action="store_true", help="Stats globales")
    parser.add_argument("--retrain", action="store_true", help="Forcer re-entrainement")
    parser.add_argument("--weekly", action="store_true", help="Resume hebdomadaire")
    parser.add_argument("--dry-run", action="store_true", help="Sans Telegram")
    args = parser.parse_args()

    if args.stats:
        run_stats()
    elif args.retrain:
        run_retrain(dry_run=args.dry_run)
    elif args.weekly:
        run_weekly(dry_run=args.dry_run)
    else:
        run_review(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
```

### Step 2: Commit

```bash
git add scripts/run_feedback.py
git commit -m "feat(scripts): add run_feedback.py — review, stats, retrain, weekly summary"
```

---

## Task 8: Integrer les jobs feedback dans run_scanner.py

**Files:**
- Modify: `scripts/run_scanner.py`
- Modify: `config/watchlist.yaml`

### Step 1: Add feedback config to watchlist.yaml

```yaml
# Feedback loop
feedback:
  review_hour: 18          # Review J+3 a 18h
  rules_update_hour: 18    # Mise a jour regles a 18h30
  weekly_day: "sun"         # Resume hebdo le dimanche
  weekly_hour: 20           # A 20h
  retrain_day: "sun"        # Check retrain le dimanche
  retrain_hour: 19          # A 19h
  win_threshold: 4.5        # % pour considerer un WIN
  min_reviews_retrain: 50   # Reviews minimum pour retrain
```

### Step 2: Add feedback jobs to run_scheduler()

In `run_scanner.py`, add 4 new jobs after the existing ones:

```python
# Job 5: Review J+3 — quotidien a 18h (lun-ven)
scheduler.add_job(
    run_daily_review,
    CronTrigger(hour=feedback_config.get("review_hour", 18), minute=0,
                day_of_week="mon-fri"),
    args=[db, config, telegram, dry_run],
    id="daily_review",
    name="Review J+3",
)

# Job 6: Update regles — quotidien a 18h30 (lun-ven)
scheduler.add_job(
    update_filter_rules,
    CronTrigger(hour=feedback_config.get("rules_update_hour", 18), minute=30,
                day_of_week="mon-fri"),
    args=[db, config],
    id="update_rules",
    name="Update regles",
)

# Job 7: Check retrain — dimanche 19h
scheduler.add_job(
    check_retrain,
    CronTrigger(hour=feedback_config.get("retrain_hour", 19), minute=0,
                day_of_week=feedback_config.get("retrain_day", "sun")),
    args=[db, config, telegram, dry_run],
    id="check_retrain",
    name="Check retrain",
)

# Job 8: Resume hebdo — dimanche 20h
scheduler.add_job(
    send_weekly_summary,
    CronTrigger(hour=feedback_config.get("weekly_hour", 20), minute=0,
                day_of_week=feedback_config.get("weekly_day", "sun")),
    args=[db, config, telegram, dry_run],
    id="weekly_summary",
    name="Resume hebdomadaire",
)
```

Add the 4 new functions:
```python
def run_daily_review(db, config, telegram, dry_run):
    from src.feedback.signal_reviewer import SignalReviewer
    from src.feedback.performance_tracker import PerformanceTracker

    feedback_cfg = config.get("feedback", {})
    reviewer = SignalReviewer(db, win_threshold=feedback_cfg.get("win_threshold", 4.5))
    reviews = reviewer.review_pending()

    if reviews and telegram and not dry_run:
        base_threshold = config.get("scoring", {}).get("threshold", 0.75)
        tracker = PerformanceTracker(db, base_threshold=base_threshold)
        summary = tracker.get_daily_summary(reviews)
        telegram.send_alert_sync(summary)

    logger.info(f"Review quotidienne: {len(reviews)} signaux reviewes")


def update_filter_rules(db, config):
    import json
    from src.feedback.performance_tracker import PerformanceTracker

    base_threshold = config.get("scoring", {}).get("threshold", 0.75)
    tracker = PerformanceTracker(db, base_threshold=base_threshold)
    rules = tracker.generate_filter_rules()

    threshold = tracker.compute_adaptive_threshold()
    if threshold != base_threshold:
        db.insert_filter_rule({
            "rule_type": "ADAPTIVE_THRESHOLD",
            "rule_json": json.dumps({"threshold": threshold}),
            "win_rate": None, "sample_size": None,
            "created_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": 1,
        })

    logger.info(f"Regles mises a jour: {len(rules)} regles, seuil={threshold:.2f}")


def check_retrain(db, config, telegram, dry_run):
    from src.feedback.model_retrainer import ModelRetrainer

    feedback_cfg = config.get("feedback", {})
    model_path = config.get("scoring", {}).get("model_path", "data/models/nicolas_v1.joblib")

    retrainer = ModelRetrainer(
        db, min_reviews_for_retrain=feedback_cfg.get("min_reviews_retrain", 50)
    )

    if not retrainer.should_retrain():
        logger.info("Pas assez de reviews pour re-entrainer")
        return

    result = retrainer.retrain_with_validation(model_path)
    report = retrainer.format_retrain_report(result)

    if telegram and not dry_run:
        telegram.send_alert_sync(report)

    logger.info(f"Re-entrainement: deployed={result['deployed']}")


def send_weekly_summary(db, config, telegram, dry_run):
    from datetime import datetime, timedelta
    from src.feedback.performance_tracker import PerformanceTracker

    base_threshold = config.get("scoring", {}).get("threshold", 0.75)
    tracker = PerformanceTracker(db, base_threshold=base_threshold)

    today = datetime.now()
    week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end = today.strftime("%Y-%m-%d")

    summary = tracker.get_weekly_summary(week_start, week_end)

    if telegram and not dry_run:
        telegram.send_alert_sync(summary)

    logger.info("Resume hebdomadaire envoye")
```

### Step 3: Run all tests

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

### Step 4: Commit

```bash
git add scripts/run_scanner.py config/watchlist.yaml
git commit -m "feat(scanner): integrate feedback loop jobs — review, rules, retrain, weekly"
```

---

## Task 9: Register initial model version in DB

**Files:**
- Modify: `scripts/run_scanner.py` (in `run_scheduler` or init)

### Step 1: Add model registration at startup

At the start of `run_scheduler()` and `run_once()`, after `db.init_db()`:

```python
# Register current model if not in DB
if db.get_active_model_version() is None:
    db.insert_model_version({
        "version": "v1",
        "file_path": model_path,
        "trained_at": "2026-02-20 00:00:00",
        "training_signals": 0,
        "accuracy": 0.0,
        "precision_score": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "is_active": 1,
        "notes": "Initial model from historical trades",
    })
```

### Step 2: Commit

```bash
git add scripts/run_scanner.py
git commit -m "feat(scanner): register initial model version on first run"
```

---

## Task 10: Tests integration + deploy VPS

### Step 1: Run all tests

Run: `uv run pytest tests/ -v`
Expected: ALL PASS

### Step 2: Test end-to-end locally

Run: `uv run python scripts/run_feedback.py --stats --dry-run`
Run: `uv run python scripts/run_scanner.py --once --dry-run`

### Step 3: Commit all changes

```bash
git add -A
git commit -m "feat(feedback): complete feedback loop — review J+3, adaptive filtering, auto-retrain"
```

### Step 4: Push to GitHub

```bash
git push origin master
```

### Step 5: Deploy on VPS

```bash
ssh root@31.97.196.120 "cd ~/pea-scanner && git pull && tmux kill-session -t pea-scanner; tmux new-session -d -s pea-scanner 'cd ~/pea-scanner && uv run python scripts/run_scanner.py 2>&1 | tee logs/scanner.log'"
```

### Step 6: Verify VPS deployment

```bash
ssh root@31.97.196.120 "tmux capture-pane -t pea-scanner -p | tail -20"
```
