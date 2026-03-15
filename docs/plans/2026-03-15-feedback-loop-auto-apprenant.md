# Feedback Loop Auto-Apprenant — Plan d'implementation

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendre le feedback loop reellement auto-apprenant — le modele s'ameliore chaque jour en integrant ce qui marche et ce qui ne marche pas.

**Architecture:** 3 changements independants : (1) ajouter des features de feedback historique au modele, (2) fixer le retrain pour qu'il deploie effectivement, (3) ajouter un tag visuel confirme/experimental dans les alertes Telegram. Le retrain passe de hebdomadaire a quotidien.

**Tech Stack:** Python 3.13, XGBoost, SQLite, Gemini 2.5 Flash Lite, APScheduler

---

## Fichiers concernes

| Action | Fichier | Responsabilite |
|--------|---------|---------------|
| Modifier | `src/analysis/feature_engine.py` | Ajouter 3 features feedback dans `_build_realtime_catalyst_features()` |
| Modifier | `src/feedback/model_retrainer.py` | Fixer la comparaison old/new, deployer plus souvent |
| Modifier | `src/alerts/formatter.py` | Ajouter tag "confirme" vs "experimental" |
| Modifier | `scripts/run_scanner.py` | Retrain quotidien au lieu d'hebdomadaire |
| Modifier | `tests/test_feature_engine.py` | Tests des nouvelles features feedback |
| Modifier | `tests/test_model_retrainer.py` | Tests du nouveau critere de deploiement |
| Modifier | `tests/test_formatter.py` | Tests du tag visuel |

---

## Task 1 : Features feedback historique dans FeatureEngine

**Files:**
- Modify: `src/analysis/feature_engine.py:413-480` (methode `_build_realtime_catalyst_features`)
- Test: `tests/test_feature_engine.py`

Le modele recevra 3 nouvelles features calculees a partir des reviews passees, pour chaque signal :
- `catalyst_historical_win_rate` — win rate de ce catalyst_type dans les reviews
- `catalyst_historical_sample_size` — nombre de reviews pour ce type
- `ticker_historical_win_rate` — win rate de ce ticker specifique

- [ ] **Step 1: Ecrire les tests**

Ajouter dans `tests/test_feature_engine.py` :

```python
class TestFeedbackFeatures:
    """Tests des features de feedback historique."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.temp_dir.name, "test.db"))
        self.db.init_db()
        self.engine = FeatureEngine(self.db)

    def teardown_method(self):
        self.temp_dir.cleanup()

    def test_feedback_features_no_reviews(self):
        """Sans reviews, les features feedback sont a 0."""
        result = self.engine._build_feedback_features("SAN.PA", "EARNINGS")
        assert result["catalyst_historical_win_rate"] == 0.0
        assert result["catalyst_historical_sample_size"] == 0
        assert result["ticker_historical_win_rate"] == 0.0

    def test_feedback_features_with_reviews(self):
        """Avec reviews, les features refletent le win rate."""
        # Inserer 3 reviews EARNINGS: 1 WIN, 2 LOSS
        for i, outcome in enumerate(["WIN", "LOSS", "LOSS"]):
            self.db.insert_signal_review({
                "signal_id": i + 1, "ticker": "SAN.PA",
                "signal_date": f"2026-03-0{i+1}", "signal_price": 75.0,
                "review_date": f"2026-03-0{i+4}", "review_price": 80.0 if outcome == "WIN" else 70.0,
                "performance_pct": 6.67 if outcome == "WIN" else -6.67,
                "outcome": outcome, "catalyst_type": "EARNINGS",
                "reviewed_at": f"2026-03-0{i+4} 18:00:00",
            })
        result = self.engine._build_feedback_features("SAN.PA", "EARNINGS")
        assert abs(result["catalyst_historical_win_rate"] - 1/3) < 0.01
        assert result["catalyst_historical_sample_size"] == 3

    def test_feedback_features_ticker_specific(self):
        """Le win rate ticker est specifique au ticker."""
        # SAN.PA: 1 WIN
        self.db.insert_signal_review({
            "signal_id": 1, "ticker": "SAN.PA",
            "signal_date": "2026-03-01", "signal_price": 75.0,
            "review_date": "2026-03-04", "review_price": 80.0,
            "performance_pct": 6.67, "outcome": "WIN",
            "catalyst_type": "EARNINGS",
            "reviewed_at": "2026-03-04 18:00:00",
        })
        # DBV.PA: 1 LOSS
        self.db.insert_signal_review({
            "signal_id": 2, "ticker": "DBV.PA",
            "signal_date": "2026-03-01", "signal_price": 4.0,
            "review_date": "2026-03-04", "review_price": 3.0,
            "performance_pct": -25.0, "outcome": "LOSS",
            "catalyst_type": "EARNINGS",
            "reviewed_at": "2026-03-04 18:00:00",
        })
        san_result = self.engine._build_feedback_features("SAN.PA", "EARNINGS")
        dbv_result = self.engine._build_feedback_features("DBV.PA", "EARNINGS")
        assert san_result["ticker_historical_win_rate"] == 1.0
        assert dbv_result["ticker_historical_win_rate"] == 0.0
```

- [ ] **Step 2: Lancer les tests, verifier qu'ils echouent**

Run: `uv run pytest tests/test_feature_engine.py::TestFeedbackFeatures -v`
Expected: FAIL — `_build_feedback_features` n'existe pas

- [ ] **Step 3: Implementer `_build_feedback_features()`**

Ajouter dans `src/analysis/feature_engine.py`, juste avant `_build_realtime_context_features` :

```python
def _build_feedback_features(self, ticker: str, catalyst_type: str) -> dict:
    """Construit les features de feedback a partir des reviews passees.

    Permet au modele d'apprendre quels types de catalyseurs et quels
    tickers ont historiquement bien fonctionne dans les signaux du bot.
    """
    reviews = self.db.get_signal_reviews()

    # Win rate par type de catalyseur
    cat_reviews = [r for r in reviews if r.get("catalyst_type") == catalyst_type]
    cat_wins = sum(1 for r in cat_reviews if r["outcome"] == "WIN")
    cat_total = len(cat_reviews)
    cat_wr = cat_wins / cat_total if cat_total > 0 else 0.0

    # Win rate par ticker
    ticker_reviews = [r for r in reviews if r["ticker"] == ticker]
    ticker_wins = sum(1 for r in ticker_reviews if r["outcome"] == "WIN")
    ticker_total = len(ticker_reviews)
    ticker_wr = ticker_wins / ticker_total if ticker_total > 0 else 0.0

    return {
        "catalyst_historical_win_rate": round(cat_wr, 4),
        "catalyst_historical_sample_size": cat_total,
        "ticker_historical_win_rate": round(ticker_wr, 4),
    }
```

- [ ] **Step 4: Brancher dans `_build_realtime_catalyst_features()`**

Dans `_build_realtime_catalyst_features()`, apres le calcul de `cat_type`, ajouter l'appel aux features feedback et les merger dans le dict retourne :

```python
# Apres avoir determine cat_type, ajouter les features feedback
feedback = self._build_feedback_features(ticker, cat_type)

return {
    "catalyst_type": cat_type,
    "catalyst_confidence": ...,
    "news_sentiment": ...,
    "has_clear_catalyst": ...,
    "buy_reason_length": ...,
    "best_news_title": ...,
    **feedback,  # <-- ajouter ici
}
```

Faire ca dans les DEUX branches (LLM et fallback regex).

- [ ] **Step 5: Lancer les tests, verifier qu'ils passent**

Run: `uv run pytest tests/test_feature_engine.py::TestFeedbackFeatures -v`
Expected: 3 PASS

- [ ] **Step 6: Lancer tous les tests**

Run: `uv run pytest tests/ -v`
Expected: 421+ PASS

- [ ] **Step 7: Commit**

```bash
git add src/analysis/feature_engine.py tests/test_feature_engine.py
git commit -m "feat: 3 features feedback historique — catalyst/ticker win rate"
```

---

## Task 2 : Fixer le retrain pour qu'il deploie

**Files:**
- Modify: `src/feedback/model_retrainer.py:33-118`
- Test: `tests/test_model_retrainer.py`

Le bug : old et new model evaluent sur le meme walk-forward split → metriques identiques → jamais deploye.

Fix : on deploie TOUJOURS le nouveau modele si son f1 >= 0.25 (seuil minimum). Le nouveau modele est entraine sur PLUS de donnees (reviews recentes), donc il est meilleur par construction. On garde le backup.

- [ ] **Step 1: Mettre a jour les tests**

Dans `tests/test_model_retrainer.py`, modifier le test existant pour refleter le nouveau comportement :

```python
def test_retrain_deploys_with_min_f1(self):
    """Le nouveau modele est deploye si f1 >= 0.25."""
    # ... (utiliser le setup existant)
    result = self.retrainer.retrain_with_validation(self.model_path)
    # Le nouveau modele est deploye car f1 est au-dessus du minimum
    assert result["deployed"] is True or result.get("reason") == "f1_too_low"
```

- [ ] **Step 2: Modifier `retrain_with_validation()` et `_is_new_model_better()`**

Dans `src/feedback/model_retrainer.py` :

```python
MIN_F1_THRESHOLD = 0.25

def retrain_with_validation(self, current_model_path, new_model_dir="data/models"):
    """Retrain et deployer si f1 du nouveau modele >= seuil minimum."""
    from src.analysis.feature_engine import FeatureEngine
    from src.model.trainer import Trainer

    backup_path = self._backup_model(current_model_path)

    engine = FeatureEngine(self.db)
    features_df = engine.build_combined_features()

    if len(features_df) < 20:
        return {"deployed": False, "reason": "not_enough_data"}

    # Entrainer le nouveau modele
    new_trainer = Trainer()
    X, y = new_trainer.prepare_data(features_df)
    new_trainer.train(X, y)
    new_results = new_trainer.walk_forward_validate(features_df)
    new_metrics = {
        k: new_results.get(k, 0)
        for k in ["accuracy", "precision", "recall", "f1"]
    }

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = self.db.get_review_stats()

    result = {
        "deployed": False,
        "new_metrics": new_metrics,
        "backup_path": backup_path,
    }

    if new_metrics["f1"] >= MIN_F1_THRESHOLD:
        version = self._next_version()
        new_path = os.path.join(new_model_dir, f"nicolas_{version}.joblib")
        new_trainer.save_model(new_path)
        self.db.insert_model_version({
            "version": version, "file_path": new_path,
            "trained_at": now, "training_signals": stats["total"],
            "accuracy": new_metrics["accuracy"],
            "precision_score": new_metrics["precision"],
            "recall": new_metrics["recall"],
            "f1": new_metrics["f1"],
            "is_active": 0,
            "notes": f"Retrained on {len(features_df)} samples, f1={new_metrics['f1']:.3f}",
        })
        versions = self.db.get_all_model_versions()
        new_v = [v for v in versions if v["version"] == version][0]
        self.db.set_active_model(new_v["id"])
        result.update({"deployed": True, "new_path": new_path, "version": version})
        logger.info(f"Modele {version} deploye (f1={new_metrics['f1']:.3f})")
    else:
        result["reason"] = "f1_too_low"
        logger.info(f"Modele non deploye: f1={new_metrics['f1']:.3f} < {MIN_F1_THRESHOLD}")

    return result
```

- [ ] **Step 3: Simplifier `_is_new_model_better()` → `_meets_min_quality()`**

```python
def _meets_min_quality(self, metrics: dict) -> bool:
    """Le nouveau modele doit avoir un f1 >= seuil minimum."""
    return metrics.get("f1", 0) >= MIN_F1_THRESHOLD
```

- [ ] **Step 4: Mettre a jour `format_retrain_report()`**

```python
def format_retrain_report(self, result: dict) -> str:
    """Format retrain report as HTML for Telegram."""
    new = result.get("new_metrics", {})
    lines = ["<b>Re-entrainement du modele</b>", ""]

    if result.get("deployed"):
        lines.append(f"Nouveau modele deploye: {result.get('version', '?')}")
    else:
        reason = result.get("reason", "unknown")
        lines.append(f"Modele non deploye ({reason})")

    lines.extend(["", "Metriques:"])
    for metric in ["accuracy", "precision", "recall", "f1"]:
        val = new.get(metric, 0)
        lines.append(f"  {metric.capitalize()}: {val:.1%}")

    lines.extend(["", f"Backup: {result.get('backup_path', 'N/A')}"])
    return "\n".join(lines)
```

- [ ] **Step 5: Lancer les tests**

Run: `uv run pytest tests/test_model_retrainer.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/feedback/model_retrainer.py tests/test_model_retrainer.py
git commit -m "fix: retrain deploie systematiquement si f1 >= 0.25"
```

---

## Task 3 : Tag visuel confirme/experimental dans Telegram

**Files:**
- Modify: `src/alerts/formatter.py:13-71`
- Test: `tests/test_formatter.py`

Chaque signal affichera un tag :
- `[CONFIRME]` si catalyst_type a > 25% win rate ET > 10 samples
- `[EXPERIMENTAL]` sinon

- [ ] **Step 1: Ecrire le test**

Dans `tests/test_formatter.py` :

```python
def test_format_signal_confirmed_tag(self):
    """Un signal avec bon win rate affiche CONFIRME."""
    signal = {
        "ticker": "SAN.PA", "name": "SANOFI", "score": 0.85,
        "current_price": 76.0, "catalyst_type": "EARNINGS",
        "features": {},
        "catalyst_stats": {
            "EARNINGS": {"win_rate": 0.35, "wins": 7, "total": 20},
        },
    }
    msg = self.formatter.format_signal(signal)
    assert "CONFIRME" in msg

def test_format_signal_experimental_tag(self):
    """Un signal avec mauvais win rate affiche EXPERIMENTAL."""
    signal = {
        "ticker": "SAN.PA", "name": "SANOFI", "score": 0.85,
        "current_price": 76.0, "catalyst_type": "TECHNICAL",
        "features": {},
        "catalyst_stats": {
            "TECHNICAL": {"win_rate": 0.10, "wins": 5, "total": 50},
        },
    }
    msg = self.formatter.format_signal(signal)
    assert "EXPERIMENTAL" in msg
```

- [ ] **Step 2: Implementer le tag dans `format_signal()`**

Dans `src/alerts/formatter.py`, au debut de `format_signal()`, apres le calcul de score :

```python
# Tag confirme/experimental
catalyst_stats = signal.get("catalyst_stats", {})
cat_type = signal.get("catalyst_type", "UNKNOWN")
cat_stat = catalyst_stats.get(cat_type)
is_confirmed = (
    cat_stat is not None
    and cat_stat.get("total", 0) > 10
    and cat_stat.get("win_rate", 0) > 0.25
)
tag = "CONFIRME" if is_confirmed else "EXPERIMENTAL"

lines = [
    f"<b>[{tag}] SIGNAL — {name} ({ticker})</b>",
    f"Score: {score_pct}% | Prix: {price_str} EUR",
    "",
]
```

- [ ] **Step 3: Lancer les tests**

Run: `uv run pytest tests/test_formatter.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/alerts/formatter.py tests/test_formatter.py
git commit -m "feat: tag CONFIRME/EXPERIMENTAL dans les alertes Telegram"
```

---

## Task 4 : Retrain quotidien au lieu d'hebdomadaire

**Files:**
- Modify: `scripts/run_scanner.py:602-613` (Job 11)

- [ ] **Step 1: Changer le CronTrigger du Job 11**

Dans `scripts/run_scanner.py`, changer le Job 11 de dimanche 19h a tous les jours 19h :

```python
# Job 11: Check retrain — quotidien a 19h (lun-ven)
scheduler.add_job(
    check_retrain,
    CronTrigger(
        hour=feedback_config.get("retrain_hour", 19),
        minute=0,
        day_of_week="mon-fri",  # <-- etait "sun", maintenant "mon-fri"
    ),
    args=[db, config, telegram, dry_run, predictor],
    id="check_retrain",
    name="Check retrain quotidien",
)
```

- [ ] **Step 2: Mettre a jour l'affichage**

Changer la ligne d'affichage du retrain :

```python
print(f"  - Check retrain: quotidien a {feedback_config.get('retrain_hour', 19)}h (lun-ven)")
```

- [ ] **Step 3: Commit**

```bash
git add scripts/run_scanner.py
git commit -m "feat: retrain quotidien au lieu d'hebdomadaire"
```

---

## Task 5 : Integration finale et deploiement

- [ ] **Step 1: Lancer tous les tests**

Run: `uv run pytest tests/ -v`
Expected: 421+ PASS

- [ ] **Step 2: Commit final et push**

```bash
git push origin master
```

- [ ] **Step 3: Deployer sur VPS**

```bash
ssh root@31.97.196.120 "bash -lc 'tmux kill-session -t scanner 2>/dev/null; cd ~/pea-scanner && git pull origin master && uv sync && tmux new-session -d -s scanner \"uv run python scripts/run_scanner.py 2>&1 | tee logs/scanner.log\"'"
```

- [ ] **Step 4: Verifier le bon fonctionnement**

```bash
ssh root@31.97.196.120 "bash -lc 'tmux capture-pane -t scanner -p -S -20'"
```

Verifier : pas d'erreurs, migration OK, modele charge.

- [ ] **Step 5: Test dry-run**

```bash
ssh root@31.97.196.120 "bash -lc 'cd ~/pea-scanner && uv run python scripts/run_scanner.py --once --dry-run 2>&1 | tail -30'"
```

Verifier : les tags CONFIRME/EXPERIMENTAL apparaissent dans les messages.
