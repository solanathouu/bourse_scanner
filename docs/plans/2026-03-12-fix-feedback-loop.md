# Fix Feedback Loop — Bot Auto-Apprenant

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corriger les 5 bugs qui empechent le bot d'apprendre de ses erreurs et de s'ameliorer tout seul.

**Architecture:** Le feedback loop est en place (signal_reviewer, performance_tracker, model_retrainer) mais 5 composants sont casses en production : (1) orderbook parser mauvais format, (2) retrain crash KeyError date_achat, (3) walk-forward split_date hardcode, (4) catalyst_news_title toujours None, (5) retrain ne recharge pas le modele dans le predictor en cours. On corrige chaque bug avec TDD.

**Tech Stack:** Python 3.13, pytest, XGBoost, SQLite, Boursorama scraping

---

## Diagnostic des 5 bugs

| # | Bug | Cause racine | Impact |
|---|-----|-------------|--------|
| 1 | Orderbook features toujours 0.0 | `_parse_orderbook()` cherche `data["bids"]` mais Boursorama renvoie `data["orderbook"]["lines"]` avec format `{bid, bidSize, bidNb, ask, askSize, askNb}` | 6943 snapshots inutiles, 4 features mortes |
| 2 | Retrain crash `KeyError: date_achat` | `build_realtime_features()` ne retourne pas `date_achat` dans le dict. Quand les reviews sont combinees pour le retrain, la colonne manque | Retrain n'a JAMAIS tourne (crash le 8 mars) |
| 3 | Walk-forward split_date hardcode | `split_date="2025-12-01"` dans `trainer.py`. Pour 2026, tous les trades+reviews sont dans le test set | Metriques de validation absurdes |
| 4 | `catalyst_news_title` toujours None | Hardcode `None` dans `predictor.py:65` | Alertes Telegram sans titre de catalyseur |
| 5 | Predictor ne recharge pas apres retrain | Apres un retrain reussi, le scanner continue avec l'ancien modele en memoire | Le nouveau modele ne sert pas |

---

## Chunk 1: File Structure

### Fichiers a modifier

| Fichier | Responsabilite | Modification |
|---------|---------------|-------------|
| `src/data_collection/orderbook_collector.py:104-157` | Parse orderbook Boursorama | Corriger `_parse_orderbook()` pour format `orderbook.lines` |
| `src/analysis/feature_engine.py:350-407` | Features temps reel | Ajouter `date_achat` au retour de `build_realtime_features()` |
| `src/analysis/feature_engine.py:298-348` | Features combinees | Fallback date si `signal_date` vide |
| `src/model/trainer.py:110-172` | Walk-forward validation | Split date dynamique au lieu de hardcode |
| `src/model/predictor.py:27-73` | Scoring temps reel | Remonter `catalyst_news_title` depuis best news |
| `src/model/predictor.py:17-26` | Init predictor | Ajouter methode `reload_model()` |
| `scripts/run_scanner.py:242-263` | Job retrain | Recharger le predictor apres retrain reussi |
| `tests/test_orderbook_collector.py` | Tests orderbook | Ajouter test nouveau format |
| `tests/test_feature_engine.py` | Tests features | Test date_achat dans realtime features |
| `tests/test_trainer.py` | Tests trainer | Test split_date dynamique |
| `tests/test_predictor.py` | Tests predictor | Test catalyst_news_title |

---

### Task 1: Fix orderbook parser (format Boursorama)

**Files:**
- Modify: `src/data_collection/orderbook_collector.py:104-157`
- Test: `tests/test_orderbook_collector.py`

Le vrai format JSON Boursorama est:
```json
{
  "symbol": "...",
  "orderbook": {
    "lines": [
      {"askNb": 5, "askSize": 6600, "ask": 1.5035, "bidNb": 3, "bidSize": 6800, "bid": 1.4705},
      {"askNb": 2, "askSize": 3000, "ask": 1.5100, "bidNb": 4, "bidSize": 4200, "bid": 1.4650}
    ]
  }
}
```

Le parser actuel cherche `data["bids"]` et `data["asks"]` qui n'existent pas.

- [ ] **Step 1: Ecrire le test qui echoue**

Dans `tests/test_orderbook_collector.py`, ajouter:
```python
def test_parse_orderbook_real_format(self):
    """Test parsing du vrai format Boursorama (orderbook.lines)."""
    data = {
        "symbol": "1rPSAN",
        "orderbook": {
            "lines": [
                {"askNb": 5, "askSize": 6600, "ask": 76.50, "bidNb": 3, "bidSize": 6800, "bid": 76.25},
                {"askNb": 2, "askSize": 3000, "ask": 76.60, "bidNb": 4, "bidSize": 4200, "bid": 76.20},
                {"askNb": 1, "askSize": 1500, "ask": 76.70, "bidNb": 2, "bidSize": 2000, "bid": 76.15},
            ]
        }
    }
    result = self.collector._parse_orderbook(data)
    assert result["best_bid"] == 76.25
    assert result["best_ask"] == 76.50
    assert result["bid_volume_total"] == 13000  # 6800+4200+2000
    assert result["ask_volume_total"] == 11100  # 6600+3000+1500
    assert result["bid_orders_total"] == 9  # 3+4+2
    assert result["ask_orders_total"] == 8  # 5+2+1
    assert result["spread_pct"] > 0
    assert result["bid_ask_volume_ratio"] > 1  # Plus d'acheteurs
```

- [ ] **Step 2: Lancer le test, verifier qu'il echoue**

Run: `uv run pytest tests/test_orderbook_collector.py::TestOrderBookCollector::test_parse_orderbook_real_format -v`
Expected: FAIL (best_bid=0, best_ask=0)

- [ ] **Step 3: Corriger `_parse_orderbook()`**

Remplacer `_parse_orderbook` dans `orderbook_collector.py:104-157` par:
```python
def _parse_orderbook(self, data: dict) -> dict:
    """Extrait les metriques du carnet d'ordres.

    Supporte le format Boursorama: data["orderbook"]["lines"]
    ou chaque ligne a {bid, bidSize, bidNb, ask, askSize, askNb}.
    """
    lines = []
    orderbook = data.get("orderbook", {})
    if isinstance(orderbook, dict):
        lines = orderbook.get("lines", [])

    # Fallback ancien format (bids/asks separees)
    if not lines:
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        best_bid = bids[0].get("price", 0) if bids else 0
        best_ask = asks[0].get("price", 0) if asks else 0
        bid_volumes = [b.get("quantity", 0) or 0 for b in bids]
        ask_volumes = [a.get("quantity", 0) or 0 for a in asks]
        bid_orders = [b.get("orders", 0) or 0 for b in bids]
        ask_orders = [a.get("orders", 0) or 0 for a in asks]
    else:
        # Format Boursorama: orderbook.lines
        best_bid = lines[0].get("bid", 0) or 0 if lines else 0
        best_ask = lines[0].get("ask", 0) or 0 if lines else 0
        bid_volumes = [ln.get("bidSize", 0) or 0 for ln in lines]
        ask_volumes = [ln.get("askSize", 0) or 0 for ln in lines]
        bid_orders = [ln.get("bidNb", 0) or 0 for ln in lines]
        ask_orders = [ln.get("askNb", 0) or 0 for ln in lines]

    bid_volume_total = sum(bid_volumes)
    ask_volume_total = sum(ask_volumes)
    bid_orders_total = sum(bid_orders)
    ask_orders_total = sum(ask_orders)

    spread_pct = 0.0
    if best_bid > 0 and best_ask > 0:
        spread_pct = round((best_ask - best_bid) / best_bid * 100, 4)

    bid_ask_volume_ratio = 0.0
    if ask_volume_total > 0:
        bid_ask_volume_ratio = round(
            bid_volume_total / ask_volume_total, 4,
        )

    bid_depth_concentration = 0.0
    if bid_volume_total > 0 and len(bid_volumes) >= 3:
        top3 = sum(bid_volumes[:3])
        bid_depth_concentration = round(top3 / bid_volume_total, 4)
    elif bid_volume_total > 0:
        bid_depth_concentration = 1.0

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "bid_volume_total": bid_volume_total,
        "ask_volume_total": ask_volume_total,
        "bid_orders_total": bid_orders_total,
        "ask_orders_total": ask_orders_total,
        "spread_pct": spread_pct,
        "bid_ask_volume_ratio": bid_ask_volume_ratio,
        "bid_depth_concentration": bid_depth_concentration,
    }
```

- [ ] **Step 4: Aussi corriger `_build_orderbook_features` dans `feature_engine.py:489-527`**

Le `_build_orderbook_features` parse aussi `raw_json` pour `bid_depth_concentration`. Il faut aligner sur le nouveau format:
```python
def _build_orderbook_features(self, ticker: str) -> dict:
    """Construit les features du carnet d'ordres."""
    default = {f: 0.0 for f in ORDERBOOK_FEATURES}

    snapshot = self.db.get_latest_orderbook(ticker)
    if snapshot is None:
        return default

    return {
        "bid_ask_volume_ratio": snapshot.get("bid_ask_volume_ratio") or 0.0,
        "bid_ask_order_ratio": self._compute_order_ratio(snapshot),
        "spread_pct": snapshot.get("spread_pct") or 0.0,
        "bid_depth_concentration": snapshot.get("bid_depth_concentration") or 0.0,
    }

def _compute_order_ratio(self, snapshot: dict) -> float:
    """Calcule le ratio bid/ask orders depuis le snapshot."""
    bid_orders = snapshot.get("bid_orders_total") or 0
    ask_orders = snapshot.get("ask_orders_total") or 0
    if ask_orders > 0:
        return round(bid_orders / ask_orders, 4)
    return 0.0
```

Note: `bid_depth_concentration` est maintenant calcule et stocke par `_parse_orderbook` directement, plus besoin de parser `raw_json` dans feature_engine.

- [ ] **Step 5: Lancer les tests orderbook**

Run: `uv run pytest tests/test_orderbook_collector.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/data_collection/orderbook_collector.py src/analysis/feature_engine.py tests/test_orderbook_collector.py
git commit -m "fix(orderbook): parse real Boursorama format (orderbook.lines)"
```

---

### Task 2: Fix retrain crash — ajouter `date_achat` aux features temps reel

**Files:**
- Modify: `src/analysis/feature_engine.py:350-407`
- Modify: `src/analysis/feature_engine.py:298-348`
- Test: `tests/test_feature_engine.py`

Le probleme: `build_realtime_features()` retourne un dict SANS `date_achat`. Quand ce dict est serialise en `features_json` dans un signal, puis repris par `build_combined_features()` pour le retrain, la colonne `date_achat` manque et `walk_forward_validate()` crash.

- [ ] **Step 1: Ecrire le test qui echoue**

Dans `tests/test_feature_engine.py`, ajouter:
```python
def test_realtime_features_include_date_achat(self):
    """build_realtime_features doit inclure date_achat pour le retrain."""
    # Setup: inserer des prix pour un ticker
    prices = []
    from datetime import datetime, timedelta
    base = datetime(2026, 1, 1)
    for i in range(30):
        d = (base + timedelta(days=i)).strftime("%Y-%m-%d")
        prices.append({
            "ticker": "SAN.PA", "date": d,
            "open": 75 + i * 0.1, "high": 76 + i * 0.1,
            "low": 74 + i * 0.1, "close": 75.5 + i * 0.1,
            "volume": 100000,
        })
    self.db.insert_prices_batch(prices)

    features = self.engine.build_realtime_features("SAN.PA", 78.5, "2026-01-25")
    assert features is not None
    assert "date_achat" in features
    assert features["date_achat"] == "2026-01-25"
```

- [ ] **Step 2: Lancer le test, verifier qu'il echoue**

Run: `uv run pytest tests/test_feature_engine.py::TestBuildRealtimeFeatures::test_realtime_features_include_date_achat -v`
Expected: FAIL (KeyError: 'date_achat')

- [ ] **Step 3: Ajouter `date_achat` au retour de `build_realtime_features()`**

Dans `feature_engine.py`, modifier le return de `build_realtime_features()` (lignes 401-407):
```python
return {
    "date_achat": date,  # AJOUTER CETTE LIGNE
    **tech_features,
    **cat_features,
    **fund_features,
    **ctx_features,
    **ob_features,
}
```

- [ ] **Step 4: Securiser `build_combined_features()` pour signal_date vide**

Dans `feature_engine.py`, ligne 328, remplacer:
```python
features["date_achat"] = review.get("signal_date", "")
```
Par:
```python
features["date_achat"] = review.get("signal_date") or date
```
Ou `date` est la date du signal. En pratique, utiliser un fallback solide:
```python
signal_date = review.get("signal_date")
if not signal_date:
    signal_date = "2025-01-01"
features["date_achat"] = signal_date
```

- [ ] **Step 5: Lancer les tests feature_engine**

Run: `uv run pytest tests/test_feature_engine.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/analysis/feature_engine.py tests/test_feature_engine.py
git commit -m "fix(feature_engine): add date_achat to realtime features for retrain"
```

---

### Task 3: Fix walk-forward split_date dynamique

**Files:**
- Modify: `src/model/trainer.py:110-172`
- Test: `tests/test_trainer.py`

Le probleme: `split_date="2025-12-01"` est hardcode. Pour des reviews en 2026, tout le dataset est dans le test set, le train set est vide, le retrain echoue ou donne des metriques absurdes.

- [ ] **Step 1: Ecrire le test qui echoue**

Dans `tests/test_trainer.py`, ajouter:
```python
def test_walk_forward_auto_split(self):
    """Walk-forward avec split_date=None doit calculer un split automatique."""
    import pandas as pd
    # Dataset avec dates de mai 2025 a mars 2026
    rows = []
    for i in range(40):
        month = 5 + i // 5
        year = 2025 + (month - 1) // 12
        month = ((month - 1) % 12) + 1
        rows.append({
            "date_achat": f"{year}-{month:02d}-{(i % 28) + 1:02d}",
            "rsi_14": 40 + i, "macd_histogram": 0.1,
            "bollinger_position": 0.5, "range_position_10": 0.5,
            "range_position_20": 0.5, "range_amplitude_10": 10,
            "range_amplitude_20": 15, "volume_ratio_20": 1.0,
            "atr_14_pct": 3.0, "variation_1j": 0.5,
            "variation_5j": 1.0, "distance_sma20": 0,
            "distance_sma50": 0, "catalyst_type": "TECHNICAL",
            "catalyst_confidence": 0.5, "news_sentiment": 0.0,
            "has_clear_catalyst": 1, "buy_reason_length": 50,
            "pe_ratio": 15, "pb_ratio": 2, "target_upside_pct": 10,
            "analyst_count": 5, "days_to_earnings": 30,
            "recommendation_score": 3, "day_of_week": i % 5,
            "nb_previous_trades": 2, "previous_win_rate": 0.8,
            "days_since_last_trade": 20,
            "bid_ask_volume_ratio": 0, "bid_ask_order_ratio": 0,
            "spread_pct": 0, "bid_depth_concentration": 0,
            "is_winner": 1 if i % 3 != 0 else 0,
            "trade_id": i + 1,
        })
    df = pd.DataFrame(rows)

    trainer = Trainer()
    result = trainer.walk_forward_validate(df, split_date=None)
    assert "error" not in result
    assert result["train_size"] > 0
    assert result["test_size"] > 0
```

- [ ] **Step 2: Lancer le test, verifier qu'il echoue**

Run: `uv run pytest tests/test_trainer.py::TestTrainer::test_walk_forward_auto_split -v`
Expected: FAIL ou resultats absurdes (train_size=0)

- [ ] **Step 3: Corriger `walk_forward_validate`**

Dans `trainer.py`, modifier la signature et le debut de `walk_forward_validate` (lignes 110-124):
```python
def walk_forward_validate(self, features_df: pd.DataFrame,
                           split_date: str | None = None) -> dict:
    """Validation walk-forward: train avant split_date, test apres.

    Si split_date est None, utilise le 75e percentile des dates
    pour avoir ~75% train / ~25% test.
    """
    df = features_df.copy()

    if split_date is None:
        if "date_achat" not in df.columns or df["date_achat"].isna().all():
            return {"error": "Pas de colonne date_achat"}
        dates = df["date_achat"].dropna().sort_values()
        if len(dates) < 10:
            return {"error": "Pas assez de donnees pour split"}
        split_date = dates.iloc[int(len(dates) * 0.75)]

    train_mask = df["date_achat"] < split_date
    test_mask = df["date_achat"] >= split_date
    # ... reste identique
```

- [ ] **Step 4: Lancer les tests trainer**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/trainer.py tests/test_trainer.py
git commit -m "fix(trainer): dynamic split_date for walk-forward validation"
```

---

### Task 4: Remonter catalyst_news_title dans le predictor

**Files:**
- Modify: `src/model/predictor.py:27-73`
- Modify: `src/analysis/feature_engine.py:409-449`
- Test: `tests/test_predictor.py`

Le probleme: `catalyst_news_title` est hardcode a `None` dans predictor.py:65. On doit remonter le titre de la meilleure news identifiee par `_build_realtime_catalyst_features`.

- [ ] **Step 1: Modifier `_build_realtime_catalyst_features` pour retourner le titre**

Dans `feature_engine.py`, modifier `_build_realtime_catalyst_features` pour inclure `best_news_title`:
```python
def _build_realtime_catalyst_features(self, ticker: str, date: str) -> dict:
    """Construit les features catalyseur a partir des news recentes en BDD."""
    from datetime import timedelta

    dt = datetime.strptime(date, "%Y-%m-%d")
    date_start = (dt - timedelta(days=5)).strftime("%Y-%m-%d")

    news_list = self.db.get_news_in_window(ticker, date_start, date)

    if not news_list:
        return {
            "catalyst_type": "TECHNICAL",
            "catalyst_confidence": 0.0,
            "news_sentiment": 0.0,
            "has_clear_catalyst": 0,
            "buy_reason_length": 0,
            "best_news_title": None,
        }

    classifier = NewsClassifier()
    classified = classifier.classify_batch(news_list)

    best = max(classified, key=lambda n: n.get("sentiment") or 0.0)
    cat_type = best.get("catalyst_type", "UNKNOWN")

    sentiments = [n.get("sentiment") or 0.0 for n in news_list]
    avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

    has_catalyst = 1 if cat_type not in ("TECHNICAL", "UNKNOWN") else 0

    return {
        "catalyst_type": cat_type,
        "catalyst_confidence": 0.5 if has_catalyst else 0.0,
        "news_sentiment": round(avg_sentiment, 4),
        "has_clear_catalyst": has_catalyst,
        "buy_reason_length": len(best.get("title", "")),
        "best_news_title": best.get("title"),
    }
```

- [ ] **Step 2: Utiliser `best_news_title` dans predictor**

Dans `predictor.py`, modifier `score_ticker` (ligne 65):
```python
return {
    "ticker": ticker,
    "date": date,
    "score": round(score, 4),
    "current_price": round(current_price, 2),
    "catalyst_type": features.get("catalyst_type", "UNKNOWN"),
    "catalyst_news_title": features.pop("best_news_title", None),
    "features": features,
    "features_json": json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v
         for k, v in features.items()
         if k not in ("catalyst_type", "best_news_title")}
    ),
    "technical_summary": self._build_technical_summary(features),
}
```

- [ ] **Step 3: Lancer les tests predictor et feature_engine**

Run: `uv run pytest tests/test_predictor.py tests/test_feature_engine.py -v`
Expected: PASS (adapter les tests existants si besoin)

- [ ] **Step 4: Commit**

```bash
git add src/model/predictor.py src/analysis/feature_engine.py tests/test_predictor.py
git commit -m "feat(predictor): extract catalyst_news_title from best matching news"
```

---

### Task 5: Recharger le modele apres retrain + fix du scanner

**Files:**
- Modify: `src/model/predictor.py:17-26`
- Modify: `scripts/run_scanner.py:242-263`
- Test: `tests/test_predictor.py`

Le probleme: Apres un retrain reussi (dimanche 19h), le modele est sauvegarde sur disque mais le `Predictor` en memoire continue d'utiliser l'ancien modele. Il faut recharger.

- [ ] **Step 1: Ajouter `reload_model()` au Predictor**

Dans `predictor.py`, ajouter apres `__init__`:
```python
def reload_model(self, model_path: str | None = None):
    """Recharge le modele depuis le disque.

    Utile apres un retrain pour utiliser le nouveau modele
    sans redemarrer le scanner.
    """
    if model_path is None:
        model_path = self._model_path
    self.trainer.load_model(model_path)
    # Vider le cache de prix pour repartir frais
    self.engine._price_cache.clear()
    logger.info(f"Modele recharge: {model_path}")
```

Et dans `__init__`, sauvegarder le path:
```python
def __init__(self, db: Database,
             model_path: str = "data/models/nicolas_v1.joblib"):
    self.db = db
    self._model_path = model_path
    self.engine = FeatureEngine(db)
    self.trainer = Trainer()
    self.trainer.load_model(model_path)
```

- [ ] **Step 2: Modifier `check_retrain` dans `run_scanner.py`**

Dans `run_scanner.py`, modifier `check_retrain` pour recharger le predictor:
```python
def check_retrain(db, config, telegram, dry_run, predictor=None):
    """Verifie si re-entrainement necessaire et execute."""
    from src.feedback.model_retrainer import ModelRetrainer

    feedback_cfg = config.get("feedback", {})
    model_path = config.get("scoring", {}).get("model_path", "data/models/nicolas_v1.joblib")

    retrainer = ModelRetrainer(
        db, min_reviews_for_retrain=feedback_cfg.get("min_reviews_retrain", 20)
    )

    if not retrainer.should_retrain():
        logger.info("Pas assez de reviews pour re-entrainer")
        return

    result = retrainer.retrain_with_validation(model_path)
    report = retrainer.format_retrain_report(result)

    if result.get("deployed") and predictor is not None:
        new_path = result.get("new_path", model_path)
        predictor.reload_model(new_path)
        logger.info(f"Predictor recharge avec {new_path}")

    if telegram and not dry_run:
        telegram.send_alert_sync(report)

    logger.info(f"Re-entrainement: deployed={result['deployed']}")
```

- [ ] **Step 3: Passer `predictor` au job 11 dans le scheduler**

Dans `run_scheduler()`, modifier le job 11 (lignes 598-608):
```python
# Job 11: Check retrain — dimanche 19h
scheduler.add_job(
    check_retrain,
    CronTrigger(
        hour=feedback_config.get("retrain_hour", 19),
        minute=0,
        day_of_week=feedback_config.get("retrain_day", "sun"),
    ),
    args=[db, config, telegram, dry_run, predictor],
    id="check_retrain",
    name="Check retrain",
)
```

- [ ] **Step 4: Lancer tous les tests**

Run: `uv run pytest tests/ -v`
Expected: Tous les 404+ tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/model/predictor.py scripts/run_scanner.py tests/test_predictor.py
git commit -m "fix(scanner): reload model after successful retrain"
```

---

### Task 6: Deployer sur le VPS et forcer le premier retrain

**Files:**
- Aucun fichier a modifier (deploiement)

- [ ] **Step 1: Pousser le code**

```bash
git push origin master
```

- [ ] **Step 2: Pull sur le VPS**

```bash
ssh root@31.97.196.120 "cd ~/pea-scanner && git pull"
```

- [ ] **Step 3: Arreter le scanner**

```bash
ssh root@31.97.196.120 "tmux send-keys -t scanner C-c"
```

- [ ] **Step 4: Verifier la cle API OpenAI**

```bash
ssh root@31.97.196.120 "cd ~/pea-scanner && grep OPENAI .env | head -c 30"
```
Expected: La nouvelle cle (sk-proj-_9BZn...)

- [ ] **Step 5: Forcer le retrain manuellement**

```bash
ssh root@31.97.196.120 "cd ~/pea-scanner && bash -lc 'uv run python scripts/run_feedback.py --retrain'"
```
Expected:
- "Dataset combine: X trades + Y reviews = Z samples"
- "Nouveau modele deploye: v2 (f1 X -> Y)" OU "Ancien modele conserve"

- [ ] **Step 6: Tester le scoring avec le nouveau modele**

```bash
ssh root@31.97.196.120 "cd ~/pea-scanner && bash -lc 'uv run python scripts/run_scanner.py --once --dry-run'"
```
Expected: Les scores doivent etre plus varies (pas tous > 0.85)

- [ ] **Step 7: Redemarrer le scanner**

```bash
ssh root@31.97.196.120 "tmux send-keys -t scanner 'cd ~/pea-scanner && bash -lc \"uv run python scripts/run_scanner.py\"' Enter"
```

- [ ] **Step 8: Verifier que le scanner tourne**

```bash
ssh root@31.97.196.120 "tmux capture-pane -t scanner -p | tail -10"
```
Expected: "PEA Scanner demarre" + 13 jobs programmes

---

## Resume des changements

Apres ces corrections, le bot aura un vrai cycle d'apprentissage:

```
Signal emis -> J+3 review (WIN/LOSS/NEUTRAL)
                    |
                    v
           signal_reviews table (134+ reviews)
                    |
                    v
           Dimanche 19h: check_retrain
                    |
           build_combined_features()
           = trades historiques + reviews
                    |
           walk_forward_validate (split dynamique)
                    |
           Si f1 +1% -> deployer nouveau modele
                    |
           predictor.reload_model()
                    |
           Lundi 9h30: scoring avec nouveau modele
           (scores plus discriminants)
```

Le bot va enfin apprendre que 62% de ses signaux actuels sont des LOSS et ajuster ses scores a la baisse.
