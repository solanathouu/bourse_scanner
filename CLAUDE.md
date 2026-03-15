# PEA Scanner Bot

Bot Python de veille boursiere PEA. Analyse des donnees marche en temps reel, correlation avec les actualites, modele predictif ML entraine sur l'historique de trading personnel, alertes Telegram avec score de confiance.

## Commandes

```bash
uv run pytest tests/ -v          # Lancer tous les tests (405 tests)
uv run pytest tests/ -v -x       # Stopper au premier echec
uv run python scripts/init_db.py       # Initialiser la base SQLite
uv run python scripts/import_pdfs.py   # Importer les PDF dans la base
uv run python scripts/explore_pdf.py <fichier.pdf>  # Debug: voir le texte brut d'un PDF
uv run python scripts/collect_historical.py              # Collecter tout (prix + toutes news + fondamentaux)
uv run python scripts/collect_historical.py --prices     # Prix seulement (yfinance)
uv run python scripts/collect_historical.py --all-news   # Toutes les sources news
uv run python scripts/collect_historical.py --alphavantage  # Alpha Vantage (sentiment)
uv run python scripts/collect_historical.py --marketaux  # Marketaux (sentiment)
uv run python scripts/collect_historical.py --rss        # Flux RSS Google News FR
uv run python scripts/collect_historical.py --fundamentals  # Fondamentaux yfinance (PE, PB, analystes)
uv run python scripts/collect_historical.py --newsdata   # News Newsdata.io
uv run python scripts/collect_historical.py --delisted   # Prix tickers delistes (Boursorama)
uv run python scripts/score_sentiment.py           # Scorer sentiment LLM des news sans sentiment
uv run python scripts/score_sentiment.py --stats    # Distribution du sentiment
uv run python scripts/score_sentiment.py --dry-run  # Combien a scorer
uv run python scripts/import_prices_csv.py <fichier.csv> --ticker 2CRSI.PA  # Import prix CSV
uv run python scripts/match_catalysts.py           # Matcher trades <-> catalyseurs
uv run python scripts/match_catalysts.py --stats    # Stats catalyseurs seulement
uv run python scripts/train_model.py               # Entrainer le modele + evaluer
uv run python scripts/train_model.py --features     # Explorer les features
uv run python scripts/analyze_features.py           # Stats gagnants vs perdants
uv run python scripts/analyze_features.py --trade 42  # Features d'un trade specifique
uv run python scripts/analyze_trades_llm.py           # Analyser trades via Gemini
uv run python scripts/analyze_trades_llm.py --trade 42  # Analyser un trade specifique
uv run python scripts/analyze_trades_llm.py --stats     # Stats des analyses LLM
uv run python scripts/run_scanner.py           # Lancer le scanner (boucle infinie)
uv run python scripts/run_scanner.py --once     # Scorer une fois et quitter
uv run python scripts/run_scanner.py --dry-run  # Scorer sans envoyer de Telegram
uv run python scripts/run_feedback.py              # Review J+3 + mise a jour regles
uv run python scripts/run_feedback.py --stats       # Stats feedback loop (win rate, regles)
uv run python scripts/run_feedback.py --retrain     # Forcer re-entrainement modele
uv run python scripts/run_feedback.py --weekly      # Envoyer resume hebdomadaire
uv run python scripts/run_feedback.py --dry-run     # Review sans envoyer Telegram
```

## Architecture

```
pea-scanner/
├── src/                        # Code source (packages Python)
│   ├── core/                   # Couche fondation (BDD, logging, config)
│   ├── extraction/             # Module 1: Extraction PDF -> SQLite
│   ├── data_collection/        # Module 2: Collecte prix/news/fondamentaux (6 sources actives)
│   ├── analysis/               # Module 3: LLMAnalyzer, LLMSentimentScorer, NewsClassifier, TechnicalIndicators, FeatureEngine, CatalystMatcher
│   ├── model/                  # Module 4: Trainer (XGBoost), Evaluator, Predictor
│   ├── alerts/                 # Module 5: SignalFilter (adaptatif), AlertFormatter, TelegramBot
│   └── feedback/               # Module 6: SignalReviewer, PerformanceTracker, ModelRetrainer
├── scripts/                    # Points d'entree CLI
├── tests/                      # Tests (miroir de src/)
├── config/                     # Fichiers de configuration YAML
├── data/                       # Donnees (non versionne: PDFs, .db, models)
├── docs/plans/                 # Documents de design et plans d'implementation
└── logs/                       # Logs applicatifs (non versionne)
```

**Flux de donnees:**
```
PDF (one-shot) -> SQLite -> [Scheduler] -> Collecte donnees -> Feature engine -> ML scoring -> Filtrage -> Telegram
```

**Dependances entre modules (sens unique, jamais circulaire):**
```
core (fondation, 0 deps internes)
  <- extraction (utilise core.database)
  <- data_collection (utilise core.database, core.config)
  <- analysis (utilise core.database, data_collection)
  <- model (utilise core.database, analysis)
  <- alerts (utilise core.database, model)
  <- feedback (utilise core.database, model, analysis)
```

## Regles d'architecture

- Lire `.claude/rules/architecture.md` pour les regles completes
- Lire `.claude/rules/python-conventions.md` pour les conventions de code
- Lire `.claude/rules/testing.md` pour les regles de tests

## Stack technique

| Composant | Choix |
|-----------|-------|
| Python | 3.13 |
| Package manager | uv |
| BDD | SQLite (sqlite3 stdlib) |
| PDF | pdfplumber |
| Data | pandas, numpy |
| Prix | yfinance |
| News | gnews, Alpha Vantage API, Marketaux API, RSS (feedparser) |
| ML | xgboost, scikit-learn |
| LLM | google-genai (Gemini 2.5 Flash Lite) |
| Dashboard | FastAPI + Jinja2 + Chart.js (port 8050) |
| Indicateurs techniques | ta (RSI, MACD, Bollinger, ATR) |
| Scheduler | APScheduler 3.x |
| Telegram | python-telegram-bot 22.x |
| Logging | loguru |
| Config | pyyaml + python-dotenv |
| Tests | pytest |

## Base de donnees

SQLite dans `data/trades.db`. Tables principales (12 tables):

- `executions` — Avis d'execution PDF parses (1 ligne = 1 PDF)
- `trades_complets` — Trades reconstitues achat->vente (FIFO)
- `trade_catalyseurs` — News/events associes a chaque trade
- `trade_analyses_llm` — Analyses LLM des trades (catalyseur, raison achat/vente, qualite)
- `prices` — Prix OHLCV collectes
- `fundamentals` — Donnees fondamentales (PE, PB, consensus analystes, earnings date)
- `news` — Actualites collectees (avec sentiment LLM)
- `signals` — Signaux temps reel emis par le modele (ticker, date, score, signal_price, catalyseur, sent_at)
- `signal_reviews` — Reviews J+3 des signaux (performance, outcome WIN/LOSS/NEUTRAL, failure_reason)
- `filter_rules` — Regles de filtrage adaptatives generees par le feedback loop
- `model_versions` — Historique des versions du modele (metriques, is_active)
- `orderbook_snapshots` — Snapshots carnet d'ordres Boursorama (bid/ask, volumes, spread, ratio)

## Etapes d'implementation

| Etape | Statut | Description |
|-------|--------|-------------|
| 1 | DONE | Extraction PDF SG -> SQLite (214 PDF, 166 trades) |
| 2 | DONE | Collecte donnees historiques — 4 sources, 1357 prix, 1824 news |
| 3 | DONE | Correlation historique — catalyst_matcher, 649 associations, 113/166 trades |
| 4 | DONE | Feature engineering + ML — NewsClassifier, TechnicalIndicators, FeatureEngine, XGBoost Trainer, Evaluator |
| 4bis | DONE | LLM Trade Analysis — GPT-4o-mini analyse 141 trades, remplace regex par analyse profonde |
| 4ter | DONE | Enrichissement donnees — sentiment LLM, fondamentaux yfinance, Newsdata.io, RSS etendu, Boursorama delistes |
| 5 | DONE | Pipeline temps reel + alertes Telegram — Predictor, SignalFilter, TelegramBot, APScheduler, 30 valeurs |
| 6 | DONE | Tests integration, stabilisation, deploiement VPS |
| 7 | DONE | Feedback loop auto-apprenant — SignalReviewer J+3, PerformanceTracker, ModelRetrainer, filtrage adaptatif |

## Current Project State

| Aspect | Status | Details |
|--------|--------|---------|
| Code | Etape 1-7 DONE + refonte feedback loop 15 mars | LLM news classifier, feedback features, TP system, regime marche, modele simplifie, data leakage corrige. Dashboard FastAPI. |
| Config | .env configure | ALPHA_VANTAGE_API_KEY + MARKETAUX_API_KEY + GEMINI_API_KEY + NEWSDATA_API_KEY + TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID |
| Tests | 434/434 PASS | +16 llm_news_classifier, +4 feedback_features, +2 market_regime, +3 formatter, +4 signal_reviewer_tp |
| Data VPS | 216 signaux, 157 reviews, 7570 news, 2756+ prix, 13168 orderbook | Win rate global: 16.6% (26W 94L 37N). Modele v3 actif (f1=0.316). |
| Dashboard | http://31.97.196.120:8050 | 4 pages: accueil, news, signal, portfolio. Lecture seule. |
| Docs | Plans + design specs | docs/plans/2026-03-15-*.md |
| Git | Pushe sur origin | master a jour, VPS synchro |

## Sources de donnees actives

| Source | Fichier | Articles | Sentiment | Cle API |
|--------|---------|----------|-----------|---------|
| yfinance | price_collector.py | 1357+ prix | - | Non |
| GNews | news_collector.py | 412 | LLM | Non |
| Alpha Vantage | alpha_vantage_collector.py | 700 | Oui (natif) | Oui (.env) |
| Marketaux | marketaux_collector.py | 43 | Oui (natif) | Oui (.env) |
| RSS Google News FR | rss_collector.py | 669+ | LLM | Non |
| Newsdata.io | newsdata_collector.py | nouveau | Oui (natif) | Oui (.env) |
| Boursorama (delistes) | boursorama_scraper.py | prix | - | Non |
| yfinance fondamentaux | fundamental_collector.py | PE/PB/analysts | - | Non |
| LLM Sentiment | llm_sentiment.py | scorer ~1081 news | Gemini 2.5 Flash Lite | Oui (.env) |
| LLM News Classifier | llm_news_classifier.py | catalyseur + pertinence | Gemini 2.5 Flash Lite | Oui (.env) |
| Boursorama Orderbook | orderbook_collector.py | carnet d'ordres | - | Non |
| CAC40 Index | price_collector.py (^FCHI) | regime marche | - | Non |

**Tickers sans prix yfinance (delistes, fallback Boursorama):** 2CRSI.PA, ALTBG.PA, AFYREN.PA

## PHILOSOPHIE FONDAMENTALE DU PROJET

**LE BOT DOIT TRADER COMME NICOLAS TRADE.** C'est le coeur absolu du projet.

Le but n'est PAS de creer un bot generique qui agregue des donnees internet et les recrache.
Le but EST de creer un algorithme d'intelligence qui COMPREND le style de trading personnel
de Nicolas en analysant ses 166 trades historiques, et qui REPRODUIT sa logique de decision.

### Ce que ca veut dire concretement

Pour CHAQUE trade historique, l'algorithme doit:
1. **Comprendre POURQUOI Nicolas a achete** — Quel catalyseur l'a pousse a entrer en position?
   (annonce de bilan, upgrade analyste, news positive, rebond technique, volume anormal...)
2. **Comprendre POURQUOI Nicolas a vendu** — Objectif atteint? Stop loss? Changement de contexte?
3. **Extraire les PATTERNS RECURRENTS** — Quels types de catalyseurs declenchent les meilleurs trades?
   Quelles conditions techniques accompagnent ses entrees gagnantes?
4. **Construire un profil de trading** — "Nicolas achete quand [conditions], il vend quand [conditions]"

### L'etape 4 est LA CLE de tout le projet

L'etape 4 (Feature Engineering + ML) doit:
- Matcher FINEMENT chaque trade avec les news/events de l'epoque (pas un simple score temporel)
- Classifier les catalyseurs par type (EARNINGS, UPGRADE, FDA_APPROVAL, CONTRAT, etc.)
- Apprendre les patterns: "quand il y a une annonce de bilan + RSI < 40 -> Nicolas achete et gagne X%"
- Le modele ML doit etre entraine sur les DECISIONS de Nicolas, pas sur des regles generiques
- Le score de confiance final doit refleter "est-ce que Nicolas prendrait ce trade?"

### Anti-patterns a eviter absolument

- NE PAS faire un bot generique "RSI < 30 = acheter"
- NE PAS scorer les news avec un simple matching temporel sans comprendre le CONTENU
- NE PAS ignorer le contexte technique au moment de l'achat
- NE PAS traiter tous les trades de la meme facon (un trade sur annonce de bilan ≠ un trade technique)
- NE PAS oublier que c'est la COMBINAISON catalyseur + conditions techniques qui fait le trade

## Changements recents (mars 2026)

**15 mars — Refonte majeure du feedback loop + dashboard:**

1. **Migration OpenAI -> Gemini 2.5 Flash Lite**: Modele LLM moins gourmand en tokens.
2. **LLM News Classifier**: Remplace le regex NewsClassifier (73% UNKNOWN) par Gemini qui comprend semantiquement les news. Classifie le catalyseur + score de pertinence (filtre le bruit RSS). Cache en BDD.
3. **Feedback loop auto-apprenant**:
   - 3 features feedback injectees dans le modele: catalyst_historical_win_rate, catalyst_historical_sample_size, ticker_historical_win_rate
   - Retrain quotidien (lun-ven 19h) au lieu d'hebdomadaire
   - Deploiement si f1 >= 0.25 (plus de comparaison biaisee old/new)
   - Tags [CONFIRME] / [EXPERIMENTAL] dans les alertes Telegram
4. **Take Profit**: Review regarde le HIGH J+0 a J+3. Si high touche +4.5% = WIN (meme si le close redescend).
5. **Regime marche**: Features CAC40 (trend SMA20, RSI marche, variation 5j) pour que le modele sache si on est en bull ou bear market.
6. **Modele simplifie**: max_depth=2, regularisation (subsample, colsample, reg_lambda, min_child_weight). Evite l'overfitting sur 268 samples.
7. **Data leakage corrige**: Le prompt LLM d'analyse des trades ne recoit plus le rendement ni le resultat.
8. **Dashboard FastAPI**: http://31.97.196.120:8050 — 4 pages (accueil, news, signal detail, portfolio virtuel). Lecture seule, dark theme.
9. **Tracking par version**: Chaque signal enregistre la version du modele. Portfolio filtrable par version.
10. **Modele v3 deploye**: f1=0.316, precision=37.5%, accuracy=84%. Entraine sur 268 samples (111 trades + 157 reviews).

**Regles absolues du projet:**
- Ne JAMAIS supprimer de donnees d'entrainement (meme le bruit RSS sert au modele)
- Envoyer TOUS les signaux sur Telegram (ne filtrer que quand le win rate sera assez haut)
- Le dashboard est en LECTURE SEULE, aucune influence sur l'algorithme

## Next Immediate Action

**Laisser le bot tourner et surveiller via le dashboard.**

Le bot tourne sur le VPS avec le modele v3. Le feedback loop est autonome:
1. **Lundi 17 mars**: Premiers signaux v3 emis pendant les heures de marche
2. **Jeudi 20 mars**: Premieres reviews J+3 des signaux v3 (avec systeme TP)
3. **Chaque soir 19h**: Retrain quotidien automatique (v4, v5, ...)
4. **Dashboard**: http://31.97.196.120:8050 pour suivre en temps reel
5. **Ne rien filtrer** tant que le win rate n'est pas assez haut (decision Nicolas)

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
- **IMPORTANT**: Nicolas ne prend pas des trades au hasard. Chaque achat est motive par un
  catalyseur specifique (news, annonce, signal technique). Le modele doit apprendre CES motivations.
