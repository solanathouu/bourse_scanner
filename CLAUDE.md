# PEA Scanner Bot

Bot Python de veille boursiere PEA. Analyse des donnees marche en temps reel, correlation avec les actualites, modele predictif ML entraine sur l'historique de trading personnel, alertes Telegram avec score de confiance.

## Commandes

```bash
uv run pytest tests/ -v          # Lancer tous les tests (183 tests)
uv run pytest tests/ -v -x       # Stopper au premier echec
uv run python scripts/init_db.py       # Initialiser la base SQLite
uv run python scripts/import_pdfs.py   # Importer les PDF dans la base
uv run python scripts/explore_pdf.py <fichier.pdf>  # Debug: voir le texte brut d'un PDF
uv run python scripts/collect_historical.py              # Collecter tout (prix + toutes news)
uv run python scripts/collect_historical.py --prices     # Prix seulement (yfinance)
uv run python scripts/collect_historical.py --all-news   # Toutes les sources news
uv run python scripts/collect_historical.py --alphavantage  # Alpha Vantage (sentiment)
uv run python scripts/collect_historical.py --marketaux  # Marketaux (sentiment)
uv run python scripts/collect_historical.py --rss        # Flux RSS Google News FR
uv run python scripts/match_catalysts.py           # Matcher trades <-> catalyseurs
uv run python scripts/match_catalysts.py --stats    # Stats catalyseurs seulement
uv run python scripts/train_model.py               # Entrainer le modele + evaluer
uv run python scripts/train_model.py --features     # Explorer les features
uv run python scripts/analyze_features.py           # Stats gagnants vs perdants
uv run python scripts/analyze_features.py --trade 42  # Features d'un trade specifique
uv run python scripts/analyze_trades_llm.py           # Analyser trades via GPT-4o-mini
uv run python scripts/analyze_trades_llm.py --trade 42  # Analyser un trade specifique
uv run python scripts/analyze_trades_llm.py --stats     # Stats des analyses LLM
```

## Architecture

```
pea-scanner/
├── src/                        # Code source (packages Python)
│   ├── core/                   # Couche fondation (BDD, logging, config)
│   ├── extraction/             # Module 1: Extraction PDF -> SQLite
│   ├── data_collection/        # Module 2: Collecte prix/news (4 sources actives)
│   ├── analysis/               # Module 3: LLMAnalyzer, NewsClassifier, TechnicalIndicators, FeatureEngine, CatalystMatcher
│   ├── model/                  # Module 4: Trainer (XGBoost), Evaluator
│   └── alerts/                 # Module 5: Telegram (a venir)
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
| LLM | openai (GPT-4o-mini) |
| Indicateurs techniques | ta (RSI, MACD, Bollinger, ATR) |
| Scheduler | APScheduler |
| Telegram | python-telegram-bot |
| Logging | loguru |
| Config | pyyaml + python-dotenv |
| Tests | pytest |

## Base de donnees

SQLite dans `data/trades.db`. Tables principales:

- `executions` — Avis d'execution PDF parses (1 ligne = 1 PDF)
- `trades_complets` — Trades reconstitues achat->vente (FIFO)
- `trade_catalyseurs` — News/events associes a chaque trade
- `trade_analyses_llm` — Analyses LLM des trades (catalyseur, raison achat/vente, qualite)
- `prices` — Prix OHLCV collectes
- `indicators` — Indicateurs techniques calcules
- `news` — Actualites collectees
- `signals` — Signaux emis par le modele

## Etapes d'implementation

| Etape | Statut | Description |
|-------|--------|-------------|
| 1 | DONE | Extraction PDF SG -> SQLite (214 PDF, 166 trades) |
| 2 | DONE | Collecte donnees historiques — 4 sources, 1357 prix, 1824 news |
| 3 | DONE | Correlation historique — catalyst_matcher, 649 associations, 113/166 trades |
| 4 | DONE | Feature engineering + ML — NewsClassifier, TechnicalIndicators, FeatureEngine, XGBoost Trainer, Evaluator |
| 4bis | DONE | LLM Trade Analysis — GPT-4o-mini analyse 141 trades, remplace regex par analyse profonde |
| 5 | TODO | Pipeline temps reel + alertes Telegram |
| 6 | TODO | Tests integration, stabilisation, deploiement VPS |

## Current Project State

| Aspect | Status | Details |
|--------|--------|---------|
| Code | Etape 1+2+3+4+4bis DONE | pdf_parser, trade_matcher, database, 4 collectors, ticker_mapper, catalyst_matcher, news_classifier, technical_indicators, feature_engine, llm_analyzer, trainer, evaluator, CLI |
| Config | .env configure | ALPHA_VANTAGE_API_KEY + MARKETAUX_API_KEY + OPENAI_API_KEY |
| Tests | 183/183 PASS | database(23), pdf_parser(17), trade_matcher(7), ticker_mapper(9), price_collector(5), news_collector(5), alpha_vantage(4), marketaux(5), rss(5), catalyst_matcher(22), news_classifier(24), technical_indicators(17), feature_engine(15), llm_analyzer(10), trainer(10), evaluator(6) |
| Data | Collectee + LLM analysee | 214 exec, 166 trades, 1357 prix, 1824 news, 649 catalyseurs, 141 analyses LLM |
| Docs | Design + plans Etapes 2-4bis | docs/plans/2026-02-22-etape*-*.md, 2026-02-23-etape4-*.md, 2026-02-24-etape4-llm-*.md |
| Git | Pushe sur origin | 27 commits, master a jour |

## Sources de donnees actives

| Source | Fichier | Articles | Sentiment | Cle API |
|--------|---------|----------|-----------|---------|
| yfinance | price_collector.py | 1357 prix | - | Non |
| GNews | news_collector.py | 412 | Non | Non |
| Alpha Vantage | alpha_vantage_collector.py | 700 | Oui | Oui (.env) |
| Marketaux | marketaux_collector.py | 43 | Oui | Oui (.env) |
| RSS Google News FR | rss_collector.py | 669 | Non | Non |

**Tickers sans prix yfinance (probablement delistes):** 2CRSI.PA, ALTBG.PA, AFYREN.PA

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

## Next Immediate Action

**Etape 5: Pipeline temps reel + alertes Telegram.**

Objectif: deployer le modele entraine pour scanner en temps reel la watchlist
de 30 valeurs, calculer un score "Nicolas prendrait ce trade?", et envoyer
des alertes Telegram quand le score depasse le seuil.

Sous-etapes:
1. **Predictor temps reel** — Charger le modele, calculer les features en direct
2. **Signal filter** — Logique de filtrage intelligent (seuil, cooldown, dedup)
3. **Telegram bot** — Envoi des alertes formatees
4. **Scheduler** — APScheduler pour la boucle de collecte + scoring
5. **Configuration** — Watchlist 30 valeurs, seuils, intervalles

**Watchlist 30 valeurs** fournie (screenshots data/).

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
- **IMPORTANT**: Nicolas ne prend pas des trades au hasard. Chaque achat est motive par un
  catalyseur specifique (news, annonce, signal technique). Le modele doit apprendre CES motivations.
