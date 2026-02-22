# PEA Scanner Bot

Bot Python de veille boursiere PEA. Analyse des donnees marche en temps reel, correlation avec les actualites, modele predictif ML entraine sur l'historique de trading personnel, alertes Telegram avec score de confiance.

## Commandes

```bash
uv run pytest tests/ -v          # Lancer tous les tests (97 tests)
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
```

## Architecture

```
pea-scanner/
├── src/                        # Code source (packages Python)
│   ├── core/                   # Couche fondation (BDD, logging, config)
│   ├── extraction/             # Module 1: Extraction PDF -> SQLite
│   ├── data_collection/        # Module 2: Collecte prix/news (4 sources actives)
│   ├── analysis/               # Module 3: Features + ML (a venir)
│   ├── model/                  # Module 3: Entrainement + scoring (a venir)
│   └── alerts/                 # Module 4: Telegram (a venir)
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
| Indicateurs techniques | ta |
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
| 4 | TODO | Feature engineering + entrainement ML |
| 5 | TODO | Pipeline temps reel + alertes Telegram |
| 6 | TODO | Tests integration, stabilisation, deploiement VPS |

## Current Project State

| Aspect | Status | Details |
|--------|--------|---------|
| Code | Etape 1+2+3 DONE | pdf_parser, trade_matcher, database, 4 collectors, ticker_mapper, catalyst_matcher, CLI |
| Config | .env configure | ALPHA_VANTAGE_API_KEY + MARKETAUX_API_KEY |
| Tests | 97/97 PASS | database(18), pdf_parser(17), trade_matcher(7), ticker_mapper(9), price_collector(5), news_collector(5), alpha_vantage(4), marketaux(5), rss(5), catalyst_matcher(22) |
| Data | Collectee + correlee | 214 exec, 166 trades, 1357 prix, 1824 news, 649 catalyseurs |
| Docs | Design + plans Etapes 2-3 | docs/plans/2026-02-22-etape*-*.md |
| Git | Pushe sur origin | 19 commits, master a jour |

## Sources de donnees actives

| Source | Fichier | Articles | Sentiment | Cle API |
|--------|---------|----------|-----------|---------|
| yfinance | price_collector.py | 1357 prix | - | Non |
| GNews | news_collector.py | 412 | Non | Non |
| Alpha Vantage | alpha_vantage_collector.py | 700 | Oui | Oui (.env) |
| Marketaux | marketaux_collector.py | 43 | Oui | Oui (.env) |
| RSS Google News FR | rss_collector.py | 669 | Non | Non |

**Tickers sans prix yfinance (probablement delistes):** 2CRSI.PA, ALTBG.PA, AFYREN.PA

## Next Immediate Action

**Etape 4: Feature engineering + entrainement ML.** Creer le pipeline de features et entrainer un modele XGBoost.

Objectif: transformer les donnees brutes (prix, news, catalyseurs) en features pour un modele predictif.

Donnees disponibles pour le ML:
- 166 trades (89% win rate, 141 clotures)
- 649 associations trade-catalyseur (score moyen 0.84)
- 1357 prix OHLCV, 1824 news avec sentiment partiel
- Sources: gnews (401 assoc), alpha_vantage (100), RSS (103), marketaux (25)

**Watchlist 30 valeurs** fournie (screenshots data/) — sera utilisee a l'etape 5 seulement.

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
