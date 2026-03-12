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
uv run python scripts/analyze_trades_llm.py           # Analyser trades via GPT-4o-mini
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
| LLM | openai (GPT-4o-mini) |
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
| Code | Etape 1-7 DONE + feedback loop corrige | 5 bugs critiques corriges le 12 mars 2026. Modele retraine sur dataset equilibre (90W/155L). Scanner tourne sur VPS avec 13 jobs. |
| Config | .env configure | ALPHA_VANTAGE_API_KEY + MARKETAUX_API_KEY + OPENAI_API_KEY (nouvelle cle 12 mars) + NEWSDATA_API_KEY + TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID |
| Tests | 405/405 PASS | database(59), pdf_parser(17), trade_matcher(7), ticker_mapper(11), price_collector(5), news_collector(5), alpha_vantage(4), marketaux(5), rss(5), catalyst_matcher(22), news_classifier(24), technical_indicators(17), feature_engine(38), llm_analyzer(10), llm_sentiment(8), fundamental_collector(7), newsdata_collector(7), boursorama_scraper(7), trainer(10), evaluator(6), predictor(10), signal_filter(13), formatter(7), telegram_bot(4), signal_reviewer(32), performance_tracker(29), model_retrainer(25), orderbook_collector(9) |
| Data VPS | 202 signaux, 134 reviews, 7132 news, 2702 prix, 6943 orderbook | Win rate global: 16.4% (22W 84L 28N). Modele retraine le 12 mars sur dataset equilibre. |
| Docs | Design + plans Etapes 2-5 + fix feedback loop | docs/plans/2026-03-12-fix-feedback-loop.md |
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
| LLM Sentiment | llm_sentiment.py | scorer ~1081 news | GPT-4o-mini | Oui (.env) |
| Boursorama Orderbook | orderbook_collector.py | carnet d'ordres | - | Non |

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

**9 mars — Philosophie: envoyer TOUT pour que le bot apprenne.** Le feedback loop sur-filtrait (EXCLUDE_CATALYST bloquait 96% des signaux). Corrige:

1. **Filtrage debrided**: EXCLUDE_CATALYST remplace par CATALYST_STATS (informatif, pas bloquant).
2. **Seuil abaisse a 0.50**: On envoie TOUS les signaux pour avoir des donnees negatives.
3. **4 nouvelles sources dans le scheduler**: Alpha Vantage, Marketaux, Newsdata.io, LLM Sentiment. Total: 13 jobs.
4. **Apprentissage continu**: `build_combined_features()` fusionne trades historiques + signal reviews.
5. **Carnet d'ordres**: OrderBookCollector scrape Boursorama (31 symboles). 4 features orderbook.

**12 mars — 5 bugs critiques corriges (le bot n'apprenait pas):**

1. **Orderbook parser**: `_parse_orderbook()` cherchait `data["bids"]` mais Boursorama renvoie `data["orderbook"]["lines"]` avec format `{bid, bidSize, bidNb, ask, askSize, askNb}`. 6943 snapshots etaient vides. Corrige.
2. **Retrain crash KeyError date_achat**: `build_realtime_features()` ne retournait pas `date_achat`. Quand les reviews etaient combinees pour le retrain, la colonne manquait et `walk_forward_validate()` crashait. Corrige.
3. **Walk-forward split_date hardcode**: `split_date="2025-12-01"` dans `trainer.py`. Pour 2026, tout etait dans le test set. Remplace par split dynamique (75e percentile).
4. **catalyst_news_title toujours None**: Hardcode `None` dans `predictor.py`. Maintenant extrait de la meilleure news matchee.
5. **Predictor ne rechargeait pas apres retrain**: Apres un retrain reussi, le scanner continuait avec l'ancien modele en memoire. Ajout de `reload_model()`.
6. **Cle API OpenAI invalide sur VPS**: Remplacee par nouvelle cle.
7. **Modele retraine**: Dataset equilibre 90 winners / 155 losers (etait 89% winners). Scores maintenant de 0.09 a 0.97 (etait 0.85-0.98 pour tout).

## Next Immediate Action

**Surveiller le bot et attendre les prochaines reviews.**

Le bot tourne sur le VPS avec le modele retraine. Prochaines etapes:
1. **Surveiller** les reviews J+3 des prochains jours — verifier que le win rate s'ameliore avec le nouveau modele
2. **Verifier** que le LLM sentiment scorer fonctionne (nouvelle cle API OpenAI)
3. **Verifier** que les orderbook features ne sont plus 0.0 (parser corrige, mais donnees seulement pendant horaires marche)
4. **Dimanche 15 mars 19h**: Premier retrain automatique avec le cycle complet
5. **Ameliorer** le retrainer: la comparaison old/new est biaisee (les deux modeles font le meme walk-forward, donc memes metriques)

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
- **IMPORTANT**: Nicolas ne prend pas des trades au hasard. Chaque achat est motive par un
  catalyseur specifique (news, annonce, signal technique). Le modele doit apprendre CES motivations.
