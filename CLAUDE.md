# PEA Scanner Bot

Bot Python de veille boursiere PEA. Analyse des donnees marche en temps reel, correlation avec les actualites, modele predictif ML entraine sur l'historique de trading personnel, alertes Telegram avec score de confiance.

## Commandes

```bash
uv run pytest tests/ -v          # Lancer tous les tests (379 tests)
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

SQLite dans `data/trades.db`. Tables principales (11 tables):

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
| Code | Etape 1-7 DONE + feedback debrided | Filtrage adaptatif remplace par CATALYST_STATS informatif (plus de blocage). 12 jobs scheduler (4 nouvelles sources). Seuil 0.50 pour maximiser les signaux et le feedback. |
| Config | .env configure | ALPHA_VANTAGE_API_KEY + MARKETAUX_API_KEY + OPENAI_API_KEY + NEWSDATA_API_KEY + TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID |
| Tests | 379/379 PASS | database(56), pdf_parser(17), trade_matcher(7), ticker_mapper(11), price_collector(5), news_collector(5), alpha_vantage(4), marketaux(5), rss(5), catalyst_matcher(22), news_classifier(24), technical_indicators(17), feature_engine(26), llm_analyzer(10), llm_sentiment(8), fundamental_collector(7), newsdata_collector(7), boursorama_scraper(7), trainer(10), evaluator(6), predictor(10), signal_filter(13), formatter(7), telegram_bot(4), signal_reviewer(32), performance_tracker(29), model_retrainer(22) |
| Data | Collectee + LLM analysee + enrichie + 97 reviews | 214 exec, 166 trades, 1357+ prix, 1824+ news, 649 catalyseurs, 141 analyses LLM, 97 signal reviews |
| Docs | Design + plans Etapes 2-5 | docs/plans/2026-02-22-etape*-*.md, 2026-02-23-etape4-*.md, 2026-02-24-etape4-llm-*.md |
| Git | Pushe sur origin | 30+ commits, master a jour |

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

**Philosophie: envoyer TOUT pour que le bot apprenne.** Le feedback loop sur-filtrait (EXCLUDE_CATALYST bloquait 96% des signaux). Corrige:

1. **Filtrage debrided**: EXCLUDE_CATALYST remplace par CATALYST_STATS (informatif, pas bloquant). Les alertes Telegram montrent le win rate historique du catalyseur mais ne bloquent plus l'envoi.
2. **Seuil abaisse a 0.50**: On envoie TOUS les signaux, meme faibles, pour avoir des donnees negatives (le modele etait entraine sur 89% de trades gagnants).
3. **4 nouvelles sources dans le scheduler**: Alpha Vantage (8h), Marketaux (12h), Newsdata.io (14h), LLM Sentiment (toutes les 2h). Total: 12 jobs.
4. **Seuil adaptatif adouci**: +0.02 si WR<30%, +0.01 si WR<40%, cap base+0.10 (etait +0.04/+0.02, cap 0.95).
5. **Fix retrain**: `date_achat` ajoute au vecteur de features pour le walk-forward split.

**Limitation connue**: Le retrain utilise toujours les 111 trades historiques. Les 97+ signal reviews ne sont PAS encore incorporees dans les donnees d'entrainement. C'est la prochaine evolution majeure.

## Next Immediate Action

**Monitorer les signaux avec la nouvelle config.** Le bot tourne sur le VPS avec 12 jobs, seuil 0.50.

Actions immediates:
1. **Verifier** que les 4 nouvelles sources collectent bien des donnees sur le VPS
2. **Monitorer** les alertes Telegram — on devrait recevoir beaucoup plus de signaux
3. **Observer** les reviews J+3 (37 reviews en attente, en attente des prix)

Prochaines evolutions:
1. **PRIORITE**: Incorporer les signal_reviews dans les donnees d'entrainement du modele (actuellement seuls les 111 trades historiques sont utilises)
2. **Attendre 50+ reviews** avec la nouvelle config pour evaluer la qualite
3. **Ameliorer** le modele quand on aura assez de donnees negatives

## Donnees cles du trading

- **Style**: Swing court terme (quelques jours), objectif 4-5% par trade
- **Perimetre**: Actions eligibles PEA (marches europeens)
- **Watchlist**: 30 valeurs (liste fournie par l'utilisateur)
- **Performance historique**: 89% win rate, +9.21% rendement moyen, 20.7j duree moyenne
- **19 actions tradees** sur ~10 mois (mai 2025 - fev 2026)
- **IMPORTANT**: Nicolas ne prend pas des trades au hasard. Chaque achat est motive par un
  catalyseur specifique (news, annonce, signal technique). Le modele doit apprendre CES motivations.
