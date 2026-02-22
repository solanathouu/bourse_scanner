# PEA Stock Scanner Bot — Document de Planification Complet

## Contexte du Projet

**Objectif** : Bot Python déployé sur VPS qui analyse des données boursières en temps réel, les corrèle avec des actualités, applique un modèle prédictif entraîné sur l'historique de trading personnel, et envoie des alertes Telegram avec score de confiance.

**Périmètre** : Actions éligibles PEA (marchés européens). Phase 1 sur watchlist de 30 valeurs, extension progressive.

**Style de trading** : Swing trading court terme (quelques jours), objectif 4-5% par trade, volume élevé, taux de réussite prioritaire.

**Données d'entraînement** : ~200 avis d'exécution PDF Société Générale sur 10 mois.

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────┐
│                        VPS (Linux)                              │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │  MODULE 1    │   │  MODULE 2    │   │  MODULE 3           │  │
│  │  Extraction  │──▶│  Données     │──▶│  Moteur Prédictif   │  │
│  │  PDF         │   │  Marché      │   │  (ML + Backtesting) │  │
│  └──────────────┘   │  + News      │   └─────────┬───────────┘  │
│                     └──────────────┘             │              │
│                                                  ▼              │
│                                        ┌─────────────────────┐  │
│                                        │  MODULE 4           │  │
│                                        │  Alertes Telegram   │  │
│                                        └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

**Flux de données** :
1. Module 1 : Extraction et structuration des PDF → base de données d'entraînement
2. Module 2 : Collecte temps réel des prix, volumes, indicateurs + agrégation news
3. Module 3 : Corrélation historique (pourquoi chaque trade a été pris), entraînement du modèle, scoring en temps réel
4. Module 4 : Filtrage intelligent des signaux + envoi Telegram conditionnel

---

## MODULE 1 — Extraction et Structuration des PDF

### 1.1 Objectif

Transformer les ~200 avis d'exécution PDF Société Générale en une base de données structurée et chronologique par action.

### 1.2 Champs à extraire par PDF

| Champ | Type | Exemple |
|-------|------|---------|
| date_execution | datetime | 2024-03-15 |
| heure_execution | time | 09:32:15 |
| sens | enum | ACHAT / VENTE |
| nom_action | string | LVMH |
| isin | string | FR0000121014 |
| quantite | integer | 50 |
| prix_unitaire | float | 842.30 |
| montant_brut | float | 42115.00 |
| commission | float | 4.95 |
| frais | float | 0.00 |
| montant_net | float (calculé) | 42119.95 |

### 1.3 Pipeline d'extraction

```
PDF brut → Extraction texte (pdfplumber) → Parsing regex/template → Validation → Base SQLite
```

**Étapes détaillées** :

1. **Extraction texte** : Utiliser `pdfplumber` (meilleur que PyPDF2 pour les tableaux et structures fixes). Les avis SG ont un format standardisé, donc un seul template de parsing suffit.

2. **Parsing** : Identifier les positions fixes des champs dans le document SG. Créer des regex adaptées au format spécifique. Tester sur 5-10 PDF d'abord pour valider le template.

3. **Validation** : Vérifier que montant_brut ≈ quantité × prix_unitaire. Vérifier que la date est cohérente (dans les 10 mois). Alerter sur les anomalies de parsing.

4. **Stockage** : Base SQLite (`trades.db`) avec table `executions`.

### 1.4 Structuration par action

Une fois tous les PDF parsés, le bot doit :

- Regrouper par ISIN/nom d'action
- Ordonner chronologiquement
- Reconstruire les positions : chaque ACHAT ouvre ou renforce, chaque VENTE réduit ou clôture
- Calculer pour chaque trade complet (achat → vente) : rendement brut (%), rendement net (après frais), durée de détention (jours/heures), volume engagé

**Table `trades_complets`** :

| Champ | Description |
|-------|-------------|
| id_trade | Identifiant unique |
| isin | Code ISIN |
| nom_action | Nom |
| date_achat | Date + heure d'achat |
| date_vente | Date + heure de vente |
| prix_achat | Prix unitaire à l'achat |
| prix_vente | Prix unitaire à la vente |
| quantite | Nombre d'actions |
| rendement_brut_pct | % de gain/perte brut |
| rendement_net_pct | % après frais |
| duree_jours | Durée de détention |
| frais_totaux | Commission + frais achat + vente |

### 1.5 Librairies Python recommandées

| Librairie | Usage |
|-----------|-------|
| `pdfplumber` | Extraction texte des PDF (gère bien les tableaux) |
| `sqlite3` | Base de données locale (stdlib Python) |
| `pandas` | Manipulation et analyse des données extraites |
| `re` | Regex pour le parsing du template SG |

---

## MODULE 2 — Collecte de Données Marché et News

### 2.1 Sources de données de prix/volumes

#### API gratuites recommandées (par ordre de priorité)

| Source | Type | Limites gratuites | Forces | Faiblesses |
|--------|------|-------------------|--------|------------|
| **Yahoo Finance** (`yfinance`) | Librairie Python | Illimité (non officiel) | Données historiques complètes, temps réel ~15min delay, couvre Euronext. Ticker format : `MC.PA` (LVMH Paris) | Non officiel, peut casser. Pas de vrai temps réel |
| **Alpha Vantage** | API REST | 25 requêtes/jour (free), 75/min (premium $49.99/m) | Intraday 1min-60min, indicateurs techniques intégrés (RSI, MACD, SMA, EMA), données fondamentales | Limite très basse en gratuit. Couverture européenne partielle |
| **Twelve Data** | API REST | 800 requêtes/jour, 8/min | Temps réel et historique, 50+ indicateurs techniques, websocket disponible, bonne couverture Euronext | Websocket en premium uniquement |
| **Finnhub** | API REST + Websocket | 60 appels/min | News en temps réel, sentiment analysis intégré, calendrier résultats/dividendes, websocket gratuit (US seulement) | Websocket Europe = premium |
| **Financial Modeling Prep** (FMP) | API REST | 250 requêtes/jour | Données fondamentales très complètes, screener intégré, historique dividendes | Couverture européenne limitée sur le tier gratuit |
| **Marketstack** | API REST | 100 requêtes/mois (free) | Données EOD fiables, 70+ bourses mondiales dont Euronext | Trop limité en gratuit pour du temps réel |
| **Boursorama** | Scraping | N/A | Données françaises très complètes, consensus analystes, actualités FR | Scraping uniquement, risque de blocage |
| **Euronext** | API REST (données delayed) | Données delayed gratuites | Source officielle Euronext, fiable | Données delayed 15min |

#### Stratégie de combinaison recommandée

**Couche primaire (prix + volumes)** :
- `yfinance` pour l'historique et les snapshots réguliers (polling toutes les 1-2 minutes)
- `Alpha Vantage` ou `Twelve Data` pour les indicateurs techniques pré-calculés

**Couche secondaire (fondamentaux)** :
- `Financial Modeling Prep` pour les données fondamentales, ratios, calendrier résultats
- `Finnhub` pour le calendrier earnings et les métriques fondamentales

**Couche de secours** :
- Scraping Boursorama/Investing.com si une API tombe ou manque de couverture

### 2.2 Sources de news et actualités

#### API et flux RSS (priorité 1)

| Source | Type | Contenu | Accès |
|--------|------|---------|-------|
| **Finnhub News API** | API REST | News marché en anglais, catégorisées par ticker | Gratuit (60/min) |
| **Alpha Vantage News Sentiment** | API REST | News + score de sentiment par ticker | Gratuit (25/jour) |
| **NewsAPI.org** | API REST | Agrégateur multi-sources, filtrage par mot-clé | Gratuit (100 requêtes/jour, delayed 24h en gratuit — production $449/m) |
| **GNews API** | API REST | Agrégateur Google News, filtrable par langue FR | Gratuit (100 requêtes/jour) |
| **Boursorama RSS** | RSS | Actualités françaises, par action | Gratuit |
| **Zone Bourse RSS** | RSS | Analyses et news FR, consensus | Gratuit |
| **Investing.com RSS** | RSS | News internationales et FR | Gratuit |
| **Les Echos Bourse RSS** | RSS | Actualités économiques et financières FR | Gratuit |
| **BFM Bourse RSS** | RSS | News marché FR en temps réel | Gratuit |
| **Reuters RSS** | RSS | News internationales, corporate | Gratuit |
| **Cercle Finance** | RSS | Dépêches boursières FR | Gratuit |
| **AOF (Agence Option Finance)** | RSS | Recommandations analystes, résultats | Gratuit |
| **Capital.fr RSS** | RSS | Actualités économiques FR | Gratuit |

#### Scraping (priorité 2 — si les API/RSS ne suffisent pas)

| Source | Contenu cible | Difficulté |
|--------|---------------|------------|
| **Investing.com** | News, analyses techniques, calendrier économique | Moyenne (anti-bot modéré) |
| **Boursorama** | Forum, consensus, actualités par action | Moyenne |
| **Zone Bourse** | Recommandations analystes, objectifs de cours | Facile |
| **TradingView** | Indicateurs communautaires, idées de trades | Difficile (SPA React) |
| **Euronext.com** | Communiqués officiels, corporate actions | Facile |
| **AMF (autoritemarchesfinanciers.fr)** | Publications réglementaires, déclarations de franchissement de seuils | Facile |

### 2.3 Données supplémentaires utiles

| Donnée | Source recommandée | Utilité |
|--------|-------------------|---------|
| Calendrier des résultats | Finnhub, FMP, Investing.com | Anticiper la volatilité pré-earnings |
| Consensus analystes | Zone Bourse (scraping), FMP | Comparer prix actuel vs. objectif moyen |
| Short interest / positions vendeuses | AMF (publications réglementaires) | Détecter les squeezes potentiels |
| Volumes anormaux | Calculé via yfinance (volume vs. moyenne 20j) | Signal de momentum |
| Calendrier économique | Investing.com (scraping), Finnhub | Événements macro qui impactent le marché |
| Insider trading | AMF, Finnhub | Dirigeants qui achètent/vendent |

### 2.4 Architecture de collecte

```
┌─────────────────────────────────┐
│       SCHEDULER (APScheduler)    │
│                                  │
│  Toutes les 1-2 min :           │
│  ├─ yfinance → prix/volumes     │
│  │                               │
│  Toutes les 5-10 min :          │
│  ├─ Finnhub News API            │
│  ├─ RSS feeds (feedparser)      │
│  │                               │
│  Toutes les heures :            │
│  ├─ Alpha Vantage indicators    │
│  ├─ Scraping Boursorama/ZB      │
│  │                               │
│  Quotidien (pré-ouverture) :    │
│  ├─ Calendrier résultats        │
│  ├─ Consensus analystes         │
│  └─ Données fondamentales FMP   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│    BASE DE DONNÉES (SQLite)      │
│                                  │
│  Tables :                        │
│  ├─ prices (OHLCV par ticker)   │
│  ├─ indicators (RSI, MACD...)   │
│  ├─ news (titre, source, date,  │
│  │        ticker, sentiment)     │
│  ├─ fundamentals (PE, EPS...)   │
│  └─ events (earnings, macro)    │
└─────────────────────────────────┘
```

### 2.5 Librairies Python recommandées

| Librairie | Usage |
|-----------|-------|
| `yfinance` | Données de prix et volumes |
| `requests` | Appels API REST |
| `feedparser` | Parsing des flux RSS |
| `beautifulsoup4` + `httpx` | Scraping web |
| `apscheduler` | Planification des tâches de collecte |
| `sqlite3` ou `sqlalchemy` | Stockage |

---

## MODULE 3 — Moteur Prédictif

### 3.1 Phase A — Corrélation historique (PDF ↔ Events)

**Objectif** : Pour chaque trade dans `trades_complets`, retrouver automatiquement le catalyseur (news, événement) qui a probablement motivé la décision.

**Méthode** :

1. Pour chaque trade (date d'achat connue), requêter les news dans une fenêtre de -48h à +2h autour de l'achat.
2. Filtrer les news qui mentionnent le ticker ou le nom de l'entreprise.
3. Scorer la pertinence : proximité temporelle × pertinence sémantique (via embeddings ou mots-clés).
4. Stocker le ou les catalyseurs les plus probables dans une table `trade_catalyseurs`.

**Sources pour la recherche rétrospective** :
- Archives Investing.com / Boursorama (scraping des archives par date)
- Google News search filtrée par date (`before:YYYY-MM-DD after:YYYY-MM-DD`)
- Finnhub historical news (si dispo sur la période)
- Wayback Machine / archives web si nécessaire

**Table `trade_catalyseurs`** :

| Champ | Description |
|-------|-------------|
| id_trade | Référence au trade |
| date_news | Date de la news/événement |
| source | Origine (Investing, Boursorama, etc.) |
| titre | Titre de l'article |
| type_catalyseur | enum : EARNINGS, UPGRADE, DOWNGRADE, NEWS_POSITIVE, NEWS_NEGATIVE, MACRO, SECTOR, TECHNIQUE |
| score_pertinence | 0.0 à 1.0 |
| resume | Résumé court de la news |

### 3.2 Phase B — Feature Engineering

Pour chaque point de données (= chaque moment où le bot doit évaluer un signal), construire un vecteur de features.

**Features techniques (calculées à partir des prix)** :

| Feature | Description | Fenêtre |
|---------|-------------|---------|
| rsi_14 | Relative Strength Index | 14 périodes |
| macd_signal | MACD - Signal Line | 12/26/9 |
| macd_histogram | Histogramme MACD | 12/26/9 |
| sma_20 | Moyenne mobile simple | 20 jours |
| sma_50 | Moyenne mobile simple | 50 jours |
| ema_9 | Moyenne mobile exponentielle | 9 jours |
| bollinger_position | Position du prix dans les bandes | 20/2 |
| volume_ratio | Volume actuel / moyenne 20j | 20 jours |
| atr_14 | Average True Range (volatilité) | 14 périodes |
| prix_vs_sma20 | Écart en % prix/SMA20 | — |
| prix_vs_sma50 | Écart en % prix/SMA50 | — |
| variation_1j | Variation % sur 1 jour | 1 jour |
| variation_5j | Variation % sur 5 jours | 5 jours |
| gap_ouverture | Écart ouverture vs. clôture veille | — |
| plus_haut_52s | Distance au plus haut 52 semaines (%) | 252 jours |

**Features news/sentiment** :

| Feature | Description |
|---------|-------------|
| nb_news_24h | Nombre de news dans les 24 dernières heures |
| sentiment_moyen_24h | Score de sentiment moyen (-1 à +1) |
| news_volume_anormal | Booléen : plus de news que d'habitude |
| type_news_dominant | Catégorie dominante (earnings, upgrade, etc.) |
| intensite_news | Poids pondéré des news (source fiable = plus de poids) |

**Features fondamentales** :

| Feature | Description |
|---------|-------------|
| pe_ratio | Price/Earnings ratio actuel |
| distance_objectif_consensus | Écart en % vs. objectif analystes |
| nb_jours_avant_earnings | Nombre de jours avant prochains résultats |
| recommandation_consensus | Score moyen analystes (1=vente forte, 5=achat fort) |

### 3.3 Phase C — Entraînement du modèle

**Approche recommandée** : Gradient Boosting (XGBoost ou LightGBM).

Raisons :
- Performe bien sur des datasets de taille modeste (~200 trades)
- Gère nativement les features mixtes (numériques + catégorielles)
- Interprétable (feature importance)
- Moins de risque d'overfitting qu'un réseau de neurones sur 200 samples

**Cible de prédiction (target)** :

Option A (classification) : Le prix monte de ≥ X% dans les N prochains jours → label binaire (1 = signal, 0 = pas de signal). C'est la plus adaptée à ton style : tu cherches des trades à 4-5% en quelques jours.

Option B (régression) : Prédire le % de variation à horizon N jours. Plus ambitieux, moins stable avec 200 samples.

**Recommandation : Option A** avec X = 3% (seuil conservateur pour viser 4-5% avec marge) et N = 5 jours (horizon swing).

**Validation** :

- Walk-forward validation (pas de cross-validation classique, car les données sont temporelles). Entraîner sur les 7 premiers mois, tester sur les 3 derniers.
- Métriques : precision (pour minimiser les faux positifs), recall, F1-score, et surtout le profit simulé sur la période de test.

**Seuil de mise en production** :
- Precision ≥ 65% (sur les signaux BUY émis, au moins 65% sont effectivement profitables)
- Ratio gain moyen / perte moyenne > 1.5
- Le modèle doit être comparé à un baseline naïf (ex : acheter dès que RSI < 30)

### 3.4 Phase D — Scoring en temps réel

Quand le bot reçoit de nouvelles données (prix, news), il :

1. Calcule les features en temps réel pour chaque action de la watchlist
2. Passe le vecteur de features dans le modèle entraîné
3. Obtient une probabilité (score de confiance)
4. Applique le filtre de seuil de confiance (configurable, ex : > 0.65)
5. Si le seuil est franchi → génère un signal
6. Si le seuil n'est pas franchi → rien ne se passe (pas d'alerte inutile)

### 3.5 Librairies Python recommandées

| Librairie | Usage |
|-----------|-------|
| `xgboost` ou `lightgbm` | Modèle de classification |
| `scikit-learn` | Preprocessing, validation, métriques |
| `pandas` | Manipulation des features |
| `ta` (Technical Analysis) | Calcul des indicateurs techniques (RSI, MACD, Bollinger, etc.) |
| `numpy` | Calculs numériques |
| `joblib` | Sérialisation du modèle entraîné |

---

## MODULE 4 — Alertes Telegram

### 4.1 Configuration

- Créer un bot Telegram via @BotFather
- Récupérer le token API
- Récupérer le chat_id du canal/groupe cible
- Librairie : `python-telegram-bot` ou `httpx` direct sur l'API Telegram

### 4.2 Format d'alerte

```
🟢 SIGNAL BUY — LVMH (MC.PA)

Score de confiance : 78%

📊 Données techniques :
• Prix actuel : 842.30€
• RSI(14) : 38.2
• Volume : +145% vs. moy. 20j
• Support SMA50 : 831.00€

📰 Catalyseur détecté :
• [Investing.com] "LVMH : résultats T3 au-dessus du consensus, croissance Asie +12%"
• Sentiment : Positif (0.82)

🎯 Objectif estimé : +4.2% (→ 877.70€)
⏱️ Horizon : 3-5 jours

```

### 4.3 Logique de filtrage intelligent

Le bot NE notifie PAS si :
- Le score de confiance est sous le seuil paramétré

Le bot NOTIFIE immédiatement si :
- Score de confiance ≥ seuil ET au moins un catalyseur news détecté
- Volume anormal détecté (> 200% moyenne) même sans catalyseur news identifié
- Gap d'ouverture significatif (> 2%) sur une action de la watchlist

### 4.4 Librairies Python recommandées

| Librairie | Usage |
|-----------|-------|
| `python-telegram-bot` | Envoi de messages Telegram |
| `httpx` | Alternative légère pour l'API Telegram |

---

## Structure du Projet

```
pea-scanner/
│
├── config/
│   ├── config.yaml          # Configuration générale (API keys, seuils, watchlist)
│   └── tickers_watchlist.yaml # Liste des 30 tickers initiaux
│
├── data/
│   ├── pdfs/                 # Avis d'exécution PDF bruts
│   ├── trades.db             # Base SQLite principale
│   └── models/               # Modèles entraînés sérialisés (.joblib)
│
├── src/
│   ├── extraction/
│   │   ├── pdf_parser.py         # Extraction texte des PDF SG
│   │   ├── trade_matcher.py      # Matching achat/vente et reconstruction trades
│   │   └── db_loader.py          # Chargement en base
│   │
│   ├── data_collection/
│   │   ├── price_collector.py    # Collecte prix/volumes (yfinance + APIs)
│   │   ├── news_collector.py     # Collecte news (APIs + RSS + scraping)
│   │   ├── fundamental_collector.py # Données fondamentales
│   │   ├── rss_feeds.py          # Parsing flux RSS
│   │   └── scrapers/
│   │       ├── investing_scraper.py
│   │       ├── boursorama_scraper.py
│   │       └── zonebourse_scraper.py
│   │
│   ├── analysis/
│   │   ├── feature_engine.py     # Calcul de toutes les features
│   │   ├── technical_indicators.py # Indicateurs techniques
│   │   ├── sentiment_analyzer.py  # Analyse de sentiment des news
│   │   └── catalyst_matcher.py    # Corrélation trades historiques ↔ events
│   │
│   ├── model/
│   │   ├── trainer.py            # Entraînement du modèle
│   │   ├── backtester.py         # Backtesting et validation
│   │   ├── predictor.py          # Scoring en temps réel
│   │   └── evaluator.py          # Métriques de performance
│   │
│   ├── alerts/
│   │   ├── telegram_bot.py       # Envoi des alertes
│   │   ├── signal_filter.py      # Logique de filtrage intelligent
│   │   └── formatter.py          # Formatage des messages
│   │
│   └── core/
│       ├── scheduler.py          # Orchestration des tâches
│       ├── database.py           # Couche d'accès base de données
│       └── logger.py             # Logging centralisé
│
├── scripts/
│   ├── init_db.py                # Initialisation de la base
│   ├── import_pdfs.py            # Import batch des PDF
│   ├── train_model.py            # Script d'entraînement
│   ├── backtest.py               # Script de backtesting
│   └── run_scanner.py            # Point d'entrée principal
│
├── tests/
│   ├── test_pdf_parser.py
│   ├── test_feature_engine.py
│   ├── test_predictor.py
│   └── test_signal_filter.py
│
├── requirements.txt
├── .env                          # API keys (non versionné)
└── README.md
```

---

## Ordre d'Implémentation Recommandé

### Étape 1 — Extraction PDF (estimé : 1-2 jours)
1. Parser un PDF SG manuellement pour comprendre le format exact
2. Coder `pdf_parser.py` avec `pdfplumber`
3. Tester sur 5 PDF, ajuster les regex
4. Batch import des 200 PDF
5. Coder `trade_matcher.py` pour reconstruire les trades complets
6. Vérifier manuellement 10-15 trades reconstruits

### Étape 2 — Collecte de données marché (estimé : 2-3 jours)
1. Configurer `yfinance` avec les tickers de la watchlist (format Euronext : `MC.PA`, `BN.PA`, etc.)
2. Intégrer Alpha Vantage et/ou Twelve Data pour les indicateurs
3. Mettre en place les flux RSS avec `feedparser`
4. Intégrer Finnhub News API
5. Coder les scrapers de secours (Boursorama, Investing.com)
6. Configurer le scheduler de collecte

### Étape 3 — Corrélation historique (estimé : 2-3 jours)
1. Pour chaque trade historique, rechercher les news autour de la date d'achat
2. Scorer et associer les catalyseurs
3. Constituer la table `trade_catalyseurs`
4. Validation manuelle d'un échantillon (vérifier que les catalyseurs trouvés sont pertinents)

### Étape 4 — Feature engineering + Entraînement (estimé : 3-4 jours)
1. Calculer les indicateurs techniques historiques pour chaque action tradée
2. Reconstruire les features de sentiment historiques
3. Assembler les vecteurs de features pour chaque trade
4. Entraîner le modèle XGBoost/LightGBM
5. Walk-forward validation
6. Analyser les feature importances
7. Itérer si les métriques sont insuffisantes

### Étape 5 — Pipeline temps réel + Telegram (estimé : 2-3 jours)
1. Connecter le feature engine au flux de données temps réel
2. Intégrer le modèle de scoring
3. Coder la logique de filtrage intelligent
4. Configurer le bot Telegram
5. Test end-to-end sur les 30 valeurs de la watchlist

### Étape 6 — Tests et stabilisation (estimé : 2-3 jours)
1. Tests unitaires sur chaque module
2. Test d'intégration du pipeline complet
3. Monitoring des erreurs et edge cases
4. Optimisation des intervalles de collecte
5. Déploiement VPS et configuration systemd/supervisor

---

## Fichier `requirements.txt`

```
# Extraction PDF
pdfplumber==0.11.4

# Données marché
yfinance==0.2.40
alpha-vantage==3.0.0
requests==2.32.3
httpx==0.27.0

# News et RSS
feedparser==6.0.11
beautifulsoup4==4.12.3

# Analyse technique
ta==0.11.0
pandas==2.2.2
numpy==1.26.4

# Machine Learning
xgboost==2.1.1
scikit-learn==1.5.1
lightgbm==4.5.0
joblib==1.4.2

# Telegram
python-telegram-bot==21.5

# Scheduling
APScheduler==3.10.4

# Base de données
sqlalchemy==2.0.31

# Utilitaires
python-dotenv==1.0.1
pyyaml==6.0.2
loguru==0.7.2
```

---

## Fichier `config.yaml` (template)

```yaml
# PEA Scanner Configuration

api_keys:
  alpha_vantage: "${ALPHA_VANTAGE_KEY}"
  finnhub: "${FINNHUB_KEY}"
  twelve_data: "${TWELVE_DATA_KEY}"
  fmp: "${FMP_KEY}"
  newsapi: "${NEWSAPI_KEY}"
  gnews: "${GNEWS_KEY}"
  telegram_token: "${TELEGRAM_TOKEN}"
  telegram_chat_id: "${TELEGRAM_CHAT_ID}"

scanner:
  price_interval_seconds: 120        # Polling prix toutes les 2 min
  news_interval_seconds: 300         # Polling news toutes les 5 min
  indicators_interval_seconds: 3600  # Indicateurs techniques toutes les heures
  fundamentals_interval_seconds: 86400 # Fondamentaux 1x/jour

model:
  type: "xgboost"                    # xgboost ou lightgbm
  confidence_threshold: 0.65         # Seuil minimum pour signal
  target_gain_pct: 3.0               # Seuil de hausse pour label positif
  target_horizon_days: 5             # Horizon de prédiction
  min_precision: 0.65                # Precision minimale requise

alerts:
  cooldown_hours: 6                  # Anti-spam : pas de signal identique avant 6h
  notify_outside_market_hours: false # Ne pas notifier hors horaires (sauf exception)
  volume_spike_threshold: 2.0        # Volume > 200% moyenne → alerte même sans catalyseur
  gap_threshold_pct: 2.0             # Gap ouverture > 2% → alerte

rss_feeds:
  - name: "Boursorama"
    url: "https://www.boursorama.com/rss/actualites"
  - name: "Zone Bourse"
    url: "https://www.zonebourse.com/rss/"
  - name: "Investing FR"
    url: "https://fr.investing.com/rss/news.rss"
  - name: "Les Echos"
    url: "https://www.lesechos.fr/rss/rss_bourse.xml"
  - name: "BFM Bourse"
    url: "https://bourse.bfmtv.com/rss/info/"
  - name: "Reuters Business"
    url: "http://feeds.reuters.com/reuters/businessNews"
  - name: "Cercle Finance"
    url: "https://www.cerclefinance.com/rss"
  - name: "Capital"
    url: "https://www.capital.fr/entreprises-marches/rss"

database:
  path: "data/trades.db"

logging:
  level: "INFO"
  file: "logs/scanner.log"
```

---

## Schéma de Base de Données

```sql
-- Avis d'exécution bruts (PDF parsés)
CREATE TABLE executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date_execution DATE NOT NULL,
    heure_execution TIME NOT NULL,
    sens TEXT NOT NULL CHECK (sens IN ('ACHAT', 'VENTE')),
    nom_action TEXT NOT NULL,
    isin TEXT,
    quantite INTEGER NOT NULL,
    prix_unitaire REAL NOT NULL,
    montant_brut REAL NOT NULL,
    commission REAL DEFAULT 0,
    frais REAL DEFAULT 0,
    montant_net REAL,
    fichier_source TEXT
);

-- Trades reconstitués (achat → vente)
CREATE TABLE trades_complets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    isin TEXT,
    nom_action TEXT NOT NULL,
    date_achat DATETIME NOT NULL,
    date_vente DATETIME,
    prix_achat REAL NOT NULL,
    prix_vente REAL,
    quantite INTEGER NOT NULL,
    rendement_brut_pct REAL,
    rendement_net_pct REAL,
    duree_jours REAL,
    frais_totaux REAL,
    statut TEXT DEFAULT 'OUVERT' CHECK (statut IN ('OUVERT', 'CLOTURE'))
);

-- Catalyseurs identifiés pour chaque trade
CREATE TABLE trade_catalyseurs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_trade INTEGER REFERENCES trades_complets(id),
    date_news DATETIME,
    source TEXT,
    titre TEXT,
    type_catalyseur TEXT CHECK (type_catalyseur IN (
        'EARNINGS', 'UPGRADE', 'DOWNGRADE', 'NEWS_POSITIVE',
        'NEWS_NEGATIVE', 'MACRO', 'SECTOR', 'TECHNIQUE'
    )),
    score_pertinence REAL,
    resume TEXT
);

-- Prix et volumes collectés
CREATE TABLE prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    source TEXT
);

-- Indicateurs techniques calculés
CREATE TABLE indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    macd_histogram REAL,
    sma_20 REAL,
    sma_50 REAL,
    ema_9 REAL,
    bollinger_upper REAL,
    bollinger_lower REAL,
    atr_14 REAL,
    volume_ratio REAL
);

-- News collectées
CREATE TABLE news (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT,
    timestamp DATETIME NOT NULL,
    source TEXT NOT NULL,
    titre TEXT NOT NULL,
    url TEXT,
    sentiment_score REAL,
    type_news TEXT,
    contenu_resume TEXT
);

-- Signaux émis
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    timestamp DATETIME NOT NULL,
    type_signal TEXT CHECK (type_signal IN ('BUY', 'WATCH')),
    confidence_score REAL NOT NULL,
    prix_actuel REAL,
    objectif_pct REAL,
    horizon_jours INTEGER,
    catalyseur_principal TEXT,
    notifie BOOLEAN DEFAULT FALSE,
    resultat_reel_pct REAL  -- rempli a posteriori pour suivi performance
);

-- Index pour les requêtes fréquentes
CREATE INDEX idx_prices_ticker_ts ON prices(ticker, timestamp);
CREATE INDEX idx_news_ticker_ts ON news(ticker, timestamp);
CREATE INDEX idx_indicators_ticker_ts ON indicators(ticker, timestamp);
CREATE INDEX idx_signals_ticker_ts ON signals(ticker, timestamp);
CREATE INDEX idx_executions_date ON executions(date_execution);
```

---

## Points d'Attention et Risques

### Risques techniques
- **yfinance** n'est pas une API officielle : Yahoo peut changer le format ou bloquer. Prévoir un fallback.
- **200 trades** est un dataset modeste pour du ML. Le modèle sera forcément limité au début. Prévoir un mécanisme de réentraînement au fil des nouveaux trades.
- **Le scraping** peut casser à tout moment si un site change sa structure HTML. Isoler chaque scraper pour pouvoir le désactiver indépendamment.
- **Le temps réel** sur les actions européennes est souvent en delay de 15 min en gratuit. Le vrai temps réel nécessite des flux payants (Euronext, Interactive Brokers API).

### Risques méthodologiques
- **Overfitting** : Avec 200 trades et beaucoup de features, le modèle peut apprendre le bruit. Réduire le nombre de features, utiliser la régularisation, et valider strictement en walk-forward.
- **Survivorship bias** : Le modèle est entraîné sur tes trades passés, qui reflètent tes biais de sélection. Il ne verra jamais les trades que tu n'as pas pris.
- **Corrélation ≠ Causalité** : Le fait qu'une news précède un trade ne prouve pas qu'elle l'a causé. La couche de corrélation historique est une heuristique, pas une vérité.

### Cadre légal
- Le bot est un **outil d'aide à la décision**, pas un robot de trading automatisé.
- Le PEA est soumis à la réglementation française : pas de vente à découvert, uniquement des actions éligibles (Europe).
- Le scraping doit respecter les CGU des sites. En cas de doute, se limiter aux API et RSS.

---

## Annexe — Mode Paper Trading (bonus)

À implémenter après la mise en production des alertes :

1. Créer une table `paper_trades` qui enregistre chaque signal comme un trade virtuel
2. Au moment de la clôture (atteinte de l'objectif ou expiration de l'horizon), enregistrer le résultat
3. Dashboard simple (Flask ou Streamlit) pour suivre la performance du modèle en conditions réelles sans engagement financier
4. Comparer les performances paper vs. trades réels pour mesurer la valeur ajoutée du bot
