# Etape 4 — ML Pipeline Design: "Trader comme Nicolas"

**Date**: 2026-02-23
**Statut**: Approuve
**Objectif**: Comprendre POURQUOI Nicolas prend chaque trade, extraire ses patterns de decision,
et entrainer un modele qui reproduit sa logique personnelle.

## Contexte

### Le style de trading de Nicolas (valide en brainstorming)

1. **Range trading**: Nicolas identifie des actions qui oscillent entre support et resistance
   ("dents de scie") sur des fenetres de 10-20 jours
2. **Achat en bas du range**: Il entre en position quand le prix est pres du support
3. **Vente en haut du range**: Il sort quand le prix approche la resistance
4. **Catalyseur news = accelerateur**: Une news positive (resultats, contrat, FDA, upgrade)
   confirme le timing d'entree. C'est la combinaison technique + catalyseur qui fait le trade.
5. **Breakout occasionnel**: Parfois le range se casse vers le haut et il laisse courir
   (ex: ADOCIA +154%)
6. **Objectif**: ~4-5% par trade, precision maximale

### Donnees disponibles

- 141 trades clotures (124 gagnants / 17 perdants, 88% win rate)
- 1357 prix OHLCV sur 17 tickers
- 1824 news de 4 sources (GNews 412, Alpha Vantage 700, RSS 669, Marketaux 43)
- 649 associations trade-catalyseur (113/166 trades matches)

### Decouverte critique de l'analyse de donnees

- Les trades perdants ont PLUS de catalyseurs (5.7) que les gagnants (5.2)
- 28 trades sans catalyseur ont 82% win rate
- => Le matching actuel (etape 3) est trop superficiel. Il trouve des news proches temporellement
  mais ne comprend pas le CONTENU. L'etape 4 doit aller plus loin.

## Decisions de design

| Decision | Choix | Justification |
|----------|-------|---------------|
| Approche ML | Classifier catalyseurs + Detecteur range + XGBoost | Capture le POURQUOI (news) et le QUAND (technique) de Nicolas |
| Classification news | Regles mots-cles FR+EN | Titres financiers explicites, pas besoin de NLP lourd |
| Detection range | Support/resistance rolling 10j et 20j | Nicolas regarde 10-20 jours pour definir son range |
| Indicateurs techniques | RSI, MACD, Bollinger, ATR, SMA, volume via `ta` | Standard, fiable, deja dans les deps |
| Target ML | Binaire: gagnant (>0%) vs perdant | Simple et robuste avec 141 samples |
| Validation | Walk-forward temporel | Mai-Nov 2025 (train) -> Dec 2025-Fev 2026 (test) |
| Seuil de reussite | Precision > 88% (battre le baseline naif) | Le baseline "toujours acheter" = 88% win rate |

## Architecture des fichiers

```
src/analysis/                          (Niveau 2 — importe core/ et data_collection/)
├── catalyst_matcher.py                (EXISTANT - etape 3)
├── news_classifier.py                 (NOUVEAU - classifie les news par type)
├── technical_indicators.py            (NOUVEAU - calcule RSI, MACD, range, etc.)
└── feature_engine.py                  (NOUVEAU - assemble le vecteur de features)

src/model/                             (Niveau 3 — importe core/ et analysis/)
├── trainer.py                         (NOUVEAU - entraine XGBoost)
├── evaluator.py                       (NOUVEAU - metriques et analyse)
└── predictor.py                       (ETAPE 5 - scoring temps reel)

scripts/
├── train_model.py                     (NOUVEAU - CLI entrainement)
└── analyze_features.py                (NOUVEAU - CLI exploration des features)

data/models/                           (Modeles sauvegardes)
└── nicolas_v1.joblib                  (Sortie du training)
```

## Composant 1: NewsClassifier

**Fichier**: `src/analysis/news_classifier.py`

**Role**: Analyser le CONTENU des news pour classifier le type de catalyseur.
Le matching de l'etape 3 trouve les news proches temporellement; le classifier
comprend de QUOI parle la news.

### Types de catalyseurs

```python
CATALYST_TYPES = {
    "EARNINGS": {
        "keywords_fr": ["resultats", "chiffre d'affaires", "benefice", "CA T1", "CA T2",
                        "CA T3", "CA T4", "bilan", "comptes", "trimestriels", "semestriels",
                        "annuels", "croissance du CA"],
        "keywords_en": ["earnings", "revenue", "profit", "quarterly", "results", "EPS",
                        "beat expectations", "guidance"],
        "priority": 10,
    },
    "FDA_REGULATORY": {
        "keywords_fr": ["FDA", "AMM", "autorisation", "approbation", "EMA", "phase 1",
                        "phase 2", "phase 3", "essai clinique", "reglementaire", "ANSM"],
        "keywords_en": ["FDA", "approval", "regulatory", "clinical trial", "phase",
                        "marketing authorization", "EMA"],
        "priority": 10,
    },
    "UPGRADE": {
        "keywords_fr": ["releve", "objectif de cours", "acheter", "surperformance",
                        "recommandation", "surponderer", "potentiel de hausse"],
        "keywords_en": ["upgrade", "buy", "outperform", "price target", "overweight",
                        "raises target"],
        "priority": 8,
    },
    "DOWNGRADE": {
        "keywords_fr": ["abaisse", "degradation", "sous-performance", "vendre",
                        "sous-ponderer", "alerte"],
        "keywords_en": ["downgrade", "sell", "underperform", "underweight", "cuts target"],
        "priority": 8,
    },
    "CONTRACT": {
        "keywords_fr": ["contrat", "partenariat", "acquisition", "accord", "commande",
                        "alliance", "collaboration", "joint-venture"],
        "keywords_en": ["contract", "partnership", "acquisition", "deal", "agreement",
                        "collaboration", "joint venture", "order"],
        "priority": 7,
    },
    "DIVIDEND": {
        "keywords_fr": ["dividende", "distribution", "coupon", "rendement", "detachement"],
        "keywords_en": ["dividend", "distribution", "payout", "yield"],
        "priority": 6,
    },
    "RESTRUCTURING": {
        "keywords_fr": ["restructuration", "plan social", "cession", "reorganisation",
                        "licenciement", "fermeture"],
        "keywords_en": ["restructuring", "layoffs", "divestiture", "reorganization",
                        "cost cutting"],
        "priority": 6,
    },
    "INSIDER": {
        "keywords_fr": ["dirigeant", "rachat", "declaration de franchissement",
                        "participation", "actionnariat"],
        "keywords_en": ["insider", "buyback", "stake", "shareholder", "holding"],
        "priority": 5,
    },
    "SECTOR_MACRO": {
        "keywords_fr": ["secteur", "industrie", "CAC", "marche", "indice", "BCE",
                        "inflation", "taux", "conjoncture"],
        "keywords_en": ["sector", "industry", "market", "index", "ECB", "inflation",
                        "rate", "macro"],
        "priority": 2,
    },
    "OTHER_POSITIVE": {
        "keywords_fr": ["hausse", "progression", "rebond", "reprise", "en forme"],
        "keywords_en": ["rally", "surge", "gain", "rise", "recovery", "bullish"],
        "priority": 1,
    },
    "OTHER_NEGATIVE": {
        "keywords_fr": ["baisse", "chute", "recul", "perte", "recule", "decroche"],
        "keywords_en": ["decline", "drop", "fall", "loss", "bearish", "plunge"],
        "priority": 1,
    },
}
```

### Interface

```python
class NewsClassifier:
    """Classifie les news par type de catalyseur via regles de mots-cles."""

    def classify(self, title: str, description: str | None) -> str:
        """Retourne le type de catalyseur (EARNINGS, UPGRADE, etc.).

        Cherche dans titre puis description. Si multi-match, prend le plus prioritaire.
        Retourne 'UNKNOWN' si aucun match.
        """

    def classify_news_for_trade(self, catalyseurs: list[dict], news_map: dict) -> dict:
        """Classifie toutes les news liees a un trade.

        Retourne: {
            "primary_type": str,      # Type du catalyseur le plus pertinent
            "types_found": list[str], # Tous les types trouves
            "nb_types": int,          # Diversite des catalyseurs
        }
        """
```

## Composant 2: TechnicalIndicators

**Fichier**: `src/analysis/technical_indicators.py`

**Role**: Calculer l'etat technique complet d'une action a une date donnee.
Inclut les indicateurs standards ET les features de range trading specifiques a Nicolas.

### Indicateurs calcules

```python
class TechnicalIndicators:
    """Calcule les indicateurs techniques pour un ticker."""

    def compute_all(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute toutes les colonnes d'indicateurs au DataFrame de prix.

        Indicateurs standards (via librairie `ta`):
        - rsi_14: RSI 14 periodes
        - macd_line: Ligne MACD (12/26)
        - macd_signal: Ligne signal (9)
        - macd_histogram: Histogramme MACD
        - bollinger_upper, bollinger_lower, bollinger_position: Bandes (20, 2)
        - atr_14: Average True Range 14 periodes
        - sma_20, sma_50: Moyennes mobiles simples
        - ema_9: Moyenne mobile exponentielle

        Features range trading (specifiques au style Nicolas):
        - range_high_10, range_low_10: Plus haut/bas 10 jours
        - range_high_20, range_low_20: Plus haut/bas 20 jours
        - range_position_10: Position dans le range 10j (0=support, 1=resistance)
        - range_position_20: Position dans le range 20j
        - range_amplitude_10: Amplitude du range 10j en %
        - range_amplitude_20: Amplitude du range 20j en %

        Features derivees:
        - volume_ratio_20: Volume / moyenne volume 20j
        - atr_14_pct: ATR(14) / close * 100 (volatilite en %)
        - distance_sma20: (close - sma_20) / sma_20 * 100
        - distance_sma50: (close - sma_50) / sma_50 * 100
        - variation_1j: (close - close_veille) / close_veille * 100
        - variation_5j: (close - close_5j) / close_5j * 100
        """

    def get_indicators_at_date(self, prices_df: pd.DataFrame, date: str) -> dict:
        """Retourne les indicateurs pour une date specifique."""
```

## Composant 3: FeatureEngine

**Fichier**: `src/analysis/feature_engine.py`

**Role**: Assembler le vecteur complet de features pour chaque trade.
C'est le pont entre les donnees brutes et le modele ML.

### Interface

```python
class FeatureEngine:
    """Assemble le vecteur de features pour chaque trade de Nicolas."""

    def __init__(self, db: Database):
        self.db = db
        self.tech = TechnicalIndicators()
        self.classifier = NewsClassifier()
        self.mapper = TickerMapper()

    def build_trade_features(self, trade: dict) -> dict | None:
        """Construit le vecteur de features pour UN trade au moment de l'achat.

        Retourne None si pas assez de donnees prix (< 50 jours avant l'achat).

        Features retournees (~25):
        - 13 features techniques (au jour de l'achat)
        - 6 features catalyseur (type, nombre, score, sentiment, sources)
        - 4 features contexte (jour semaine, nb trades precedents, win rate, jours depuis dernier)
        - 1 target: rendement_brut_pct > 0 (gagnant/perdant)
        """

    def build_all_features(self) -> pd.DataFrame:
        """Construit la matrice de features pour TOUS les trades clotures.

        Retourne un DataFrame de ~141 lignes x ~25 colonnes + 1 target.
        Les trades sans donnees prix suffisantes sont exclus (avec warning).
        """

    def get_feature_names(self) -> list[str]:
        """Liste ordonnee des noms de features."""
```

### Vecteur de features complet

| # | Feature | Source | Description |
|---|---------|--------|-------------|
| 1 | range_position_10 | TechnicalIndicators | Position dans range 10j (0=support, 1=resistance) |
| 2 | range_position_20 | TechnicalIndicators | Position dans range 20j |
| 3 | range_amplitude_10 | TechnicalIndicators | Amplitude range 10j en % |
| 4 | range_amplitude_20 | TechnicalIndicators | Amplitude range 20j en % |
| 5 | rsi_14 | TechnicalIndicators | RSI 14 periodes |
| 6 | macd_histogram | TechnicalIndicators | Histogramme MACD |
| 7 | bollinger_position | TechnicalIndicators | Position dans bandes Bollinger |
| 8 | volume_ratio_20 | TechnicalIndicators | Volume / moyenne 20j |
| 9 | atr_14_pct | TechnicalIndicators | Volatilite en % |
| 10 | variation_1j | TechnicalIndicators | Variation veille |
| 11 | variation_5j | TechnicalIndicators | Variation 5 jours |
| 12 | distance_sma20 | TechnicalIndicators | Ecart % au SMA20 |
| 13 | distance_sma50 | TechnicalIndicators | Ecart % au SMA50 |
| 14 | catalyst_type | NewsClassifier | Type principal encode (one-hot ou ordinal) |
| 15 | nb_catalysts | CatalystMatcher | Nombre de catalyseurs J-3/J+1 |
| 16 | best_catalyst_score | CatalystMatcher | Score du meilleur catalyseur |
| 17 | has_text_match | CatalystMatcher | Au moins 1 match texte direct |
| 18 | sentiment_avg | News table | Sentiment moyen (quand dispo, sinon 0) |
| 19 | nb_news_sources | News/Catalyseurs | Nombre de sources differentes |
| 20 | day_of_week | Trade date | Jour (0=lundi, 4=vendredi) |
| 21 | nb_previous_trades | Historique Nicolas | Trades precedents sur cette action |
| 22 | previous_win_rate | Historique Nicolas | Win rate perso sur cette action |
| 23 | days_since_last_trade | Historique Nicolas | Jours depuis dernier trade |
| TARGET | is_winner | Trade resultat | rendement_brut_pct > 0 (1/0) |

## Composant 4: Trainer

**Fichier**: `src/model/trainer.py`

**Role**: Entrainer le modele XGBoost sur les decisions de Nicolas.

### Pipeline

```python
class Trainer:
    """Entraine un XGBoost sur les trades historiques de Nicolas."""

    def __init__(self, db: Database):
        self.db = db
        self.feature_engine = FeatureEngine(db)
        self.model = None

    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Construit X (features) et y (target) a partir des trades clotures."""

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Entraine le modele XGBoost.

        Parametres XGBoost:
        - objective: binary:logistic
        - max_depth: 4 (petit dataset = eviter overfitting)
        - n_estimators: 100
        - learning_rate: 0.1
        - scale_pos_weight: ratio negatifs/positifs (~0.14)
        - eval_metric: logloss

        Retourne: dict avec metriques d'entrainement.
        """

    def walk_forward_validate(self, X: pd.DataFrame, y: pd.Series,
                               split_date: str = "2025-12-01") -> dict:
        """Validation walk-forward: train sur avant split_date, test sur apres.

        Retourne: {
            "accuracy", "precision", "recall", "f1",
            "confusion_matrix", "baseline_accuracy",
            "predictions": list de dicts avec details par trade
        }
        """

    def save_model(self, path: str = "data/models/nicolas_v1.joblib"):
        """Sauvegarde le modele entraine."""

    def load_model(self, path: str = "data/models/nicolas_v1.joblib"):
        """Charge un modele sauvegarde."""
```

## Composant 5: Evaluator

**Fichier**: `src/model/evaluator.py`

**Role**: Analyser les performances du modele et comprendre ses decisions.

### Interface

```python
class Evaluator:
    """Analyse les performances et l'interpretabilite du modele."""

    def feature_importance(self, model, feature_names: list[str]) -> pd.DataFrame:
        """Retourne les features triees par importance.

        Si range_position et catalyst_type sont dans le top 5, le modele
        a bien appris le style Nicolas.
        """

    def error_analysis(self, predictions: list[dict]) -> dict:
        """Analyse les trades mal predits.

        Retourne: {
            "false_positives": trades predits gagnants mais perdants,
            "false_negatives": trades predits perdants mais gagnants,
            "patterns": observations sur les erreurs,
        }
        """

    def compare_to_baseline(self, y_true, y_pred) -> dict:
        """Compare le modele au baseline naif (toujours predire gagnant)."""

    def print_report(self, results: dict):
        """Affiche un rapport complet en console."""
```

## Script CLI

### `scripts/train_model.py`

```bash
uv run python scripts/train_model.py                # Entrainer + evaluer
uv run python scripts/train_model.py --features      # Explorer les features seulement
uv run python scripts/train_model.py --importance     # Feature importance du modele
uv run python scripts/train_model.py --errors         # Analyse des erreurs
```

### `scripts/analyze_features.py`

```bash
uv run python scripts/analyze_features.py            # Stats descriptives des features
uv run python scripts/analyze_features.py --trade 42  # Features d'un trade specifique
```

## Criteres de succes

| Critere | Seuil | Pourquoi |
|---------|-------|----------|
| Precision sur test set | > 88% | Doit battre le baseline naif |
| Feature importance | range_position dans top 5 | Confirme que le modele capte le range trading |
| Feature importance | catalyst_type dans top 10 | Confirme que les catalyseurs comptent |
| Faux positifs | < 5 sur le test set | Un faux positif = une perte, on veut les minimiser |
| Interpretabilite | Rapport lisible | Nicolas doit comprendre POURQUOI le modele recommande |

## Dependances entre composants

```
NewsClassifier (0 dep interne, utilise regles mots-cles)
TechnicalIndicators (0 dep interne, utilise librairie `ta` + pandas)
    |
    v
FeatureEngine (utilise NewsClassifier, TechnicalIndicators, Database, TickerMapper)
    |
    v
Trainer (utilise FeatureEngine, Database)
Evaluator (utilise Trainer output)
```

## Risques et mitigations

| Risque | Impact | Mitigation |
|--------|--------|------------|
| 141 samples = petit dataset | Overfitting | max_depth=4, regularisation, walk-forward |
| Desequilibre 88/12 | Modele biaise vers "gagnant" | scale_pos_weight, analyser les faux positifs |
| Donnees prix manquantes (3 tickers delistes) | Features NULL | Exclure ces trades (~15 sur 141) |
| Sentiment disponible sur 38% des news | Features creuses | Utiliser 0 par defaut, feature optionnelle |
| Classification news par mots-cles imparfaite | Mauvais types | Valider manuellement un echantillon |
