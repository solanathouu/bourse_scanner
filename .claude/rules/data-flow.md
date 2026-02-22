# Flux de donnees — PEA Scanner

## Pipeline global

```
    ETAPE 1 (offline, one-shot)        ETAPE 2-5 (temps reel, boucle)
    ========================           ================================

    PDF SG (~200 fichiers)             yfinance / APIs / RSS / Scrapers
           |                                      |
           v                                      v
    pdf_parser.py                      price_collector.py
    (extraction texte + regex)         news_collector.py
           |                           fundamental_collector.py
           v                                      |
    trade_matcher.py                              v
    (reconstruction FIFO)              Tables: prices, news,
           |                           indicators, fundamentals
           v                                      |
    Tables: executions,                           v
    trades_complets                    feature_engine.py
                                       (vecteur de features)
                                              |
                                              v
                                       predictor.py
                                       (scoring XGBoost)
                                              |
                                              v
                                       signal_filter.py
                                       (filtrage intelligent)
                                              |
                                              v
                                       telegram_bot.py
                                       (alerte si score > seuil)
```

## Tables SQLite et leur role

| Table | Alimentee par | Lue par | Contenu |
|-------|--------------|---------|---------|
| `executions` | pdf_parser | trade_matcher | 1 ligne = 1 PDF parse |
| `trades_complets` | trade_matcher | catalyst_matcher, trainer | Trades reconstitues |
| `trade_catalyseurs` | catalyst_matcher | trainer | News associees aux trades |
| `prices` | price_collector | feature_engine | OHLCV par ticker |
| `indicators` | feature_engine | predictor | RSI, MACD, etc. |
| `news` | news_collector | sentiment_analyzer | Articles collectes |
| `signals` | predictor + signal_filter | telegram_bot | Signaux emis |

## Scheduler (APScheduler)

```
Toutes les 1-2 min:  price_collector -> table prices
Toutes les 5-10 min: news_collector -> table news
Toutes les heures:   feature_engine -> table indicators
                     predictor -> table signals
                     signal_filter -> telegram si signal
Quotidien:           fundamental_collector -> table fundamentals
```

## Format des donnees cles

- **Dates**: `YYYY-MM-DD` (stockage), `DD/MM/YYYY` (affichage FR)
- **Heures**: `HH:MM:SS`
- **Montants**: float, point decimal, 2 decimales
- **Pourcentages**: float en % (`4.25` = 4.25%, pas 0.0425)
- **Sentiment**: float de -1.0 (negatif) a +1.0 (positif)
- **Score confiance**: float de 0.0 a 1.0
- **ISIN**: string `FRXXXXXXXXXX` (12 caracteres)
- **Ticker Yahoo**: string `MC.PA` (symbole + .PA pour Paris)
