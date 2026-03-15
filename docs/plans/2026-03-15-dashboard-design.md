# Dashboard Monitoring PEA Scanner — Design Spec

## Goal

Dashboard web en lecture seule pour monitorer le bot PEA Scanner en temps reel. Aucune influence sur l'algorithme — observation pure.

## Architecture

- FastAPI server sur le VPS, port 8050, tmux session "dashboard"
- Lit trades.db en lecture seule (meme base que le scanner)
- Templates HTML Jinja2 + Chart.js (CDN) pour les graphiques
- CSS minimal, pas de framework frontend

## Pages

### Page d'accueil (`/`)
- KPIs: win rate global, nb signaux, modele actif (version + f1), nb reviews
- Graphique: evolution win rate par semaine (Chart.js line chart)
- Tableau: 20 derniers signaux (score, ticker, catalyseur, resultat WIN/LOSS/NEUTRAL/en attente)

### Page news (`/news`)
- Liste des dernieres news, titre uniquement, badge sentiment (vert/rouge/gris)
- Filtre par ticker (dropdown)
- Clic sur news -> detail: type catalyseur LLM, confiance, pertinence, explication, lien source

### Page signal (`/signal/{id}`)
- Indicateurs techniques: RSI, volume ratio, range position, MACD
- Catalyseur identifie + news declencheuse
- Lien TradingView (`https://fr.tradingview.com/chart/?symbol=EURONEXT:{symbol}`)
- Resultat J+3 si reviewe (prix signal -> prix J+3, performance %)
- Features feedback: win rate historique catalyseur + ticker

### Page portfolio (`/portfolio`)
- Graphique performance cumulee: simulation "si on avait achete a chaque signal et vendu a J+3"
- Filtre: tous / CONFIRME uniquement / par catalyseur
- Tableau trades virtuels (entree, sortie, perf)
- LECTURE SEULE — aucune influence sur l'algorithme

## Fichiers

```
src/dashboard/
  app.py              # FastAPI app, routes, queries SQLite read-only
  templates/
    base.html          # Layout (nav, header)
    index.html         # Accueil (KPIs + graphique + signaux)
    news.html          # Fil de news
    signal.html        # Detail signal
    portfolio.html     # Portefeuille virtuel
  static/
    style.css          # CSS minimal
```

## Contraintes
- Zero ecriture dans trades.db
- Pas de framework JS lourd
- Tourne a cote du scanner sans interference
- FastAPI deja dans les deps du projet
