"""Pipeline temps reel PEA Scanner.

Collecte les prix et news, score la watchlist avec le modele de Nicolas,
et envoie des alertes Telegram quand le score depasse le seuil.

Usage:
    uv run python scripts/run_scanner.py           # Lancer le scanner (boucle infinie)
    uv run python scripts/run_scanner.py --once     # Scorer une fois et quitter
    uv run python scripts/run_scanner.py --dry-run  # Scorer sans envoyer de Telegram
"""

import argparse
import os
import sys
import signal as sig

import yaml
from dotenv import load_dotenv
from loguru import logger

# Ajouter le projet au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.database import Database
from src.model.predictor import Predictor
from src.alerts.signal_filter import SignalFilter
from src.alerts.formatter import AlertFormatter
from src.alerts.telegram_bot import TelegramBot


load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trades.db")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.yaml")


def load_config() -> dict:
    """Charge la configuration de la watchlist."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def refresh_prices(db: Database, watchlist: list[dict]):
    """Collecte les prix pour tous les tickers de la watchlist."""
    from src.data_collection.price_collector import PriceCollector

    collector = PriceCollector(db)
    tickers = [item["ticker"] for item in watchlist]
    total = 0

    for ticker in tickers:
        try:
            count = collector.collect_recent(ticker, days=5)
            total += count
        except Exception as e:
            logger.error(f"Erreur prix {ticker}: {e}")

    logger.info(f"Refresh prix: {total} nouveaux prix pour {len(tickers)} tickers")


def collect_news(db: Database):
    """Collecte les news RSS."""
    from src.data_collection.rss_collector import RSSCollector

    collector = RSSCollector(db)
    result = collector.collect_all()
    logger.info(
        f"Collecte news: {result['total_news']} nouvelles, "
        f"{result['feeds_ok']} feeds OK"
    )


def refresh_fundamentals(db: Database, watchlist: list[dict]):
    """Collecte les fondamentaux pour les tickers non-ETF."""
    from src.data_collection.fundamental_collector import FundamentalCollector

    collector = FundamentalCollector(db)
    tickers = [item["ticker"] for item in watchlist if not item.get("etf")]

    for ticker in tickers:
        try:
            collector.collect(ticker)
        except Exception as e:
            logger.error(f"Erreur fondamentaux {ticker}: {e}")

    logger.info(f"Refresh fondamentaux: {len(tickers)} tickers")


def score_and_alert(predictor: Predictor, signal_filter: SignalFilter,
                    formatter: AlertFormatter, telegram: TelegramBot | None,
                    watchlist: list[dict], dry_run: bool = False):
    """Score la watchlist, filtre et envoie les alertes."""
    # Verifier horaires marche
    if not dry_run and not signal_filter.is_market_hours():
        logger.info("Marche ferme, skip scoring")
        return

    # Scorer la watchlist
    signals = predictor.score_watchlist(watchlist)

    if not signals:
        logger.info("Aucun signal genere")
        return

    # Afficher tous les scores
    print("\n--- Scores watchlist ---")
    for s in signals:
        name = s.get("name", s["ticker"])
        emoji = "+" if s["score"] >= signal_filter.threshold else " "
        print(f"  {emoji} {name:25s} {s['ticker']:12s} score={s['score']:.2f} "
              f"cat={s.get('catalyst_type', 'N/A'):15s} "
              f"{s.get('technical_summary', '')}")
    print()

    # Filtrer
    filtered = signal_filter.filter_signals(signals)

    if not filtered:
        logger.info("Aucun signal retenu apres filtrage")
        return

    # Envoyer les alertes
    for signal in filtered:
        message = formatter.format_signal(signal)

        if dry_run:
            print(f"\n[DRY-RUN] Message Telegram pour {signal['ticker']}:")
            print(message)
            print()
        elif telegram:
            success = telegram.send_alert_sync(message)
            if success:
                signal_filter.record_signal(signal)
        else:
            print(f"\n[NO TELEGRAM] {signal['ticker']} score={signal['score']:.2f}")

    logger.info(f"Alertes: {len(filtered)} signaux envoyes")


def run_once(dry_run: bool = False):
    """Execute un seul cycle de scoring."""
    config = load_config()
    watchlist = config["watchlist"]
    scoring_config = config.get("scoring", {})

    db = Database(DB_PATH)
    db.init_db()

    # Predictor
    model_path = scoring_config.get("model_path", "data/models/nicolas_v1.joblib")
    try:
        predictor = Predictor(db, model_path=model_path)
    except Exception as e:
        logger.error(f"Impossible de charger le modele: {e}")
        print(f"\nERREUR: modele non trouve a '{model_path}'")
        print("Lancer d'abord: uv run python scripts/train_model.py")
        return

    # Signal filter
    filter_config = {
        "threshold": scoring_config.get("threshold", 0.75),
        "cooldown_hours": scoring_config.get("cooldown_hours", 24),
        "market_open": config.get("market_hours", {}).get("scoring_open", "09:30"),
        "market_close": config.get("market_hours", {}).get("scoring_close", "17:30"),
    }
    signal_filter = SignalFilter(db, filter_config)

    # Formatter
    formatter = AlertFormatter()

    # Telegram
    telegram = None
    if not dry_run:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            telegram = TelegramBot(token, chat_id)
        else:
            logger.warning("TELEGRAM_BOT_TOKEN/CHAT_ID non configures")

    print(f"PEA Scanner — scoring {len(watchlist)} valeurs "
          f"(seuil={filter_config['threshold']}, "
          f"{'dry-run' if dry_run else 'live'})")

    score_and_alert(predictor, signal_filter, formatter, telegram,
                    watchlist, dry_run=dry_run)


def run_scheduler(dry_run: bool = False):
    """Lance le scanner en boucle avec APScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger

    config = load_config()
    watchlist = config["watchlist"]
    scoring_config = config.get("scoring", {})
    sched_config = config.get("scheduler", {})
    market_config = config.get("market_hours", {})

    db = Database(DB_PATH)
    db.init_db()

    # Predictor
    model_path = scoring_config.get("model_path", "data/models/nicolas_v1.joblib")
    try:
        predictor = Predictor(db, model_path=model_path)
    except Exception as e:
        logger.error(f"Impossible de charger le modele: {e}")
        print(f"\nERREUR: modele non trouve a '{model_path}'")
        return

    # Composants
    filter_config = {
        "threshold": scoring_config.get("threshold", 0.75),
        "cooldown_hours": scoring_config.get("cooldown_hours", 24),
        "market_open": market_config.get("scoring_open", "09:30"),
        "market_close": market_config.get("scoring_close", "17:30"),
    }
    signal_filter = SignalFilter(db, filter_config)
    formatter = AlertFormatter()

    telegram = None
    if not dry_run:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            telegram = TelegramBot(token, chat_id)

    scheduler = BlockingScheduler(timezone=market_config.get("timezone", "Europe/Paris"))

    # Job 1: Refresh prix — toutes les 2 min (lun-ven 9h-17h35)
    scheduler.add_job(
        refresh_prices, IntervalTrigger(
            minutes=sched_config.get("prices_interval_min", 2),
            start_date="2026-01-01 09:00:00",
        ),
        args=[db, watchlist],
        id="refresh_prices",
        name="Refresh prix",
    )

    # Job 2: Collecte news — toutes les 10 min (lun-ven 8h-18h)
    scheduler.add_job(
        collect_news, IntervalTrigger(
            minutes=sched_config.get("news_interval_min", 10),
            start_date="2026-01-01 08:00:00",
        ),
        args=[db],
        id="collect_news",
        name="Collecte news",
    )

    # Job 3: Scoring + alertes — toutes les 60 min
    scheduler.add_job(
        score_and_alert, IntervalTrigger(
            minutes=sched_config.get("scoring_interval_min", 60),
            start_date="2026-01-01 09:30:00",
        ),
        args=[predictor, signal_filter, formatter, telegram, watchlist, dry_run],
        id="score_and_alert",
        name="Scoring + alertes",
    )

    # Job 4: Fondamentaux — quotidien a 7h
    scheduler.add_job(
        refresh_fundamentals, CronTrigger(
            hour=sched_config.get("fundamentals_hour", 7),
            minute=0,
            day_of_week="mon-fri",
        ),
        args=[db, watchlist],
        id="refresh_fundamentals",
        name="Refresh fondamentaux",
    )

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Arret du scanner...")
        scheduler.shutdown(wait=False)

    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    nb_actions = len([w for w in watchlist if not w.get("etf")])
    nb_etfs = len([w for w in watchlist if w.get("etf")])

    print(f"\nPEA Scanner demarre")
    print(f"  Watchlist: {nb_actions} actions + {nb_etfs} ETFs")
    print(f"  Seuil: {filter_config['threshold']}")
    print(f"  Mode: {'dry-run' if dry_run else 'live'}")
    print(f"  Telegram: {'configure' if telegram else 'non configure'}")
    print(f"\nJobs programmes:")
    print(f"  - Prix: toutes les {sched_config.get('prices_interval_min', 2)} min")
    print(f"  - News: toutes les {sched_config.get('news_interval_min', 10)} min")
    print(f"  - Scoring: toutes les {sched_config.get('scoring_interval_min', 60)} min")
    print(f"  - Fondamentaux: quotidien a {sched_config.get('fundamentals_hour', 7)}h")
    print(f"\nCtrl+C pour arreter\n")

    scheduler.start()


def main():
    parser = argparse.ArgumentParser(description="PEA Scanner — pipeline temps reel")
    parser.add_argument("--once", action="store_true",
                        help="Scorer une fois et quitter")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scorer sans envoyer de Telegram")
    args = parser.parse_args()

    if args.once:
        run_once(dry_run=args.dry_run)
    else:
        run_scheduler(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
