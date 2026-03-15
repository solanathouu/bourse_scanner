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
from datetime import datetime, timedelta

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
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "scanner.log")

# Ajouter un sink fichier pour persister les logs
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger.add(LOG_PATH, rotation="10 MB", retention="7 days", level="INFO")


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


def collect_alpha_vantage(db: Database, watchlist: list[dict]):
    """Collecte news Alpha Vantage pour les derniers jours (25 req/jour max)."""
    from src.data_collection.alpha_vantage_collector import AlphaVantageCollector

    collector = AlphaVantageCollector(db)
    if not collector.api_key:
        return

    today = datetime.now().strftime("%Y%m%d")
    from_date = (datetime.now() - timedelta(days=3)).strftime("%Y%m%d")
    total = 0

    tickers_done = 0
    for item in watchlist:
        if item.get("etf") or tickers_done >= 25:
            continue
        try:
            count = collector.collect_for_action(
                item["name"], item["ticker"],
                f"{from_date}T0000", f"{today}T2359",
            )
            total += count
            tickers_done += 1
            import time
            time.sleep(2)
        except Exception as e:
            logger.error(f"Alpha Vantage erreur {item['ticker']}: {e}")

    logger.info(f"Alpha Vantage: {total} news pour {tickers_done} tickers")


def collect_marketaux(db: Database, watchlist: list[dict]):
    """Collecte news Marketaux pour les derniers jours."""
    from src.data_collection.marketaux_collector import MarketauxCollector

    collector = MarketauxCollector(db)
    if not collector.api_key:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    total = 0

    for item in watchlist:
        if item.get("etf"):
            continue
        try:
            count = collector.collect_for_action(
                item["name"], item["ticker"], from_date, today,
            )
            total += count
            import time
            time.sleep(2)
        except Exception as e:
            logger.error(f"Marketaux erreur {item['ticker']}: {e}")

    logger.info(f"Marketaux: {total} news collectees")


def collect_newsdata(db: Database, watchlist: list[dict]):
    """Collecte news Newsdata.io pour les derniers jours."""
    try:
        from src.data_collection.newsdata_collector import NewsdataCollector
    except Exception:
        logger.warning("NewsdataCollector non disponible")
        return

    try:
        collector = NewsdataCollector(db)
    except ValueError:
        logger.warning("NEWSDATA_API_KEY manquante")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    total = 0

    for item in watchlist:
        if item.get("etf"):
            continue
        try:
            count = collector.collect_for_action(
                item["name"], item["ticker"], from_date, today,
            )
            total += count
            import time
            time.sleep(1)
        except Exception as e:
            logger.error(f"Newsdata erreur {item['ticker']}: {e}")

    logger.info(f"Newsdata.io: {total} news collectees")


def score_sentiment_llm(db: Database):
    """Score le sentiment des news sans score via GPT-4o-mini (batch de 30)."""
    try:
        from src.analysis.llm_sentiment import LLMSentimentScorer
        scorer = LLMSentimentScorer(db)
        result = scorer.score_all_unscored(batch_size=30)
        logger.info(
            f"LLM Sentiment: {result['scored']}/{result['total']} scorees, "
            f"{result['errors']} erreurs"
        )
    except Exception as e:
        logger.error(f"LLM Sentiment erreur: {e}")


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


def run_daily_review(db, config, telegram, dry_run):
    """Review des signaux J+3 et envoi recap Telegram."""
    from src.feedback.signal_reviewer import SignalReviewer
    from src.feedback.performance_tracker import PerformanceTracker

    feedback_cfg = config.get("feedback", {})
    reviewer = SignalReviewer(db, win_threshold=feedback_cfg.get("win_threshold", 4.5))
    reviews = reviewer.review_pending()

    if reviews and telegram and not dry_run:
        base_threshold = config.get("scoring", {}).get("threshold", 0.75)
        tracker = PerformanceTracker(db, base_threshold=base_threshold)
        summary = tracker.get_daily_summary(reviews)
        telegram.send_alert_sync(summary)

    logger.info(f"Review quotidienne: {len(reviews)} signaux reviewes")


def update_filter_rules(db, config):
    """Mise a jour des stats catalyseur et seuil adaptatif."""
    import json as _json
    from src.feedback.performance_tracker import PerformanceTracker

    base_threshold = config.get("scoring", {}).get("threshold", 0.75)
    tracker = PerformanceTracker(db, base_threshold=base_threshold)
    rules = tracker.generate_filter_rules()

    threshold = tracker.compute_adaptive_threshold()
    if threshold != base_threshold:
        db.insert_filter_rule({
            "rule_type": "ADAPTIVE_THRESHOLD",
            "rule_json": _json.dumps({"threshold": threshold}),
            "win_rate": None, "sample_size": None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": 1,
        })

    logger.info(f"Stats mises a jour: {len(rules)} catalyseurs, seuil={threshold:.2f}")


def check_retrain(db, config, telegram, dry_run, predictor=None):
    """Verifie si re-entrainement necessaire et execute."""
    from src.feedback.model_retrainer import ModelRetrainer

    feedback_cfg = config.get("feedback", {})
    model_path = config.get("scoring", {}).get("model_path", "data/models/nicolas_v1.joblib")

    retrainer = ModelRetrainer(
        db, min_reviews_for_retrain=feedback_cfg.get("min_reviews_retrain", 50)
    )

    if not retrainer.should_retrain():
        logger.info("Pas assez de reviews pour re-entrainer")
        return

    result = retrainer.retrain_with_validation(model_path)
    report = retrainer.format_retrain_report(result)

    if result.get("deployed") and predictor is not None:
        new_path = result.get("new_path", model_path)
        predictor.reload_model(new_path)
        logger.info(f"Predictor recharge avec {new_path}")

    if telegram and not dry_run:
        telegram.send_alert_sync(report)

    logger.info(f"Re-entrainement: deployed={result['deployed']}")


def send_weekly_summary(db, config, telegram, dry_run):
    """Envoi du resume hebdomadaire."""
    from src.feedback.performance_tracker import PerformanceTracker

    base_threshold = config.get("scoring", {}).get("threshold", 0.75)
    tracker = PerformanceTracker(db, base_threshold=base_threshold)

    today = datetime.now()
    week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end = today.strftime("%Y-%m-%d")

    summary = tracker.get_weekly_summary(week_start, week_end)

    if telegram and not dry_run:
        telegram.send_alert_sync(summary)

    logger.info("Resume hebdomadaire envoye")


def collect_orderbook(db: Database, watchlist: list[dict]):
    """Collecte les carnets d'ordres Boursorama pour la watchlist."""
    from src.data_collection.orderbook_collector import OrderBookCollector

    collector = OrderBookCollector(db)
    result = collector.collect_all_watchlist(watchlist)
    logger.info(
        f"Orderbook: {result['collected']} collectes, "
        f"{result['errors']} erreurs"
    )


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

    # Charger les stats catalyseur pour enrichir les alertes
    from src.feedback.performance_tracker import PerformanceTracker
    tracker = PerformanceTracker(signal_filter.db)
    catalyst_stats = tracker.get_catalyst_stats()

    # Filtrer
    filtered = signal_filter.filter_signals(signals)

    if not filtered:
        logger.info("Aucun signal retenu apres filtrage")
        return

    # Envoyer les alertes
    for signal in filtered:
        signal["catalyst_stats"] = catalyst_stats
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


def _get_active_model_path(db: Database, scoring_config: dict) -> str:
    """Retourne le chemin du modele actif (BDD) ou le defaut (config)."""
    default_path = scoring_config.get("model_path", "data/models/nicolas_v1.joblib")
    active_model = db.get_active_model_version()

    if active_model is None:
        db.insert_model_version({
            "version": "v1",
            "file_path": default_path,
            "trained_at": "2026-02-20 00:00:00",
            "training_signals": 0,
            "accuracy": 0.0, "precision_score": 0.0,
            "recall": 0.0, "f1": 0.0,
            "is_active": 1,
            "notes": "Initial model from historical trades",
        })
        return default_path

    logger.info(f"Modele actif: {active_model['version']} ({active_model['file_path']})")
    return active_model["file_path"]


def run_once(dry_run: bool = False):
    """Execute un seul cycle de scoring."""
    config = load_config()
    watchlist = config["watchlist"]
    scoring_config = config.get("scoring", {})

    db = Database(DB_PATH)
    db.init_db()

    # Predictor — charger le modele actif de la BDD
    model_path = _get_active_model_path(db, scoring_config)

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
    feedback_config = config.get("feedback", {})

    db = Database(DB_PATH)
    db.init_db()

    # Predictor — charger le modele actif de la BDD
    model_path = _get_active_model_path(db, scoring_config)

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

    # Job 5: Alpha Vantage — quotidien (lun-ven)
    scheduler.add_job(
        collect_alpha_vantage, CronTrigger(
            hour=sched_config.get("alpha_vantage_hour", 8),
            minute=0,
            day_of_week="mon-fri",
        ),
        args=[db, watchlist],
        id="collect_alpha_vantage",
        name="Alpha Vantage news",
    )

    # Job 6: Marketaux — quotidien (lun-ven)
    scheduler.add_job(
        collect_marketaux, CronTrigger(
            hour=sched_config.get("marketaux_hour", 12),
            minute=0,
            day_of_week="mon-fri",
        ),
        args=[db, watchlist],
        id="collect_marketaux",
        name="Marketaux news",
    )

    # Job 7: Newsdata.io — quotidien (lun-ven)
    scheduler.add_job(
        collect_newsdata, CronTrigger(
            hour=sched_config.get("newsdata_hour", 14),
            minute=0,
            day_of_week="mon-fri",
        ),
        args=[db, watchlist],
        id="collect_newsdata",
        name="Newsdata.io news",
    )

    # Job 8: LLM Sentiment — toutes les 2h (lun-ven 8h-19h)
    scheduler.add_job(
        score_sentiment_llm, IntervalTrigger(
            minutes=sched_config.get("sentiment_interval_min", 120),
            start_date="2026-01-01 08:00:00",
        ),
        args=[db],
        id="score_sentiment",
        name="LLM Sentiment scoring",
    )

    # Job 9: Review J+3 — quotidien a 18h (lun-ven)
    scheduler.add_job(
        run_daily_review,
        CronTrigger(
            hour=feedback_config.get("review_hour", 18),
            minute=0,
            day_of_week="mon-fri",
        ),
        args=[db, config, telegram, dry_run],
        id="daily_review",
        name="Review J+3",
    )

    # Job 10: Update regles — quotidien a 18h30 (lun-ven)
    scheduler.add_job(
        update_filter_rules,
        CronTrigger(
            hour=feedback_config.get("rules_update_hour", 18),
            minute=30,
            day_of_week="mon-fri",
        ),
        args=[db, config],
        id="update_rules",
        name="Update regles",
    )

    # Job 11: Check retrain — quotidien a 19h (lun-ven)
    scheduler.add_job(
        check_retrain,
        CronTrigger(
            hour=feedback_config.get("retrain_hour", 19),
            minute=0,
            day_of_week="mon-fri",
        ),
        args=[db, config, telegram, dry_run, predictor],
        id="check_retrain",
        name="Check retrain quotidien",
    )

    # Job 12: Resume hebdo — dimanche 20h
    scheduler.add_job(
        send_weekly_summary,
        CronTrigger(
            hour=feedback_config.get("weekly_hour", 20),
            minute=0,
            day_of_week=feedback_config.get("weekly_day", "sun"),
        ),
        args=[db, config, telegram, dry_run],
        id="weekly_summary",
        name="Resume hebdomadaire",
    )

    # Job 13: Carnet d'ordres — toutes les 15 min (lun-ven 9h-17h35)
    scheduler.add_job(
        collect_orderbook, IntervalTrigger(
            minutes=sched_config.get("orderbook_interval_min", 15),
            start_date="2026-01-01 09:00:00",
        ),
        args=[db, watchlist],
        id="collect_orderbook",
        name="Carnet d'ordres",
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
    print(f"  - News RSS: toutes les {sched_config.get('news_interval_min', 10)} min")
    print(f"  - Scoring: toutes les {sched_config.get('scoring_interval_min', 60)} min")
    print(f"  - Fondamentaux: quotidien a {sched_config.get('fundamentals_hour', 7)}h")
    print(f"  - Alpha Vantage: quotidien a {sched_config.get('alpha_vantage_hour', 8)}h")
    print(f"  - Marketaux: quotidien a {sched_config.get('marketaux_hour', 12)}h")
    print(f"  - Newsdata.io: quotidien a {sched_config.get('newsdata_hour', 14)}h")
    print(f"  - LLM Sentiment: toutes les {sched_config.get('sentiment_interval_min', 120)} min")
    print(f"  - Review J+3: quotidien a {feedback_config.get('review_hour', 18)}h")
    print(f"  - Update regles: quotidien a {feedback_config.get('rules_update_hour', 18)}h30")
    print(f"  - Check retrain: quotidien a {feedback_config.get('retrain_hour', 19)}h (lun-ven)")
    print(f"  - Resume hebdo: {feedback_config.get('weekly_day', 'dim')} a {feedback_config.get('weekly_hour', 20)}h")
    print(f"  - Carnet d'ordres: toutes les {sched_config.get('orderbook_interval_min', 15)} min")
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
