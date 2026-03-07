"""Pipeline de feedback — review, regles adaptatives, re-entrainement.

Usage:
    uv run python scripts/run_feedback.py              # Review + update rules
    uv run python scripts/run_feedback.py --stats       # Stats globales
    uv run python scripts/run_feedback.py --retrain     # Forcer re-entrainement
    uv run python scripts/run_feedback.py --dry-run     # Review sans Telegram
    uv run python scripts/run_feedback.py --weekly      # Envoyer resume hebdo
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import yaml
from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.database import Database
from src.feedback.signal_reviewer import SignalReviewer
from src.feedback.performance_tracker import PerformanceTracker
from src.feedback.model_retrainer import ModelRetrainer

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "trades.db")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "watchlist.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_telegram():
    from src.alerts.telegram_bot import TelegramBot
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if token and chat_id:
        return TelegramBot(token, chat_id)
    return None


def run_review(dry_run=False):
    db = Database(DB_PATH)
    db.init_db()
    config = load_config()
    base_threshold = config.get("scoring", {}).get("threshold", 0.75)

    reviewer = SignalReviewer(db, win_threshold=config.get("feedback", {}).get("win_threshold", 4.5))
    reviews = reviewer.review_pending()

    if reviews:
        tracker = PerformanceTracker(db, base_threshold=base_threshold)
        summary = tracker.get_daily_summary(reviews)
        if dry_run:
            print(f"\n[DRY-RUN] Review quotidienne:\n{summary}\n")
        else:
            telegram = get_telegram()
            if telegram:
                telegram.send_alert_sync(summary)

    tracker = PerformanceTracker(db, base_threshold=base_threshold)
    rules = tracker.generate_filter_rules()

    threshold = tracker.compute_adaptive_threshold()
    if threshold != base_threshold:
        db.insert_filter_rule({
            "rule_type": "ADAPTIVE_THRESHOLD",
            "rule_json": json.dumps({"threshold": threshold}),
            "win_rate": None, "sample_size": None,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "active": 1,
        })

    print(f"Review: {len(reviews)} signaux reviewes")
    print(f"Regles: {len(rules)} regles actives")
    print(f"Seuil adaptatif: {threshold:.2f}")


def run_stats():
    db = Database(DB_PATH)
    db.init_db()
    stats = db.get_review_stats()
    print(f"\n=== Stats Feedback Loop ===")
    print(f"Total reviews: {stats['total']}")
    print(f"  WIN: {stats['wins']}")
    print(f"  LOSS: {stats['losses']}")
    print(f"  NEUTRAL: {stats['neutrals']}")
    if stats['total'] > 0:
        print(f"Win rate: {stats['wins'] / stats['total']:.1%}")
    rules = db.get_active_filter_rules()
    print(f"\nRegles actives: {len(rules)}")
    for r in rules:
        print(f"  - {r['rule_type']}: {r['rule_json']}")
    active_model = db.get_active_model_version()
    if active_model:
        print(f"\nModele actif: {active_model['version']}")


def run_retrain(dry_run=False):
    db = Database(DB_PATH)
    db.init_db()
    config = load_config()
    model_path = config.get("scoring", {}).get("model_path", "data/models/nicolas_v1.joblib")
    retrainer = ModelRetrainer(db, min_reviews_for_retrain=0)
    result = retrainer.retrain_with_validation(model_path)
    report = retrainer.format_retrain_report(result)
    if dry_run:
        print(f"\n[DRY-RUN] Rapport re-entrainement:\n{report}\n")
    else:
        telegram = get_telegram()
        if telegram:
            telegram.send_alert_sync(report)
    print(report)


def run_weekly(dry_run=False):
    db = Database(DB_PATH)
    db.init_db()
    config = load_config()
    base_threshold = config.get("scoring", {}).get("threshold", 0.75)
    today = datetime.now()
    week_start = (today - timedelta(days=7)).strftime("%Y-%m-%d")
    week_end = today.strftime("%Y-%m-%d")
    tracker = PerformanceTracker(db, base_threshold=base_threshold)
    summary = tracker.get_weekly_summary(week_start, week_end)
    if dry_run:
        print(f"\n[DRY-RUN] Resume hebdo:\n{summary}\n")
    else:
        telegram = get_telegram()
        if telegram:
            telegram.send_alert_sync(summary)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description="PEA Scanner -- feedback loop")
    parser.add_argument("--stats", action="store_true", help="Stats globales")
    parser.add_argument("--retrain", action="store_true", help="Forcer re-entrainement")
    parser.add_argument("--weekly", action="store_true", help="Resume hebdomadaire")
    parser.add_argument("--dry-run", action="store_true", help="Sans Telegram")
    args = parser.parse_args()
    if args.stats:
        run_stats()
    elif args.retrain:
        run_retrain(dry_run=args.dry_run)
    elif args.weekly:
        run_weekly(dry_run=args.dry_run)
    else:
        run_review(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
