"""Script d'exploration des features pour comprendre les patterns de Nicolas.

Usage:
    uv run python scripts/analyze_features.py             # Stats descriptives
    uv run python scripts/analyze_features.py --trade 42   # Features d'un trade specifique
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine

DB_PATH = "data/trades.db"


def print_descriptive_stats(features_df):
    """Affiche les stats descriptives des features."""
    winners = features_df[features_df["is_winner"] == 1]
    losers = features_df[features_df["is_winner"] == 0]

    print("=" * 70)
    print("ANALYSE DES FEATURES — Profil de trading Nicolas")
    print("=" * 70)

    print(f"\nTrades: {len(features_df)} ({len(winners)} gagnants, {len(losers)} perdants)")

    # Comparer gagnants vs perdants sur les features cles
    key_features = [
        "range_position_10", "range_position_20",
        "rsi_14", "volume_ratio_20",
        "nb_catalysts", "best_catalyst_score",
    ]

    print(f"\n{'Feature':30s} {'Gagnants (moy)':>15s} {'Perdants (moy)':>15s} {'Diff':>10s}")
    print("-" * 70)

    for feat in key_features:
        if feat in features_df.columns:
            w_mean = winners[feat].mean() if len(winners) > 0 else 0
            l_mean = losers[feat].mean() if len(losers) > 0 else 0
            diff = w_mean - l_mean
            print(f"{feat:30s} {w_mean:>15.3f} {l_mean:>15.3f} {diff:>+10.3f}")

    # Distribution des types de catalyseurs
    print(f"\nTypes de catalyseurs (gagnants):")
    if "catalyst_type" in winners.columns:
        for cat, count in winners["catalyst_type"].value_counts().items():
            print(f"  {cat:25s}: {count:3d} ({100*count/len(winners):.1f}%)")

    print(f"\nTypes de catalyseurs (perdants):")
    if "catalyst_type" in losers.columns:
        for cat, count in losers["catalyst_type"].value_counts().items():
            print(f"  {cat:25s}: {count:3d} ({100*count/len(losers):.1f}%)")


def print_trade_features(features_df, trade_id: int):
    """Affiche les features d'un trade specifique."""
    row = features_df[features_df["trade_id"] == trade_id]
    if len(row) == 0:
        print(f"Trade #{trade_id} non trouve dans les features")
        return

    row = row.iloc[0]
    result = "GAGNANT" if row["is_winner"] == 1 else "PERDANT"

    print(f"\n=== Trade #{trade_id} — {result} ===")
    print(f"\nFeatures techniques (au moment de l'achat):")
    for feat in ["range_position_10", "range_position_20", "rsi_14",
                 "macd_histogram", "bollinger_position", "volume_ratio_20",
                 "atr_14_pct", "variation_1j", "variation_5j",
                 "distance_sma20", "distance_sma50"]:
        if feat in row.index:
            print(f"  {feat:25s}: {row[feat]:.4f}")

    print(f"\nFeatures catalyseur:")
    for feat in ["catalyst_type", "nb_catalysts", "best_catalyst_score",
                 "has_text_match", "sentiment_avg", "nb_news_sources"]:
        if feat in row.index:
            print(f"  {feat:25s}: {row[feat]}")

    print(f"\nFeatures contexte:")
    for feat in ["day_of_week", "nb_previous_trades",
                 "previous_win_rate", "days_since_last_trade"]:
        if feat in row.index:
            print(f"  {feat:25s}: {row[feat]}")


def main():
    parser = argparse.ArgumentParser(description="Exploration des features")
    parser.add_argument("--trade", type=int, help="ID du trade a inspecter")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()
    engine = FeatureEngine(db)

    print("Construction des features...")
    features_df = engine.build_all_features()

    if args.trade:
        print_trade_features(features_df, args.trade)
    else:
        print_descriptive_stats(features_df)


if __name__ == "__main__":
    main()
