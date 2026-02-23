"""Script d'entrainement du modele Nicolas.

Usage:
    uv run python scripts/train_model.py                # Entrainer + evaluer
    uv run python scripts/train_model.py --features      # Voir les features seulement
    uv run python scripts/train_model.py --importance     # Feature importance
"""

import argparse

from src.core.database import Database
from src.analysis.feature_engine import FeatureEngine
from src.model.trainer import Trainer
from src.model.evaluator import Evaluator

DB_PATH = "data/trades.db"
MODEL_PATH = "data/models/nicolas_v1.joblib"


def main():
    parser = argparse.ArgumentParser(description="Entrainement du modele Nicolas")
    parser.add_argument("--features", action="store_true",
                        help="Afficher les features seulement (pas d'entrainement)")
    parser.add_argument("--importance", action="store_true",
                        help="Afficher le feature importance d'un modele existant")
    parser.add_argument("--split-date", default="2025-12-01",
                        help="Date de split walk-forward (defaut: 2025-12-01)")
    args = parser.parse_args()

    db = Database(DB_PATH)
    db.init_db()
    engine = FeatureEngine(db)

    # Construire les features
    print("Construction des features...")
    features_df = engine.build_all_features()
    print(f"Matrice: {len(features_df)} trades x {len(features_df.columns)} colonnes")
    print(f"Gagnants: {(features_df['is_winner'] == 1).sum()}")
    print(f"Perdants: {(features_df['is_winner'] == 0).sum()}")

    if args.features:
        print(f"\nFeatures disponibles:")
        for name in engine.get_feature_names():
            print(f"  - {name}")
        print(f"\nApercu des 5 premiers trades:")
        print(features_df.head().to_string())
        return

    # Entrainer et evaluer
    trainer = Trainer()
    evaluator = Evaluator()

    # Ajouter date_achat pour le split walk-forward
    trades = db.get_all_trades()
    closed = [t for t in trades if t["statut"] == "CLOTURE"]
    # Mapper trade_id -> date_achat
    date_map = {t["id"]: t["date_achat"][:10] for t in closed}
    features_df["date_achat"] = features_df["trade_id"].map(date_map)

    print(f"\nWalk-forward validation (split: {args.split_date})...")
    results = trainer.walk_forward_validate(features_df, split_date=args.split_date)

    # Feature importance
    importance_df = None
    if trainer.model is not None:
        importance_df = evaluator.feature_importance(
            trainer.model, trainer.feature_names
        )

    # Rapport complet
    evaluator.print_report(results, importance_df)

    # Juste afficher l'importance
    if args.importance:
        return

    # Re-entrainer sur TOUTES les donnees pour le modele final
    print(f"\nEntrainement final sur toutes les donnees...")
    X, y = trainer.prepare_data(features_df)
    trainer.train(X, y)
    trainer.save_model(MODEL_PATH)
    print(f"Modele sauvegarde: {MODEL_PATH}")


if __name__ == "__main__":
    main()
