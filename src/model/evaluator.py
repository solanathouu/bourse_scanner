"""Analyse des performances et interpretabilite du modele.

Repond aux questions cles:
- Est-ce que le modele bat le baseline naif?
- Quelles features comptent le plus?
- Sur quoi le modele se trompe?
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from loguru import logger


class Evaluator:
    """Analyse les performances et l'interpretabilite du modele."""

    def feature_importance(self, model, feature_names: list[str]) -> pd.DataFrame:
        """Retourne les features triees par importance.

        Si range_position et catalyst_type sont dans le top 5, le modele
        a bien appris le style Nicolas (range trading + catalyseurs).
        """
        importances = model.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def error_analysis(self, predictions: list[dict]) -> dict:
        """Analyse les trades mal predits.

        Returns:
            {
                "false_positives": list — Predits gagnants mais perdants (DANGER)
                "false_negatives": list — Predits perdants mais gagnants (opportunite ratee)
                "total_errors": int
            }
        """
        false_positives = [
            p for p in predictions
            if p["predicted"] == 1 and p["actual"] == 0
        ]
        false_negatives = [
            p for p in predictions
            if p["predicted"] == 0 and p["actual"] == 1
        ]

        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_errors": len(false_positives) + len(false_negatives),
        }

    def compare_to_baseline(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compare le modele au baseline naif (toujours predire gagnant).

        Le baseline reflette le win rate de Nicolas (88%).
        Le modele doit faire mieux que simplement dire "achete tout".
        """
        baseline_preds = np.ones(len(y_true))
        baseline_acc = accuracy_score(y_true, baseline_preds)
        model_acc = accuracy_score(y_true, y_pred)

        improvement = model_acc - baseline_acc

        return {
            "model_accuracy": round(model_acc, 4),
            "baseline_accuracy": round(baseline_acc, 4),
            "improvement": round(improvement, 4),
            "beats_baseline": improvement > 0,
        }

    def print_report(self, walk_forward_results: dict,
                     importance_df: pd.DataFrame | None = None):
        """Affiche un rapport complet en console."""
        r = walk_forward_results

        print("=" * 60)
        print("RAPPORT D'EVALUATION — Modele Nicolas v1")
        print("=" * 60)

        print(f"\nDonnees: {r.get('train_size', '?')} train, {r.get('test_size', '?')} test")
        print(f"\nMetriques sur le test set:")
        print(f"  Accuracy:   {r.get('accuracy', 0):.1%}")
        print(f"  Precision:  {r.get('precision', 0):.1%}")
        print(f"  Recall:     {r.get('recall', 0):.1%}")
        print(f"  F1-Score:   {r.get('f1', 0):.1%}")
        print(f"  Baseline:   {r.get('baseline_accuracy', 0):.1%} (toujours predire gagnant)")

        if r.get("accuracy", 0) > r.get("baseline_accuracy", 0):
            print(f"\n  >> Le modele BAT le baseline de "
                  f"+{(r['accuracy'] - r['baseline_accuracy']):.1%}")
        else:
            print(f"\n  >> Le modele ne bat PAS le baseline")

        # Matrice de confusion
        cm = r.get("confusion_matrix")
        if cm:
            print(f"\nMatrice de confusion:")
            print(f"  Predit Perdant | Predit Gagnant")
            print(f"  Vrai Perdant:  {cm[0][0]:>6} | {cm[0][1]:>6}")
            print(f"  Vrai Gagnant:  {cm[1][0]:>6} | {cm[1][1]:>6}")

        # Feature importance
        if importance_df is not None:
            print(f"\nTop 10 features les plus importantes:")
            for i, row in importance_df.head(10).iterrows():
                bar = "#" * int(row["importance"] * 50)
                print(f"  {i+1:2}. {row['feature']:25s} {row['importance']:.3f} {bar}")

        # Analyse erreurs
        preds = r.get("predictions", [])
        if preds:
            errors = self.error_analysis(preds)
            print(f"\nAnalyse des erreurs ({errors['total_errors']} erreurs):")
            if errors["false_positives"]:
                print(f"  Faux positifs (DANGEREUX — on perd de l'argent):")
                for fp in errors["false_positives"]:
                    print(f"    Trade #{fp['trade_id']}: predit gagnant (p={fp['proba']:.2f}) "
                          f"mais perdant")
            if errors["false_negatives"]:
                print(f"  Faux negatifs (on rate des opportunites):")
                for fn in errors["false_negatives"]:
                    print(f"    Trade #{fn['trade_id']}: predit perdant (p={fn['proba']:.2f}) "
                          f"mais gagnant")

        print("\n" + "=" * 60)
