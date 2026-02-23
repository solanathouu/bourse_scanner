"""Entrainement du modele XGBoost sur les trades de Nicolas.

Apprend a predire si Nicolas gagnerait un trade en se basant sur
la combinaison indicateurs techniques + type de catalyseur + contexte.
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from loguru import logger


# Mapping catalyst_type -> code numerique
CATALYST_TYPE_ENCODING = {
    "TECHNICAL": 0,
    "UNKNOWN": 1,
    "OTHER_NEGATIVE": 2,
    "OTHER_POSITIVE": 3,
    "SECTOR_MACRO": 4,
    "INSIDER": 5,
    "DIVIDEND": 6,
    "RESTRUCTURING": 7,
    "CONTRACT": 8,
    "DOWNGRADE": 9,
    "UPGRADE": 10,
    "EARNINGS": 11,
    "FDA_REGULATORY": 12,
}


class Trainer:
    """Entraine un XGBoost sur les trades historiques de Nicolas."""

    def __init__(self):
        self.model = None
        self.feature_names: list[str] = []

    def prepare_data(self, features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare X (features) et y (target) a partir du DataFrame de features.

        Encode catalyst_type en numerique. Retire trade_id et is_winner de X.
        """
        df = features_df.copy()

        # Encoder catalyst_type
        df["catalyst_type"] = df["catalyst_type"].map(CATALYST_TYPE_ENCODING).fillna(1)

        # Separer X et y
        y = df["is_winner"]
        drop_cols = ["is_winner", "trade_id"]
        # Aussi retirer date_achat si present (pas une feature)
        if "date_achat" in df.columns:
            drop_cols.append("date_achat")

        X = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Remplacer les NaN restants par 0
        X = X.fillna(0)

        self.feature_names = list(X.columns)
        return X, y

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Entraine le modele XGBoost.

        Parametres optimises pour un petit dataset (~141 samples):
        - max_depth=4 (eviter overfitting)
        - n_estimators=100
        - learning_rate=0.1
        - scale_pos_weight ajuste au ratio negatifs/positifs
        """
        # Calculer le poids pour gerer le desequilibre
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        self.model = xgb.XGBClassifier(
            max_depth=4,
            n_estimators=100,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X, y, verbose=False)

        # Metriques d'entrainement
        train_preds = self.model.predict(X)
        train_acc = accuracy_score(y, train_preds)

        logger.info(f"Modele entraine: accuracy={train_acc:.3f}, "
                     f"samples={len(y)}, pos={n_pos}, neg={n_neg}")

        return {"train_accuracy": train_acc}

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predit les classes (0/1) pour les features donnees."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Retourne les probabilites de la classe positive (gagnant)."""
        return self.model.predict_proba(X)[:, 1]

    def walk_forward_validate(self, features_df: pd.DataFrame,
                               split_date: str = "2025-12-01") -> dict:
        """Validation walk-forward: train avant split_date, test apres.

        Args:
            features_df: DataFrame complet avec features + target + date_achat.
            split_date: Date de separation train/test (YYYY-MM-DD).

        Returns:
            Dict avec metriques et predictions detaillees.
        """
        df = features_df.copy()

        train_mask = df["date_achat"] < split_date
        test_mask = df["date_achat"] >= split_date

        train_df = df[train_mask]
        test_df = df[test_mask]

        if len(test_df) == 0:
            logger.warning("Pas de donnees de test apres le split")
            return {"error": "Pas de donnees de test"}

        # Preparer et entrainer sur le train set
        X_train, y_train = self.prepare_data(train_df)
        self.train(X_train, y_train)

        # Predire sur le test set
        X_test, y_test = self.prepare_data(test_df)
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        # Baseline naif: toujours predire gagnant
        baseline_preds = np.ones(len(y_test))
        baseline_acc = accuracy_score(y_test, baseline_preds)

        # Metriques
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "baseline_accuracy": baseline_acc,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "predictions": [],
        }

        # Predictions detaillees
        for i, (idx, row) in enumerate(test_df.iterrows()):
            results["predictions"].append({
                "trade_id": int(row.get("trade_id", i)),
                "actual": int(y_test.iloc[i]),
                "predicted": int(y_pred[i]),
                "proba": round(float(y_proba[i]), 3),
            })

        logger.info(f"Walk-forward: accuracy={results['accuracy']:.3f} "
                     f"(baseline={baseline_acc:.3f}), "
                     f"train={results['train_size']}, test={results['test_size']}")

        return results

    def save_model(self, path: str = "data/models/nicolas_v1.joblib"):
        """Sauvegarde le modele entraine."""
        if self.model is None:
            raise ValueError("Le modele n'est pas entraine")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, path)
        logger.info(f"Modele sauvegarde: {path}")

    def load_model(self, path: str = "data/models/nicolas_v1.joblib"):
        """Charge un modele sauvegarde."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        logger.info(f"Modele charge: {path}")
