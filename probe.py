"""
probe.py — Hallucination probe classifier (CatBoost implementation).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

class HallucinationProbe:
    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._threshold: float = 0.5
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """
        Обучает CatBoost классификатор.
        """
        X_scaled = self._scaler.fit_transform(X)
    
        self._model = CatBoostClassifier(
            iterations=300,
            depth=3,
            learning_rate=0.05,
            l2_leaf_reg=30,
            subsample=0.8,
            colsample_bylevel=0.1,
            loss_function='Logloss',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=False
        )

        self._model.fit(X_scaled, y.astype(int))
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        probs = self.predict_proba(X_val)[:, 1]

        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict_proba().")
        X_scaled = self._scaler.transform(X)
        probs = self._model.predict_proba(X_scaled)
        return probs
