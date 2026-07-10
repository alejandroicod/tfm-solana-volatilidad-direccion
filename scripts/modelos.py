"""Modelos, métricas y utilidades de entrenamiento."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from configuracion import (
    PARAMETROS_XGB_DIRECCION,
    PARAMETROS_XGB_VOLATILIDAD,
    SEMILLA,
    UMBRAL_PROB_MAX,
    UMBRAL_PROB_MIN,
    UMBRAL_PROB_PASO,
)


def crear_modelo_xgboost(parametros: dict[str, Any]) -> XGBClassifier:
    """Crea un XGBoost binario con parámetros comunes del proyecto."""
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEMILLA,
        n_jobs=-1,
        verbosity=0,
        **parametros,
    )


def modelos_etapa_1() -> list[dict[str, Any]]:
    """Devuelve los modelos comparados en el filtro de volatilidad."""
    return [
        {
            "nombre": "LogisticRegression",
            "estimador": LogisticRegression(C=0.03, class_weight="balanced", max_iter=3000, random_state=SEMILLA),
            "escalar": True,
            "parametros": {"C": 0.03, "class_weight": "balanced"},
        },
        {
            "nombre": "SVM",
            "estimador": SVC(C=1.0, gamma="scale", kernel="rbf", class_weight="balanced", probability=False, random_state=SEMILLA),
            "escalar": True,
            "parametros": {"C": 1.0, "gamma": "scale", "kernel": "rbf", "class_weight": "balanced"},
        },
        {
            "nombre": "RandomForest",
            "estimador": RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEMILLA,
            ),
            "escalar": False,
            "parametros": {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 20, "class_weight": "balanced"},
        },
        {
            "nombre": "XGBoost",
            "estimador": crear_modelo_xgboost(PARAMETROS_XGB_VOLATILIDAD),
            "escalar": False,
            "parametros": PARAMETROS_XGB_VOLATILIDAD,
        },
    ]


def modelos_etapa_2() -> list[dict[str, Any]]:
    """Devuelve los modelos principales de la segunda etapa."""
    return [
        {
            "nombre": "RandomForest",
            "estimador": RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                min_samples_leaf=20,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=SEMILLA,
            ),
            "escalar": False,
            "parametros": {"n_estimators": 150, "max_depth": 5, "min_samples_leaf": 20, "max_features": "sqrt"},
        },
        {
            "nombre": "XGBoost",
            "estimador": crear_modelo_xgboost(PARAMETROS_XGB_DIRECCION),
            "escalar": False,
            "parametros": PARAMETROS_XGB_DIRECCION,
        },
    ]


def calcular_scores(modelo: Any, x: np.ndarray) -> np.ndarray:
    """Obtiene probabilidad o score continuo de la clase positiva."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(x)[:, 1]
    if hasattr(modelo, "decision_function"):
        bruto = modelo.decision_function(x)
        return 1 / (1 + np.exp(-bruto))
    return modelo.predict(x)


def entrenar_y_predecir(config: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Entrena un modelo y devuelve scores para validación y test.

    Si el modelo requiere normalización, el escalador se ajusta sólo con train
    para evitar fuga de información.
    """
    if config["escalar"]:
        escalador = StandardScaler()
        x_train = escalador.fit_transform(x_train)
        x_val = escalador.transform(x_val)
        x_test = escalador.transform(x_test)
    modelo = clone(config["estimador"])
    modelo.fit(x_train, y_train)
    return calcular_scores(modelo, x_val), calcular_scores(modelo, x_test)


def mejor_umbral(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Selecciona el umbral que maximiza macro F1 en validación."""
    umbrales = np.round(np.arange(UMBRAL_PROB_MIN, UMBRAL_PROB_MAX + 1e-9, UMBRAL_PROB_PASO), 3)
    valores = [(u, f1_score(y_true, (scores >= u).astype(int), average="macro", zero_division=0)) for u in umbrales]
    return max(valores, key=lambda x: x[1])[0]


def calcular_metricas(y_true: np.ndarray, scores: np.ndarray, umbral: float) -> dict[str, float]:
    """Calcula accuracy, macro F1, precision y recall."""
    pred = (scores >= umbral).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, pred, average="macro", zero_division=0),
        "f1_clase_1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "f1_clase_0": f1_score(y_true, pred, pos_label=0, zero_division=0),
        "tasa_prediccion_clase_1": float(pred.mean()),
    }


def parametros_a_json(parametros: dict[str, Any]) -> str:
    """Convierte parámetros a JSON ordenado para guardar resultados."""
    return json.dumps(parametros, sort_keys=True)
