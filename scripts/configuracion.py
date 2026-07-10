"""Configuración central del pipeline final del TFM.

Este archivo concentra los parámetros usados en la investigación final:
activo objetivo, activos exógenos, ventanas walk-forward, definición de
etiquetas y mejores hiperparámetros encontrados.
"""

from __future__ import annotations

from pathlib import Path


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ_PROYECTO / "datos" / "raw"
CARPETA_RESULTADOS = RAIZ_PROYECTO / "resultados"
CARPETA_METRICAS = CARPETA_RESULTADOS / "metricas"
CARPETA_GRAFICOS = CARPETA_RESULTADOS / "graficos"

ACTIVO_OBJETIVO = "SOL"
ACTIVOS_EXOGENOS = ["BTC", "ETH"]
SIMBOLOS = ["SOLUSDT", "BTCUSDT", "ETHUSDT"]
INTERVALO = "1h"
ANIOS = [2022, 2023, 2024, 2025]

SEMILLA = 42

# Validación walk-forward usada en el estudio final.
HORAS_ENTRENAMIENTO = 18 * 30 * 24
HORAS_VALIDACION = 60 * 24
HORAS_TEST = 60 * 24
HORAS_PASO = 90 * 24
HORAS_GAP = 60

# Etapa 1: filtro de volatilidad.
HORIZONTE_VOLATILIDAD = 12
CUANTILES_VOLATILIDAD = [0.50, 0.67]
CUANTIL_VOLATILIDAD_FINAL = 0.67

# Etapa 2: dirección condicionada.
HORIZONTE_DIRECCION = 12
UMBRAL_DIRECCION = 0.005

UMBRAL_PROB_MIN = 0.20
UMBRAL_PROB_MAX = 0.80
UMBRAL_PROB_PASO = 0.025

# Mejores hiperparámetros seleccionados para la arquitectura final.
PARAMETROS_XGB_VOLATILIDAD = {
    "n_estimators": 500,
    "max_depth": 2,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 30,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

PARAMETROS_XGB_DIRECCION = {
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.02,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 30,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 3.0,
}
