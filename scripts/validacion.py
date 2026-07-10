"""Funciones de validación walk-forward temporal."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from configuracion import HORAS_ENTRENAMIENTO, HORAS_GAP, HORAS_PASO, HORAS_TEST, HORAS_VALIDACION


@dataclass
class FoldTemporal:
    """Contenedor de índices temporales para un fold walk-forward."""

    id_fold: int
    indices_train: pd.DatetimeIndex
    indices_validacion: pd.DatetimeIndex
    indices_test: pd.DatetimeIndex


def crear_folds_walk_forward(indice: pd.DatetimeIndex) -> list[FoldTemporal]:
    """Crea folds walk-forward con train, validación y test cronológicos."""
    inicio = indice.min()
    fin = indice.max()
    cursor = inicio
    folds: list[FoldTemporal] = []
    id_fold = 0

    while True:
        train_inicio = cursor
        train_fin = train_inicio + pd.Timedelta(hours=HORAS_ENTRENAMIENTO)
        validacion_inicio = train_fin + pd.Timedelta(hours=HORAS_GAP)
        validacion_fin = validacion_inicio + pd.Timedelta(hours=HORAS_VALIDACION)
        test_inicio = validacion_fin + pd.Timedelta(hours=HORAS_GAP)
        test_fin = test_inicio + pd.Timedelta(hours=HORAS_TEST)

        if test_fin > fin:
            break

        idx_train = indice[(indice >= train_inicio) & (indice < train_fin)]
        idx_val = indice[(indice >= validacion_inicio) & (indice < validacion_fin)]
        idx_test = indice[(indice >= test_inicio) & (indice < test_fin)]

        if len(idx_train) and len(idx_val) and len(idx_test):
            assert idx_train.max() < idx_val.min()
            assert idx_val.max() < idx_test.min()
            assert len(set(idx_train).intersection(idx_val)) == 0
            assert len(set(idx_val).intersection(idx_test)) == 0
            folds.append(FoldTemporal(id_fold, idx_train, idx_val, idx_test))
            id_fold += 1

        cursor += pd.Timedelta(hours=HORAS_PASO)

    return folds
