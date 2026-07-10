"""Entrena la etapa 2: dirección de SOL condicionada a alta volatilidad."""

from __future__ import annotations

import numpy as np
import pandas as pd

from configuracion import CARPETA_METRICAS, CUANTIL_VOLATILIDAD_FINAL, PARAMETROS_XGB_VOLATILIDAD
from datos import cargar_y_alinear
from modelos import calcular_metricas, crear_modelo_xgboost, entrenar_y_predecir, mejor_umbral, modelos_etapa_2, parametros_a_json
from validacion import crear_folds_walk_forward
from variables import crear_etiqueta_direccion, crear_etiqueta_volatilidad, crear_variables


def ejecutar_etapa2() -> None:
    """Compara el predictor direccional con y sin filtro de volatilidad."""
    datos = cargar_y_alinear()
    variables = crear_variables(datos)
    etiqueta_direccion = crear_etiqueta_direccion(datos).reindex(variables.index).dropna().astype(int)
    etiqueta_volatilidad = crear_etiqueta_volatilidad(datos, CUANTIL_VOLATILIDAD_FINAL).reindex(etiqueta_direccion.index)
    indice_valido = etiqueta_direccion.index.intersection(etiqueta_volatilidad.dropna().index)
    variables = variables.reindex(indice_valido)
    etiqueta_direccion = etiqueta_direccion.reindex(indice_valido).astype(int)
    etiqueta_volatilidad = etiqueta_volatilidad.reindex(indice_valido).astype(int)
    folds = crear_folds_walk_forward(variables.index)
    filas = []

    config_vol = {
        "nombre": "XGBoost_volatilidad",
        "estimador": crear_modelo_xgboost(PARAMETROS_XGB_VOLATILIDAD),
        "escalar": False,
        "parametros": PARAMETROS_XGB_VOLATILIDAD,
    }

    for fold in folds:
        x_train = variables.loc[fold.indices_train].values
        x_val = variables.loc[fold.indices_validacion].values
        x_test = variables.loc[fold.indices_test].values
        y_train_dir = etiqueta_direccion.loc[fold.indices_train].values
        y_val_dir = etiqueta_direccion.loc[fold.indices_validacion].values
        y_test_dir = etiqueta_direccion.loc[fold.indices_test].values
        y_train_vol = etiqueta_volatilidad.loc[fold.indices_train].values
        y_val_vol = etiqueta_volatilidad.loc[fold.indices_validacion].values

        score_val_vol, score_test_vol = entrenar_y_predecir(config_vol, x_train, y_train_vol, x_val, x_test)
        umbral_vol = mejor_umbral(y_val_vol, score_val_vol)
        mascara_val_alta = score_val_vol >= umbral_vol
        mascara_test_alta = score_test_vol >= umbral_vol

        for config in modelos_etapa_2():
            score_val, score_test = entrenar_y_predecir(config, x_train, y_train_dir, x_val, x_test)

            # Caso sin filtro: se evalúa sobre todas las horas del test.
            umbral_sin_filtro = mejor_umbral(y_val_dir, score_val)
            metricas_sin_filtro = calcular_metricas(y_test_dir, score_test, umbral_sin_filtro)
            filas.append(
                {
                    "modo": "sin_filtro_volatilidad",
                    "fold": fold.id_fold,
                    "modelo": config["nombre"],
                    "parametros": parametros_a_json(config["parametros"]),
                    "cobertura": 1.0,
                    "umbral_direccion": umbral_sin_filtro,
                    **{f"test_{k}": v for k, v in metricas_sin_filtro.items()},
                }
            )

            # Caso con filtro: se evalúa sólo donde la etapa 1 predice alta volatilidad.
            if mascara_val_alta.sum() < 50 or mascara_test_alta.sum() < 50:
                continue
            if len(np.unique(y_val_dir[mascara_val_alta])) < 2 or len(np.unique(y_test_dir[mascara_test_alta])) < 2:
                continue
            umbral_con_filtro = mejor_umbral(y_val_dir[mascara_val_alta], score_val[mascara_val_alta])
            metricas_con_filtro = calcular_metricas(y_test_dir[mascara_test_alta], score_test[mascara_test_alta], umbral_con_filtro)
            filas.append(
                {
                    "modo": "con_filtro_volatilidad_xgb_p67",
                    "fold": fold.id_fold,
                    "modelo": config["nombre"],
                    "parametros": parametros_a_json(config["parametros"]),
                    "cobertura": float(mascara_test_alta.mean()),
                    "umbral_direccion": umbral_con_filtro,
                    **{f"test_{k}": v for k, v in metricas_con_filtro.items()},
                }
            )

    CARPETA_METRICAS.mkdir(parents=True, exist_ok=True)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(CARPETA_METRICAS / "etapa2_folds.csv", index=False)
    agregados = (
        resultados.groupby(["modo", "modelo", "parametros"])
        .agg(
            folds=("fold", "nunique"),
            cobertura=("cobertura", "mean"),
            accuracy=("test_accuracy", "mean"),
            macro_f1=("test_macro_f1", "mean"),
            precision=("test_precision", "mean"),
            recall=("test_recall", "mean"),
            f1_alcista=("test_f1_clase_1", "mean"),
            f1_bajista=("test_f1_clase_0", "mean"),
        )
        .reset_index()
        .sort_values(["modo", "macro_f1"], ascending=[True, False])
    )
    agregados.to_csv(CARPETA_METRICAS / "etapa2_agregado.csv", index=False)
    print(agregados.to_string(index=False))


if __name__ == "__main__":
    ejecutar_etapa2()
