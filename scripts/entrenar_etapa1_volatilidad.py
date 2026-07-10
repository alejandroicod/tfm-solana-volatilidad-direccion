"""Entrena y evalúa la etapa 1: filtro de volatilidad de SOL."""

from __future__ import annotations

import pandas as pd

from configuracion import CARPETA_METRICAS, CUANTILES_VOLATILIDAD
from datos import cargar_y_alinear
from modelos import calcular_metricas, entrenar_y_predecir, mejor_umbral, modelos_etapa_1, parametros_a_json
from validacion import crear_folds_walk_forward
from variables import crear_etiqueta_volatilidad, crear_variables


def ejecutar_etapa1() -> None:
    """Ejecuta la comparativa final de modelos para volatilidad."""
    datos = cargar_y_alinear()
    variables = crear_variables(datos)
    filas = []

    for cuantil in CUANTILES_VOLATILIDAD:
        etiqueta = crear_etiqueta_volatilidad(datos, cuantil).reindex(variables.index).dropna().astype(int)
        x = variables.reindex(etiqueta.index)
        folds = crear_folds_walk_forward(x.index)

        for fold in folds:
            x_train = x.loc[fold.indices_train].values
            x_val = x.loc[fold.indices_validacion].values
            x_test = x.loc[fold.indices_test].values
            y_train = etiqueta.loc[fold.indices_train].values
            y_val = etiqueta.loc[fold.indices_validacion].values
            y_test = etiqueta.loc[fold.indices_test].values

            if min(len(set(y_train)), len(set(y_val)), len(set(y_test))) < 2:
                continue

            for config in modelos_etapa_1():
                score_val, score_test = entrenar_y_predecir(config, x_train, y_train, x_val, x_test)
                umbral = mejor_umbral(y_val, score_val)
                metricas_val = calcular_metricas(y_val, score_val, umbral)
                metricas_test = calcular_metricas(y_test, score_test, umbral)
                filas.append(
                    {
                        "cuantil": cuantil,
                        "fold": fold.id_fold,
                        "modelo": config["nombre"],
                        "parametros": parametros_a_json(config["parametros"]),
                        "umbral": umbral,
                        "validacion_macro_f1": metricas_val["macro_f1"],
                        **{f"test_{k}": v for k, v in metricas_test.items()},
                    }
                )

    CARPETA_METRICAS.mkdir(parents=True, exist_ok=True)
    resultados = pd.DataFrame(filas)
    resultados.to_csv(CARPETA_METRICAS / "etapa1_folds.csv", index=False)
    agregados = (
        resultados.groupby(["cuantil", "modelo", "parametros"])
        .agg(
            folds=("fold", "nunique"),
            validacion_macro_f1=("validacion_macro_f1", "mean"),
            accuracy=("test_accuracy", "mean"),
            macro_f1=("test_macro_f1", "mean"),
            precision=("test_precision", "mean"),
            recall=("test_recall", "mean"),
            f1_alta_volatilidad=("test_f1_clase_1", "mean"),
            f1_baja_volatilidad=("test_f1_clase_0", "mean"),
        )
        .reset_index()
        .sort_values(["cuantil", "validacion_macro_f1"], ascending=[True, False])
    )
    agregados.to_csv(CARPETA_METRICAS / "etapa1_agregado.csv", index=False)
    print(agregados.to_string(index=False))


if __name__ == "__main__":
    ejecutar_etapa1()
