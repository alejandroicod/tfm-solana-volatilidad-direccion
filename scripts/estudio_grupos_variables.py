"""Estudio de grupos de variables para etapa 1 y etapa 2.

Este script reproduce la idea final de ablación: buscar el grupo mínimo que
mantiene buen rendimiento en volatilidad y el grupo que maximiza la etapa
direccional condicionada.
"""

from __future__ import annotations

import pandas as pd

from configuracion import CARPETA_METRICAS
from datos import cargar_y_alinear
from entrenar_etapa1_volatilidad import ejecutar_etapa1
from entrenar_etapa2_direccion import ejecutar_etapa2
from variables import crear_variables, grupos_de_variables


def guardar_grupos_variables() -> None:
    """Guarda un CSV con los grupos de variables usados en el estudio."""
    datos = cargar_y_alinear()
    variables = crear_variables(datos)
    grupos = grupos_de_variables(variables.columns.tolist())
    filas = [{"grupo": nombre, "n_variables": len(cols), "variables": "|".join(cols)} for nombre, cols in grupos.items()]
    CARPETA_METRICAS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(CARPETA_METRICAS / "grupos_variables.csv", index=False)


def ejecutar_estudio_grupos() -> None:
    """Guarda los grupos y ejecuta las dos etapas principales."""
    guardar_grupos_variables()
    ejecutar_etapa1()
    ejecutar_etapa2()


if __name__ == "__main__":
    ejecutar_estudio_grupos()
