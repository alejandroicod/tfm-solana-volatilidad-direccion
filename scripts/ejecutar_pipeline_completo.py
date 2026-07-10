"""Ejecuta el pipeline final completo.

Orden:
1. Carga datos descargados.
2. Entrena etapa 1 de volatilidad.
3. Entrena etapa 2 de dirección condicionada.
4. Guarda grupos de variables.
5. Genera gráficas finales.

Nota: si no existen CSV en `datos/raw/`, ejecutar antes:
    python scripts/descargar_datos_binance.py
"""

from __future__ import annotations

from estudio_grupos_variables import guardar_grupos_variables
from entrenar_etapa1_volatilidad import ejecutar_etapa1
from entrenar_etapa2_direccion import ejecutar_etapa2
from generar_graficos import generar_todos_los_graficos


def ejecutar_pipeline_completo() -> None:
    """Ejecuta toda la investigación final de forma reproducible."""
    guardar_grupos_variables()
    ejecutar_etapa1()
    ejecutar_etapa2()
    generar_todos_los_graficos()


if __name__ == "__main__":
    ejecutar_pipeline_completo()
