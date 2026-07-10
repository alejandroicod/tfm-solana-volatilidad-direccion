"""Descarga velas OHLCV de Binance para los símbolos del estudio.

Uso:
    python scripts/descargar_datos_binance.py

El script descarga velas de 1 hora para SOLUSDT, BTCUSDT y ETHUSDT entre
2022 y 2025. Se usa el endpoint público de Binance, por lo que no requiere
clave API. Si ya existen los archivos, se sobreescriben.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

from configuracion import ANIOS, CARPETA_DATOS, INTERVALO, SIMBOLOS


URL_KLINES = "https://api.binance.com/api/v3/klines"
COLUMNAS_BINANCE = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


def milisegundos(fecha: str) -> int:
    """Convierte una fecha legible a milisegundos Unix en UTC."""
    return int(pd.Timestamp(fecha, tz="UTC").timestamp() * 1000)


def descargar_velas(simbolo: str, intervalo: str, inicio_ms: int, fin_ms: int) -> pd.DataFrame:
    """Descarga velas de Binance paginando hasta alcanzar `fin_ms`."""
    filas = []
    cursor = inicio_ms
    while cursor < fin_ms:
        parametros = {
            "symbol": simbolo,
            "interval": intervalo,
            "startTime": cursor,
            "endTime": fin_ms,
            "limit": 1000,
        }
        respuesta = requests.get(URL_KLINES, params=parametros, timeout=30)
        respuesta.raise_for_status()
        bloque = respuesta.json()
        if not bloque:
            break
        filas.extend(bloque)
        cursor = int(bloque[-1][0]) + 1
        time.sleep(0.05)
    return pd.DataFrame(filas, columns=COLUMNAS_BINANCE)


def guardar_anio(simbolo: str, anio: int) -> Path:
    """Descarga y guarda un año completo de velas para un símbolo."""
    inicio = milisegundos(f"{anio}-01-01")
    fin = milisegundos(f"{anio + 1}-01-01")
    datos = descargar_velas(simbolo, INTERVALO, inicio, fin)
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    salida = CARPETA_DATOS / f"{simbolo}_{INTERVALO}_{anio}.csv"
    datos.to_csv(salida, index=False)
    return salida


def descargar_todo() -> None:
    """Descarga todos los símbolos y años configurados."""
    for simbolo in SIMBOLOS:
        for anio in ANIOS:
            salida = guardar_anio(simbolo, anio)
            print(f"Guardado: {salida}")


if __name__ == "__main__":
    descargar_todo()
