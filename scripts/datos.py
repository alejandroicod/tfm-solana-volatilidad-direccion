"""Carga, limpieza y alineación temporal de las velas OHLCV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from configuracion import ACTIVOS_EXOGENOS, ACTIVO_OBJETIVO, ANIOS, CARPETA_DATOS, INTERVALO


def cargar_activo(activo: str, carpeta: Path = CARPETA_DATOS) -> pd.DataFrame:
    """Carga todos los CSV de un activo y devuelve un DataFrame OHLCV limpio.

    El activo debe indicarse como `SOL`, `BTC` o `ETH`. Las columnas se
    renombran con prefijo en minúscula para evitar conflictos al unir series.
    """
    simbolo = f"{activo}USDT"
    archivos = []
    for anio in ANIOS:
        ruta = carpeta / f"{simbolo}_{INTERVALO}_{anio}.csv"
        if ruta.exists():
            archivos.append(ruta)
    if not archivos:
        raise FileNotFoundError(f"No se encontraron CSV para {activo} en {carpeta}")

    bloques = []
    for ruta in archivos:
        bruto = pd.read_csv(ruta)
        bruto.columns = [c.lower() for c in bruto.columns]
        if "open_time" in bruto.columns:
            columna_tiempo = "open_time"
        elif "timestamp" in bruto.columns:
            columna_tiempo = "timestamp"
        else:
            columna_tiempo = bruto.columns[0]
        df = bruto[[columna_tiempo, "open", "high", "low", "close", "volume"]].copy()
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df[columna_tiempo], errors="coerce"), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        bloques.append(df[["timestamp", "open", "high", "low", "close", "volume"]])

    datos = (
        pd.concat(bloques, ignore_index=True)
        .dropna()
        .drop_duplicates("timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    assert datos.index.is_monotonic_increasing
    return datos.add_prefix(f"{activo.lower()}_")


def cargar_y_alinear() -> pd.DataFrame:
    """Carga SOL, BTC y ETH, los alinea por timestamp y elimina nulos."""
    activos = [ACTIVO_OBJETIVO, *ACTIVOS_EXOGENOS]
    series = [cargar_activo(activo) for activo in activos]
    datos = pd.concat(series, axis=1, join="inner").dropna().sort_index()
    assert datos.index.is_monotonic_increasing
    assert not datos.index.duplicated().any()
    return datos.replace([np.inf, -np.inf], np.nan).dropna()


def guardar_datos_alineados(salida: Path) -> None:
    """Guarda el dataset alineado en CSV para inspección o reutilización."""
    datos = cargar_y_alinear()
    salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(salida)
