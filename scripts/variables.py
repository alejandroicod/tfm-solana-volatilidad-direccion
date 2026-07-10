"""Ingeniería de variables y creación de etiquetas sin leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd

from configuracion import (
    ACTIVO_OBJETIVO,
    ACTIVOS_EXOGENOS,
    CUANTIL_VOLATILIDAD_FINAL,
    HORIZONTE_DIRECCION,
    HORIZONTE_VOLATILIDAD,
    UMBRAL_DIRECCION,
)


def calcular_rsi(serie: pd.Series, periodo: int) -> pd.Series:
    """Calcula RSI con medias rolling simples."""
    delta = serie.diff()
    ganancia = delta.clip(lower=0).rolling(periodo).mean()
    perdida = (-delta.clip(upper=0)).rolling(periodo).mean()
    return 100 - (100 / (1 + ganancia / (perdida + 1e-9)))


def calcular_atr(maximo: pd.Series, minimo: pd.Series, cierre: pd.Series, periodo: int) -> pd.Series:
    """Calcula ATR rolling normalizable por precio."""
    rango_verdadero = pd.concat(
        [maximo - minimo, (maximo - cierre.shift(1)).abs(), (minimo - cierre.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return rango_verdadero.rolling(periodo).mean()


def crear_variables(datos: pd.DataFrame, objetivo: str = ACTIVO_OBJETIVO) -> pd.DataFrame:
    """Crea variables de SOL, BTC y ETH usando sólo pasado y presente.

    Las variables basadas en cierres, rangos o volumen se desplazan con
    `.shift(1)` cuando podrían incorporar la vela actual de forma ambigua.
    Así el modelo decide en t usando información cerrada hasta t-1.
    """
    variables = pd.DataFrame(index=datos.index)
    retornos_log = {}
    activos = [objetivo, *ACTIVOS_EXOGENOS]

    for activo in activos:
        prefijo = activo.lower()
        cierre = datos[f"{prefijo}_close"]
        maximo = datos[f"{prefijo}_high"]
        minimo = datos[f"{prefijo}_low"]
        volumen = datos[f"{prefijo}_volume"]
        ret_log = np.log(cierre / cierre.shift(1))
        retornos_log[prefijo] = ret_log

        for ventana in [6, 12, 24, 48, 72, 168, 336]:
            variables[f"{prefijo}_vol_{ventana}h"] = ret_log.rolling(ventana).std().shift(1)
            variables[f"{prefijo}_mean_ret_{ventana}h"] = ret_log.rolling(ventana).mean().shift(1)

        for ventana in [1, 2, 3, 6, 12, 24, 48]:
            variables[f"{prefijo}_ret_{ventana}h"] = cierre.pct_change(ventana).shift(1)

        for periodo in [6, 14, 24]:
            variables[f"{prefijo}_rsi_{periodo}"] = calcular_rsi(cierre, periodo).shift(1)

        variables[f"{prefijo}_atr_14_rel"] = (calcular_atr(maximo, minimo, cierre, 14) / cierre).shift(1)
        variables[f"{prefijo}_range"] = ((maximo - minimo) / cierre).shift(1)
        variables[f"{prefijo}_range_12h"] = ((maximo - minimo) / cierre).rolling(12).mean().shift(1)
        variables[f"{prefijo}_volume_rel_24h"] = (volumen / volumen.rolling(24).mean()).shift(1)
        variables[f"{prefijo}_volume_rel_168h"] = (volumen / volumen.rolling(168).mean()).shift(1)
        variables[f"{prefijo}_vol_spike"] = (
            volumen > volumen.rolling(24).mean() + 2 * volumen.rolling(24).std()
        ).astype(float).shift(1)

        for span in [21, 50, 200]:
            ema = cierre.ewm(span=span, adjust=False).mean()
            variables[f"{prefijo}_dist_ema_{span}"] = ((cierre - ema) / cierre).shift(1)

    obj = objetivo.lower()
    for exogeno in [a.lower() for a in ACTIVOS_EXOGENOS]:
        variables[f"{obj}_{exogeno}_ret_corr_24h"] = retornos_log[obj].rolling(24).corr(retornos_log[exogeno]).shift(1)
        variables[f"{obj}_{exogeno}_ret_corr_168h"] = retornos_log[obj].rolling(168).corr(retornos_log[exogeno]).shift(1)
        variables[f"{obj}_{exogeno}_vol_corr_24h"] = (
            retornos_log[obj].rolling(12).std().rolling(24).corr(retornos_log[exogeno].rolling(12).std()).shift(1)
        )
        variables[f"{obj}_{exogeno}_vol_ratio_24h"] = (
            retornos_log[obj].rolling(24).std() / retornos_log[exogeno].rolling(24).std()
        ).shift(1)
        variables[f"{obj}_{exogeno}_momentum_diff_24h"] = (
            datos[f"{obj}_close"].pct_change(24) - datos[f"{exogeno}_close"].pct_change(24)
        ).shift(1)

    variables["hour_sin"] = np.sin(2 * np.pi * datos.index.hour / 24)
    variables["hour_cos"] = np.cos(2 * np.pi * datos.index.hour / 24)
    variables["dow_sin"] = np.sin(2 * np.pi * datos.index.dayofweek / 7)
    variables["dow_cos"] = np.cos(2 * np.pi * datos.index.dayofweek / 7)
    variables["is_weekend"] = (datos.index.dayofweek >= 5).astype(int)
    return variables.replace([np.inf, -np.inf], np.nan).dropna()


def crear_etiqueta_volatilidad(datos: pd.DataFrame, cuantile: float = CUANTIL_VOLATILIDAD_FINAL) -> pd.Series:
    """Etiqueta alta volatilidad futura de SOL usando percentil histórico."""
    cierre = datos[f"{ACTIVO_OBJETIVO.lower()}_close"]
    ret_log = np.log(cierre / cierre.shift(1))
    volatilidad_futura = ret_log.shift(-1).rolling(HORIZONTE_VOLATILIDAD).std().shift(-(HORIZONTE_VOLATILIDAD - 1))
    umbral = volatilidad_futura.expanding(min_periods=720).quantile(cuantile)
    etiqueta = pd.Series(np.nan, index=datos.index)
    etiqueta[volatilidad_futura > umbral] = 1.0
    etiqueta[volatilidad_futura <= umbral] = 0.0
    etiqueta[volatilidad_futura.isna() | umbral.isna()] = np.nan
    return etiqueta.iloc[:-HORIZONTE_VOLATILIDAD].dropna().astype(int)


def crear_etiqueta_direccion(datos: pd.DataFrame) -> pd.Series:
    """Etiqueta dirección futura de SOL con umbral simétrico del 0,5%.

    1 = subida superior a +0,5%.
    0 = caída inferior a -0,5%.
    Los movimientos intermedios se descartan porque no compensan ruido ni costes.
    """
    cierre = datos[f"{ACTIVO_OBJETIVO.lower()}_close"]
    retorno_futuro = cierre.shift(-HORIZONTE_DIRECCION) / cierre - 1
    etiqueta = pd.Series(np.nan, index=datos.index)
    etiqueta[retorno_futuro > UMBRAL_DIRECCION] = 1.0
    etiqueta[retorno_futuro < -UMBRAL_DIRECCION] = 0.0
    return etiqueta.iloc[:-HORIZONTE_DIRECCION].dropna().astype(int)


def grupos_de_variables(columnas: list[str]) -> dict[str, list[str]]:
    """Define grupos de variables para las ablaciones finales."""
    sol = [c for c in columnas if c.startswith("sol_")]
    btc = [c for c in columnas if c.startswith("btc_")]
    eth = [c for c in columnas if c.startswith("eth_")]
    rel = [c for c in columnas if c.startswith("sol_btc_") or c.startswith("sol_eth_")]
    temporales = [c for c in columnas if c in {"hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"}]

    def contiene(cols: list[str], tokens: tuple[str, ...]) -> list[str]:
        return [c for c in cols if any(t in c for t in tokens)]

    sol_vol = contiene(sol, ("_vol_", "atr", "range"))
    sol_ret = contiene(sol, ("ret_", "mean_ret"))
    btc_ret_vol = contiene(btc, ("ret_", "mean_ret", "_vol_", "atr", "range"))
    eth_ret_vol = contiene(eth, ("ret_", "mean_ret", "_vol_", "atr", "range"))

    return {
        "SOL_RET_VOL": sorted(set(sol_vol + sol_ret + rel)),
        "ALL_ASSETS_NO_REL": sorted(set(sol + btc + eth)),
        "ALL_FEATURES": sorted(columnas),
        "RET_VOL_ALL_ASSETS": sorted(set(sol_vol + sol_ret + btc_ret_vol + eth_ret_vol + rel)),
        "SOL_ALL_PLUS_EXOG_RET_VOL": sorted(set(sol + btc_ret_vol + eth_ret_vol + rel)),
        "TEMPORALES": temporales,
    }
