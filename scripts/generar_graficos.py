"""Genera gráficas finales de EDA y resultados."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-tfm-final")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from configuracion import CARPETA_GRAFICOS, CARPETA_METRICAS
from datos import cargar_y_alinear


def guardar_figura(nombre: str) -> None:
    """Guarda la figura actual en la carpeta de gráficos."""
    CARPETA_GRAFICOS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(CARPETA_GRAFICOS / nombre, dpi=180, bbox_inches="tight")
    plt.close()


def grafico_precios_y_retornos() -> None:
    """Crea gráficas de EDA de precios y retornos."""
    datos = cargar_y_alinear()
    cierres = pd.DataFrame({
        "SOL": datos["sol_close"],
        "BTC": datos["btc_close"],
        "ETH": datos["eth_close"],
    })
    normalizados = cierres / cierres.iloc[0]
    retornos = np.log(cierres / cierres.shift(1)).dropna()

    normalizados.plot(figsize=(10, 4.8), linewidth=1.1)
    plt.title("Evolución normalizada de precios")
    plt.ylabel("Precio normalizado")
    plt.grid(alpha=0.25)
    guardar_figura("01_precios_normalizados.png")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, activo in zip(axes, ["SOL", "BTC", "ETH"]):
        serie = retornos[activo].clip(retornos[activo].quantile(0.005), retornos[activo].quantile(0.995))
        ax.hist(serie, bins=70, color="#2E74B5", alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(activo)
        ax.grid(alpha=0.2)
    guardar_figura("02_distribucion_retornos.png")


def grafico_metricas_etapas() -> None:
    """Crea gráficas comparativas de accuracy, macro F1, precision y recall."""
    etapa1 = pd.read_csv(CARPETA_METRICAS / "etapa1_agregado.csv")
    etapa2 = pd.read_csv(CARPETA_METRICAS / "etapa2_agregado.csv")

    mejores_etapa1 = etapa1.sort_values(["cuantil", "validacion_macro_f1"], ascending=[True, False]).groupby(["cuantil", "modelo"]).head(1)
    metricas = ["accuracy", "macro_f1", "precision", "recall"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, metrica in zip(axes.ravel(), metricas):
        pivote = mejores_etapa1.pivot(index="modelo", columns="cuantil", values=metrica)
        pivote.plot(kind="bar", ax=ax)
        ax.set_title(metrica)
        ax.set_ylim(0.35, 0.82)
        ax.grid(axis="y", alpha=0.25)
    guardar_figura("03_metricas_etapa1.png")

    mejores_etapa2 = etapa2[etapa2["modelo"].eq("XGBoost")].copy()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), sharey=True)
    for ax, metrica in zip(axes, metricas):
        ax.bar(mejores_etapa2["modo"], mejores_etapa2[metrica], color=["#54A24B", "#E45756"])
        ax.set_title(metrica)
        ax.set_ylim(0.45, 0.65)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    guardar_figura("04_etapa2_contra_sin_filtro.png")


def generar_todos_los_graficos() -> None:
    """Ejecuta todos los gráficos finales."""
    grafico_precios_y_retornos()
    grafico_metricas_etapas()


if __name__ == "__main__":
    generar_todos_los_graficos()
