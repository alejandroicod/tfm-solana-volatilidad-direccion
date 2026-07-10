"""Instalador sencillo de dependencias del proyecto.

Uso:
    python scripts/instalar_dependencias.py

El script invoca pip con las librerías necesarias para reproducir el pipeline.
Se deja en Python, como pidió el enunciado, para que pueda ejecutarse en
cualquier entorno donde exista Python y pip.
"""

from __future__ import annotations

import subprocess
import sys


DEPENDENCIAS = [
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "matplotlib",
    "tqdm",
    "requests",
]


def instalar_dependencias() -> None:
    """Instala las dependencias necesarias con pip."""
    comando = [sys.executable, "-m", "pip", "install", *DEPENDENCIAS]
    print("Instalando dependencias:")
    print(" ".join(comando))
    subprocess.check_call(comando)


if __name__ == "__main__":
    instalar_dependencias()
