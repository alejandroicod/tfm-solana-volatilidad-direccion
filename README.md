# Pipeline final TFM: Solana, volatilidad y dirección condicionada

Este repositorio contiene el código usado en el TFM.
El objetivo es estudiar un sistema de dos etapas:

1. **Etapa 1:** clasificar si SOL tendrá alta o baja volatilidad en las próximas 12 horas.
2. **Etapa 2:** cuando la etapa 1 predice alta volatilidad, clasificar si SOL subirá o bajará más de un umbral del 0,5% en las próximas 12 horas.

Se usan velas horarias OHLCV de Binance para SOL, BTC y ETH durante 2022-2025.
BTC y ETH se usan como variables exógenas porque SOL muestra dependencia del régimen general del mercado cripto.

## Instalación

```bash
python scripts/instalar_dependencias.py
```

O bien:

```bash
pip install -r requirements.txt
```

## Descarga de datos

```bash
python scripts/descargar_datos_binance.py
```

Los CSV se guardan en `datos/raw/`.

## Ejecución completa

```bash
python scripts/ejecutar_pipeline_completo.py
```

## Resultados incluidos

La carpeta `resultados/metricas/` contiene los CSV finales usados en el informe.
La carpeta `resultados/graficos/` contiene las gráficas finales.

## Notas metodológicas

- Las variables rolling usan sólo pasado y presente.
- El target se calcula con desplazamiento futuro, pero nunca se incluye como feature.
- La normalización se ajusta sólo con train dentro de cada fold.
- Los umbrales de probabilidad se seleccionan con validación, no con test.
