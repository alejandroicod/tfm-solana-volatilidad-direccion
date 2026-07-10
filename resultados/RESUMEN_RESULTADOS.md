# Resumen de resultados finales

## Datos

- Activo objetivo: SOLUSDT.
- Variables exógenas: BTCUSDT y ETHUSDT.
- Frecuencia: velas OHLCV de 1 hora.
- Periodo común usado: 2022-01-01 a 2025-12-31.

## Etapa 1: filtro de volatilidad

La primera etapa clasifica si SOL tendrá alta o baja volatilidad en las próximas 12 horas.
Se probaron dos umbrales de régimen:

- q50: separación más equilibrada entre alta y baja volatilidad.
- q67: filtro más selectivo, centrado en episodios de mayor volatilidad.

Resultado principal q67:

- Mejor modelo: XGBoost.
- Accuracy test medio: 0,7774.
- Macro F1 test medio: 0,6591.
- Precision macro: 0,7241.
- Recall macro: 0,6601.

Grupo mínimo recomendado para etapa 1:

- `SOL_RET_VOL`.
- 33 variables.
- Modelo: Logistic Regression.
- Macro F1 test medio: 0,6635.

## Etapa 2: dirección condicionada

La segunda etapa se activa sólo cuando el filtro de volatilidad predice alta volatilidad.
El target es binario:

- 1: SOL sube más de +0,5% en las próximas 12 horas.
- 0: SOL cae más de -0,5% en las próximas 12 horas.

Comparativa XGBoost:

| Caso | Cobertura | Accuracy | Macro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Sin filtro de volatilidad | 100,0% | 0,5229 | 0,5099 | 0,5326 | 0,5282 |
| Con filtro XGBoost p67 | 21,6% | 0,6063 | 0,5553 | 0,5911 | 0,5742 |

La mejora principal se observa en accuracy (+8,34 puntos) y macro F1 (+4,54 puntos).

Mejor grupo de variables para etapa 2:

- `ALL_ASSETS_NO_REL`.
- 109 variables.
- Modelo: XGBoost.
- Accuracy test medio: 0,6025.
- Macro F1 test medio: 0,5663.

## Interpretación

La predicción direccional directa de SOL fue débil cuando se aplicó a todos los datos.
El filtro de volatilidad mejora el problema porque concentra la segunda etapa en ventanas con mayor amplitud de movimiento.
El enfoque final no es un backtest de trading, sino una medición de capacidad predictiva.

En trabajos futuros debe añadirse una capa de ejecución:

- operar largo cuando la segunda etapa indique subida;
- operar corto cuando indique caída;
- incorporar comisiones, spread, slippage, stop loss, take profit y tamaño de posición.
