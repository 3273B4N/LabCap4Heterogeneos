## Ejercicio D: Energía

Para determinar el consumo energético, la energía de cada llamada a la función se calculó multiplicando la potencia por el tiempo de ejecución. Asimismo, la energía por elemento se obtuvo dividiendo la energía de cada llamada entre la cantidad de elementos (N):

```text
Energía (µJ) = Potencia (W) × Tiempo (µs)
J/elem.      = Energía (J) / N
```

Como la NVIDIA Jetson Nano usada (2GB) no cuenta con un sensor de potencia integrado, la herramienta `tegrastats` no reporta el consumo real de la placa. Por indicación del profesor, se asume un consumo constante de 5 W tanto para la CPU como para la GPU, valor que corresponde al límite del perfil de energía en el que se ejecutaron las pruebas (`nvpmodel` modo 1).

En la GPU, el tiempo, la energía y J/elem. se muestran como kernel / end-to-end; en la CPU se usa el tiempo de la función.

### FP32

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---|---|---|---|---|---|---|
| CPU | 10 | 0.495 | 5 | 2.47 | 2.47e-07 | 1.679e-08 |
| CPU | 100 | 4.415 | 5 | 22.08 | 2.21e-07 | 3.492e-09 |
| CPU | 500 | 22.007 | 5 | 110.04 | 2.20e-07 | 3.761e-09 |
| CPU | 1000 | 43.986 | 5 | 219.93 | 2.20e-07 | 6.013e-09 |
| GPU | 10 | 45.54 / 260.82 | 5 | 227.7 / 1304.1 | 2.28e-05 / 1.30e-04 | 1.237e-08 |
| GPU | 100 | 46.19 / 235.81 | 5 | 230.9 / 1179.0 | 2.31e-06 / 1.18e-05 | 1.301e-08 |
| GPU | 500 | 46.00 / 236.93 | 5 | 230.0 / 1184.7 | 4.60e-07 / 2.37e-06 | 3.379e-09 |
| GPU | 1000 | 45.75 / 206.91 | 5 | 228.8 / 1034.5 | 2.29e-07 / 1.03e-06 | 6.155e-09 |
| KV260 | 10 |  |  |  |  |  |
| KV260 | 100 |  |  |  |  |  |
| KV260 | 500 |  |  |  |  |  |
| KV260 | 1000 |  |  |  |  |  |

### FP16

| Plataforma | N | Tiempo (µs) | Potencia (W) | Energía (µJ) | J/elem. | Error máx. |
|---|---|---|---|---|---|---|
| CPU | 10 | 0.530 | 5 | 2.65 | 2.65e-07 | 1.527e-04 |
| CPU | 100 | 4.966 | 5 | 24.83 | 2.48e-07 | 6.062e-05 |
| CPU | 500 | 24.715 | 5 | 123.58 | 2.47e-07 | 1.257e-05 |
| CPU | 1000 | 49.355 | 5 | 246.78 | 2.47e-07 | 8.612e-06 |
| GPU | 10 | 35.17 / 235.99 | 5 | 175.9 / 1180.0 | 1.76e-05 / 1.18e-04 | 2.054e-04 |
| GPU | 100 | 35.70 / 258.40 | 5 | 178.5 / 1292.0 | 1.79e-06 / 1.29e-05 | 6.062e-05 |
| GPU | 500 | 34.95 / 278.19 | 5 | 174.8 / 1391.0 | 3.49e-07 / 2.78e-06 | 1.579e-05 |
| GPU | 1000 | 35.40 / 211.35 | 5 | 177.0 / 1056.8 | 1.77e-07 / 1.06e-06 | 8.607e-06 |
| KV260 | 10 |  |  |  |  |  |
| KV260 | 100 |  |  |  |  |  |
| KV260 | 500 |  |  |  |  |  |
| KV260 | 1000 |  |  |  |  |  |
