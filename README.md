# LabCap4Heterogeneos

## Ejercicio B: GPU CUDA en la Jetson Nano

### Bloques, hilos y memoria 

#### Bloques

El kernel emplea un bloque por cada fila de entrada. Dado que el softmax se aplica a un único vector (una fila), la cuadrícula tiene un solo bloque. Este bloque calcula el softmax en su totalidad: el máximo, la exponencial con suma y la normalización.

Se utiliza un solo bloque porque las tres fases requieren resultados de todo el vector (la suma y el máximo), y únicamente los hilos pertenecientes a un solo bloque son capaces de compartir datos en la memoria compartida y sincronizarse con `__syncthreads()`. 

#### Hilos 

Cada bloque tiene 256 hilos. Los hilos distribuyen el vector con un recorrido con paso, en donde el hilo `t` procesa las posiciones `t, t + 256, t + 512, …`. Por lo tanto, cada hilo procesa un máximo de ⌈N/256⌉ elementos.
En cada etapa, cada hilo calcula primero un resultado parcial utilizando sus elementos (máximo local o suma local). Además, los 256 hilos combinan estos parciales con una reducción por árbol, en donde en cada paso, la mitad de los hilos combinan su valor con el de otro hilo, y el valor total se reduce a la mitad. Hay ocho pasos para la reducción, independientemente de N. Entre los pasos, `__syncthreads()` garantiza que todos los hilos finalicen antes de leer los resultados del paso anterior.

#### Memoria

- **Memoria global**: almacena el vector de entrada y el vector de salida. Cada hilo lee sus elementos de entrada en dos ocasiones (una para el exponencial y otra para el máximo). La salida también se emplea como espacio intermedio, en la etapa 2 se escribe `exp(x − m)` y, posteriormente, en la etapa 3, se vuelve a leer y dividir por la suma.
- **Memoria compartida:** un arreglo de 256 posiciones, una para cada hilo, que se reserva al iniciar el kernel. En FP32 (`float`, 4 B) utiliza 1024 B y en FP16 (`__half`, 2 B), emplea 512 B. Se emplea primero para disminuir el máximo y después se vuelve a utilizar para reducir la suma.
- **Registros:** Cada hilo almacena en variables locales el máximo parcial, la suma parcial y los resultados finales extraídos de la memoria compartida (`row_max` y `row_sum`).
- **Memoria del host:** El vector de entrada se crea en la CPU (en FP16, se convierte a __half desde float antes de ser copiado) y es transferido a la memoria global de la GPU usando `cudaMemcpy`. De igual manera, el resultado regresa a la CPU, y en FP16 se transforma a `float` para ser comparado con la referencia. Aunque en la Jetson Nano, CPU y GPU comparten la misma RAM física, el programa emplea memoria independiente y copias explícitas, de manera similar a lo que ocurre en una GPU dedicada.

### Comandos utilizados

#### Compilación

```bash
make
```
#### Ejecución

```bash
sudo nvpmodel -q
./softmax_cuda --precision fp32 --n 100 --reps 10000 --measure kernel
./softmax_cuda --precision fp32 --n 100 --reps 10000 --measure end-to-end
./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure kernel
./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure end-to-end
```

- `--measure kernel`: lanza el kernel `reps` veces con los datos ya en la GPU.
- `--measure end-to-end`: en cada repetición copia host to device, lanza el kernel y copia device to host (incluye transferencias, lanzamiento y sincronización).

#### Perfilador

```bash
for n in 10 100 500 1000; do
  sudo tegrastats --interval 100 --logfile logs/nvprof/tegra_fp32_n${n}.log &
  sudo /usr/local/cuda/bin/nvprof --log-file logs/nvprof/fp32_n${n}.txt \
    ./softmax_cuda --precision fp32 --n $n --reps 10000 --measure end-to-end
  sudo pkill tegrastats
done
# De la misma manera que fp16, solo cambiando fp16 por fp32
```

#### Resultados 

FP32

| Plataforma | N | Tiempo kernel (µs) | Tiempo end-to-end (µs) | Potencia (W) | Energía kernel (µJ) | Energía end-to-end (µJ) | J/elem kernel | J/elem end-to-end | Error máx. |
|---|---|---|---|---|---|---|---|---|---|
| GPU | 10 | 45.54 | 260.82 | 5 (asumida) | 227.7 | 1304.1 | 2.28e-5 | 1.30e-4 | 1.237e-8 |
| GPU | 100 | 46.19 | 235.81 | 5 (asumida) | 230.9 | 1179.1 | 2.31e-6 | 1.18e-5 | 1.301e-8 |
| GPU | 500 | 46.00 | 236.93 | 5 (asumida) | 230.0 | 1184.7 | 4.60e-7 | 2.37e-6 | 3.379e-9 |
| GPU | 1000 | 45.75 | 206.91 | 5 (asumida) | 228.7 | 1034.6 | 2.29e-7 | 1.03e-6 | 6.155e-9 |

FP16

| Plataforma | N | Tiempo kernel (µs) | Tiempo end-to-end (µs) | Potencia (W) | Energía kernel (µJ) | Energía end-to-end (µJ) | J/elem kernel | J/elem end-to-end | Error máx. |
|---|---|---|---|---|---|---|---|---|---|
| GPU | 10 | 35.17 | 235.99 | 5 (asumida) | 175.8 | 1180.0 | 1.76e-5 | 1.18e-4 | 2.054e-4 |
| GPU | 100 | 35.70 | 258.40 | 5 (asumida) | 178.5 | 1292.0 | 1.79e-6 | 1.29e-5 | 6.062e-5 |
| GPU | 500 | 34.95 | 278.19 | 5 (asumida) | 174.7 | 1390.9 | 3.49e-7 | 2.78e-6 | 1.579e-5 |
| GPU | 1000 | 35.40 | 211.35 | 5 (asumida) | 177.0 | 1056.7 | 1.77e-7 | 1.06e-6 | 8.607e-6 |

Dispersión (mínimo – máximo de 5 mediciones, µs)

| N | FP32 kernel | FP32 end-to-end | FP16 kernel | FP16 end-to-end |
|---|---|---|---|---|
| 10 | 35.13 – 57.46 | 256.83 – 301.00 | 34.67 – 36.73 | 235.16 – 238.26 |
| 100 | 46.02 – 69.37 | 235.21 – 238.07 | 35.10 – 69.42 | 232.84 – 298.18 |
| 500 | 45.49 – 65.16 | 235.83 – 243.41 | 34.39 – 35.40 | 238.01 – 289.97 |
| 1000 | 44.93 – 46.39 | 203.34 – 232.79 | 35.16 – 68.53 | 205.34 – 274.71 |

Frecuencia de GPU y temperatura (tegrastats)

| N | FP32 kernel | FP32 end-to-end | FP16 kernel | FP16 end-to-end |
|---|---|---|---|---|
| 10 | 307 MHz / 30.0 °C | 76 MHz / 30.5 °C | 460 MHz / 29.0 °C | 76 MHz / 30.0 °C |
| 100 | 230 MHz / 30.0 °C | 76 MHz / 31.0 °C | 307 MHz / 29.5 °C | 76 MHz / 30.0 °C |
| 500 | 384 MHz / 30.0 °C | 76 MHz / 31.0 °C | 460 MHz / 29.5 °C | 76 MHz / 30.5 °C |
| 1000 | 460 MHz / 30.5 °C | 153 MHz / 31.5 °C | 307 MHz / 29.5 °C | 153 MHz / 30.5 °C |

#### Salidas relevantes del perfilador 

Modo end-to-end, 10000 repeticiones. En todas estas corridas la GPU estuvo a **76.8 MHz**, así que FP16 y FP32 son directamente comparables.

Tiempos promedio por llamada (µs):

| N | Kernel FP32 | Kernel FP16 | HtoD FP32 | HtoD FP16 | DtoH FP32 | DtoH FP16 | % kernel (FP32) |
|---|---|---|---|---|---|---|---|
| 10 | 73.3 | 71.9 | 2.3 | 2.3 | 4.6 | 4.5 | 91 % |
| 100 | 73.8 | 72.4 | 3.0 | 2.6 | 5.2 | 4.9 | 90 % |
| 500 | 85.5 | 83.5 | 5.5 | 3.9 | 6.7 | 5.6 | 88 % |
| 1000 | 106.0 | 103.5 | 8.8 | 5.5 | 8.5 | 6.7 | 86 % |

Extracto de las salidas (N = 1000):

```
FP32
 GPU activities:   85.93%  1.05979s     10001  105.97us  105.22us  121.05us  softmax_rows_kernel(float const *, float*, int, int)
                    7.14%  88.099ms     10001  8.8080us  8.7500us  8.8560us  [CUDA memcpy HtoD]
                    6.93%  85.456ms     10002  8.5430us  8.4380us  12.240us  [CUDA memcpy DtoH]

FP16
 GPU activities:   89.48%  1.03485s     10001  103.47us  102.77us  122.09us  softmax_rows_kernel_fp16(__half const *, __half*, int, int)
                    5.76%  66.565ms     10002  6.6550us  6.5630us  10.313us  [CUDA memcpy DtoH]
                    4.77%  55.156ms     10001  5.5150us  5.4680us  6.9280us  [CUDA memcpy HtoD]
```

### Respuestas 

#### 1. ¿Qué etapa domina el tiempo en cada plataforma y cómo cambia con N?

En la GPU, el **kernel** (86-91 % de acuerdo con nvprof) predomina en el tiempo de la GPU; las copias son pequeñas (≈7-17 µs por invocación total). Sin embargo, en el end-to-end la duración total (≈210-280 µs) supera con creces a la suma de las copias y el kernel (≈80-120 µs a 76.8 MHz): lo que queda es **llamadas a la API desde la CPU y sincronización**. En el modo kernel, se solapan los lanzamientos; en cambio, en el modo end-to-end, cada `cudaMemcpy` se bloquea y no hay superposiciones.

Con N a una frecuencia constante (nvprof, 76.8 MHz), el kernel permanece igual entre N = 10 y N = 100 (cada hilo procesa como máximo un elemento y las barreras y reducciones, que son constantes, predominan) y aumenta aproximadamente 16 % en N = 500 y aproximadamente 44 % en N = 1000, cuando cada hilo maneja dos y cuatro elementos respectivamente. Las copias aumentan de aproximadamente 7 µs a cerca de 17 µs, pero la latencia fija sigue siendo la que predomina.
   
#### 2. ¿Qué diferencia de error aparece entre FP16 y FP32? ¿Se debe al formato, la exponencial o la reducción?

FP16 tiene un error de entre 1,000 y 17,000 veces más que FP32 (por ejemplo, con N = 100, el error de FP16 es de 6.1e-5 y el de FP32 es de 1.3e-8). El error absoluto de FP16 disminuye con N (2.1e-4 → 8.6e-6) debido a que las salidas son menores; sin embargo, el **error relativo** se mantiene en aproximadamente 0.1 %, lo cual equivale al orden de la precisión de FP16 (10 bits de mantisa, cerca de tres cifras decimales).

El **formato** parece ser la causa principal: simplemente cambiar la entrada a FP16 genera errores de hasta ~1e-3 en x para valores entre [2, 4) (por ejemplo, se almacena 3.0996 en vez de 3.0993), y ese error se transmite casi por completo a `exp(x − m)` y a la salida, lo que concuerda con el ≈0.1 % detectado. La exponencial (`hexp`) y la reducción (suma por árbol, que restringe la acumulación de error a alrededor de 8 niveles) tienen un menor impacto. 






