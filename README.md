# LabCap4Heterogeneos

Laboratorio del Capítulo 4 — Plataformas heterogéneas. Medición de tiempo de
ejecución, consumo energético y error numérico del kernel **softmax** en
**CPU** (host), **GPU** NVIDIA Jetson Nano 2GB y **KV260** (Kria K26 SOM,
flujo Vitis/Kria con XRT), en **FP32** y **FP16**.


## Estructura del proyecto

    ├── CPU/                        # Ejercicio A (Jetson Nano)
    │   ├── softmax_cpu.cpp
    │   ├── Makefile
    │   ├── run_cpu.sh
    │   └── logs/                   # cpu_results.csv, tegrastats, etc.
    ├── GPU/                        # Ejercicio B (CUDA)
    │   ├── softmax_cuda.cu
    │   ├── Makefile
    │   └── logs/nvprof/
    ├── FPGA/                        # Síntesis y pruebas KV260
    │   ├── Energía/                 # Ejercicio D (host + KV260)
    │     ├── bench_softmax.cpp
    │     ├── energy_agregar.py
    │     ├── power_sampler.py
    │     └── run_energy.sh
    │   ├── SW/                      # *_fp32/ y *_fp16/ con csynth, cosim, csim
    │     ├── bench_softmax.cpp       
    │     ├── energy_agregar.py
    │     ├── power_sampler.py
    │     └── run_energy.sh              
    │   ├── HW/                      # *_fp32/ y *_fp16/ con csynth, cosim, csim
    │     ├── reports/               # reportes de utilización
    │     ├── platform/   
    │     ├── Makefile
    │     ├── softmax_hls.cpp
    │     └── tb.cpp  
    │   ├── binaries/                # Bitstreams                 
    └── README.md

---

## Ejercicio A — CPU de la Jetson Nano

Se usó como base el código de la práctica 3, `softmax_openmp.c`, del cual se
reutilizaron los tres lazos del softmax y el lazo de repeticiones. Como vector
de entrada se usa `CPU/Vector_n1000_20.txt` (semilla 20), y para los tamaños
menores a 1000 se toman los primeros N valores.

### Plataforma

Las mediciones fueron realizadas en la NVIDIA Jetson Nano 2GB Developer Kit en
modo de potencia de 5 W (`nvpmodel` modo 1), en el que solo están activos 2 de
los 4 núcleos. Según `tegrastats`, mientras corrían las pruebas la CPU estuvo a
921 MHz y su temperatura se mantuvo entre 28.5 y 30 °C. El programa corre en un
solo hilo, por eso en el registro se ve un núcleo al 100 % y el otro casi sin
uso.

    $ cat /proc/device-tree/model
    NVIDIA Jetson Nano 2GB Developer Kit

    $ lscpu
    Architecture:         aarch64
    On-line CPU(s) list:  0,1
    Off-line CPU(s) list: 2,3
    Model name:           Cortex-A57
    L1d cache:            32K
    L2 cache:             2048K
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32

### Manejo de FP16

Para trabajar con FP16 se utilizó el tipo `__fp16`. El procesador de la Nano
(un Cortex-A57) permite convertir números entre *half* y *float*, pero no
soporta operaciones aritméticas directamente en *half* (faltan las banderas
`fphp` y `asimdhp` en `lscpu`), y la librería matemática no incluye una
exponencial para *half*. Por estas limitaciones, la solución fue almacenar los
datos en *half* pero realizar los cálculos de la exponencial y la suma en
*float*. Se agregó el parámetro `--sum-fp16` para forzar que la suma se acumule
en *half* (variante etiquetada como `fp16sum`), que sirve para medir cuánto
error agrega acumular la suma en *half*.

### Verificaciones

El programa calcula primero el softmax en *double* y lo usa como referencia.
Luego revisa tres condiciones:

- **Suma de la salida:** debe ser 1 (tolerancia 1e-4 en FP32, 1e-2 en FP16).
- **Rango válido:** todos los valores entre 0 y 1.
- **Posición del máximo:** misma posición que en la referencia.

Si se cumplen, reporta "OK". También reporta el peor caso en error absoluto y
relativo.

### Comandos

Compilación:

    cd CPU
    make
    # o manualmente
    g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm

Ejecución de una configuración:

    sudo nvpmodel -q
    ./softmax_cpu --precision fp32 --n 100 --reps 10000 --measure function
    ./softmax_cpu --precision fp16 --n 100 --reps 10000 --measure function
    ./softmax_cpu --precision fp32 --n 100 --reps 10000 --measure program
    ./softmax_cpu --precision fp16 --n 100 --reps 10000 --measure program
    ./softmax_cpu --precision fp16 --sum-fp16 --n 1000 --reps 10000

`--measure function` mide solo las llamadas a softmax; `--measure program`
mide el total (lectura, conversión, llamadas, referencia y verificación). En
ambos casos hay una ejecución de calentamiento y 5 mediciones efectivas; se
reporta mediana, mínimo y máximo.

Para correr todas las configuraciones:

    cd CPU
    ./run_cpu.sh

Guarda salidas en `CPU/logs/` y junta todo en `CPU/logs/cpu_results.csv`,
además de `nvpmodel -q`, `lscpu` y el registro de `tegrastats`.

Formato de cada medición:

    precision, n, reps, measure, total_ms, us_por_llamada, error_abs, run, error_rel, suma

### Resultados

Tiempos por llamada, 10 000 repeticiones y 5 mediciones por configuración.
Datos crudos en `CPU/logs/cpu_results.csv`.

#### Solo la función

| Variante | N | Mediana (µs) | Mínimo (µs) | Máximo (µs) |
|---|---|---|---|---|
| FP32 | 10 | 0.495 | 0.493 | 0.511 |
| FP32 | 100 | 4.415 | 4.414 | 4.417 |
| FP32 | 500 | 22.007 | 21.998 | 22.038 |
| FP32 | 1000 | 43.986 | 43.984 | 43.990 |
| FP16 | 10 | 0.530 | 0.528 | 0.614 |
| FP16 | 100 | 4.966 | 4.964 | 4.970 |
| FP16 | 500 | 24.715 | 24.690 | 24.736 |
| FP16 | 1000 | 49.355 | 49.339 | 49.375 |
| FP16sum | 10 | 0.607 | 0.606 | 0.620 |
| FP16sum | 100 | 5.612 | 5.597 | 5.637 |
| FP16sum | 500 | 27.782 | 27.779 | 27.949 |
| FP16sum | 1000 | 56.546 | 56.541 | 56.750 |

Para N ≥ 100 la diferencia entre mínimo y máximo es menor a 1 %. Con N = 10 la
variación es mayor (hasta 4 % en FP32 y 16 % en una medición de FP16), porque
cada medición dura apenas unos 5 ms y cualquier interrupción del sistema
operativo pesa más.

#### Programa completo

| Variante | N | Mediana (µs) | Mínimo (µs) | Máximo (µs) |
|---|---|---|---|---|
| FP32 | 10 | 0.499 | 0.499 | 0.506 |
| FP32 | 100 | 4.436 | 4.434 | 4.445 |
| FP32 | 500 | 22.063 | 22.059 | 22.068 |
| FP32 | 1000 | 44.162 | 44.156 | 44.179 |
| FP16 | 10 | 0.534 | 0.533 | 0.539 |
| FP16 | 100 | 4.986 | 4.982 | 4.995 |
| FP16 | 500 | 24.769 | 24.753 | 24.774 |
| FP16 | 1000 | 49.496 | 49.490 | 49.522 |
| FP16sum | 10 | 0.612 | 0.611 | 0.621 |
| FP16sum | 100 | 5.622 | 5.619 | 5.640 |
| FP16sum | 500 | 27.881 | 27.877 | 27.897 |
| FP16sum | 1000 | 56.763 | 56.748 | 56.797 |

#### Error contra FP64

| Variante | N | Error absoluto máx. | Error relativo máx. | Suma de la salida | Verificación |
|---|---|---|---|---|---|
| FP32 | 10 | 1.679e-08 | 1.553e-07 | 1.00000 | OK |
| FP32 | 100 | 3.492e-09 | 3.273e-07 | 1.00000 | OK |
| FP32 | 500 | 3.761e-09 | 5.346e-07 | 1.00000 | OK |
| FP32 | 1000 | 6.013e-09 | 1.155e-06 | 1.00000 | OK |
| FP16 | 10 | 1.527e-04 | 6.630e-04 | 0.99985 | OK |
| FP16 | 100 | 6.062e-05 | 1.317e-03 | 1.00008 | OK |
| FP16 | 500 | 1.257e-05 | 5.464e-03 | 1.00001 | OK |
| FP16 | 1000 | 8.612e-06 | 1.211e-02 | 0.99999 | OK |
| FP16sum | 10 | 2.747e-04 | 1.193e-03 | 0.99972 | OK |
| FP16sum | 100 | 6.062e-05 | 1.697e-03 | 1.00022 | OK |
| FP16sum | 500 | 9.090e-05 | 1.102e-02 | 1.00592 | OK |
| FP16sum | 1000 | 8.450e-05 | 2.280e-02 | 1.01128 | Falla la suma |

---

## Ejercicio B — GPU CUDA en la Jetson Nano

### Bloques, hilos y memoria

**Bloques.** El kernel emplea un bloque por cada fila de entrada. Como el
softmax se aplica a un único vector (una fila), la cuadrícula tiene un solo
bloque. Se usa un solo bloque porque las tres fases (máximo, exponencial+suma
y normalización) requieren resultados de todo el vector, y solo hilos de un
mismo bloque pueden compartir memoria compartida y sincronizarse con
`__syncthreads()`.

**Hilos.** Cada bloque tiene 256 hilos. El hilo `t` procesa las posiciones
`t, t + 256, t + 512, …`, es decir, un máximo de ⌈N/256⌉ elementos. En cada
etapa cada hilo calcula un parcial (máximo o suma local) y luego los 256 hilos
combinan con una reducción por árbol: en cada paso la mitad de los hilos
combina su valor con el de otro y el total se reduce a la mitad. Hay 8 pasos
independientemente de N, con `__syncthreads()` entre pasos.

**Memoria.**

- **Global:** vector de entrada y de salida. Cada hilo lee sus elementos dos
  veces (una para el máximo y otra para la exponencial). La salida se usa como
  intermedio: en la etapa 2 se escribe `exp(x − m)` y en la etapa 3 se relee y
  se divide por la suma.
- **Compartida:** arreglo de 256 posiciones (1024 B en FP32, 512 B en FP16).
  Se usa primero para reducir el máximo y luego para reducir la suma.
- **Registros:** máximo parcial, suma parcial y resultados finales (`row_max`,
  `row_sum`).
- **Host:** el vector se crea en CPU (en FP16 se convierte a `__half` antes de
  copiar) y se transfiere con `cudaMemcpy`; el resultado regresa y en FP16 se
  transforma a `float` para comparar. Aunque CPU y GPU comparten RAM física en
  la Jetson Nano, el programa usa memoria independiente y copias explícitas,
  como en una GPU dedicada.

### Comandos

Compilación:

    make

Ejecución:

    sudo nvpmodel -q
    ./softmax_cuda --precision fp32 --n 100 --reps 10000 --measure kernel
    ./softmax_cuda --precision fp32 --n 100 --reps 10000 --measure end-to-end
    ./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure kernel
    ./softmax_cuda --precision fp16 --n 100 --reps 10000 --measure end-to-end

- `--measure kernel`: lanza el kernel `reps` veces con datos ya en la GPU.
- `--measure end-to-end`: cada repetición copia HtoD, lanza kernel y copia
  DtoH (incluye transferencias, lanzamiento y sincronización).

Perfilador:

    for n in 10 100 500 1000; do
      sudo tegrastats --interval 100 --logfile logs/nvprof/tegra_fp32_n${n}.log &
      sudo /usr/local/cuda/bin/nvprof --log-file logs/nvprof/fp32_n${n}.txt \
        ./softmax_cuda --precision fp32 --n $n --reps 10000 --measure end-to-end
      sudo pkill tegrastats
    done
    # De la misma manera que fp16, solo cambiando fp16 por fp32

### Resultados

#### FP32

| Plataforma | N | Tiempo kernel (µs) | Tiempo end-to-end (µs) | Potencia (W) | Energía kernel (µJ) | Energía end-to-end (µJ) | J/elem kernel | J/elem end-to-end | Error máx. |
|---|---|---|---|---|---|---|---|---|---|
| GPU | 10 | 45.54 | 260.82 | 5 (asumida) | 227.7 | 1304.1 | 2.28e-5 | 1.30e-4 | 1.237e-8 |
| GPU | 100 | 46.19 | 235.81 | 5 (asumida) | 230.9 | 1179.1 | 2.31e-6 | 1.18e-5 | 1.301e-8 |
| GPU | 500 | 46.00 | 236.93 | 5 (asumida) | 230.0 | 1184.7 | 4.60e-7 | 2.37e-6 | 3.379e-9 |
| GPU | 1000 | 45.75 | 206.91 | 5 (asumida) | 228.7 | 1034.6 | 2.29e-7 | 1.03e-6 | 6.155e-9 |

#### FP16

| Plataforma | N | Tiempo kernel (µs) | Tiempo end-to-end (µs) | Potencia (W) | Energía kernel (µJ) | Energía end-to-end (µJ) | J/elem kernel | J/elem end-to-end | Error máx. |
|---|---|---|---|---|---|---|---|---|---|
| GPU | 10 | 35.17 | 235.99 | 5 (asumida) | 175.8 | 1180.0 | 1.76e-5 | 1.18e-4 | 2.054e-4 |
| GPU | 100 | 35.70 | 258.40 | 5 (asumida) | 178.5 | 1292.0 | 1.79e-6 | 1.29e-5 | 6.062e-5 |
| GPU | 500 | 34.95 | 278.19 | 5 (asumida) | 174.7 | 1390.9 | 3.49e-7 | 2.78e-6 | 1.579e-5 |
| GPU | 1000 | 35.40 | 211.35 | 5 (asumida) | 177.0 | 1056.7 | 1.77e-7 | 1.06e-6 | 8.607e-6 |

#### Dispersión (mínimo – máximo de 5 mediciones, µs)

| N | FP32 kernel | FP32 end-to-end | FP16 kernel | FP16 end-to-end |
|---|---|---|---|---|
| 10 | 35.13 – 57.46 | 256.83 – 301.00 | 34.67 – 36.73 | 235.16 – 238.26 |
| 100 | 46.02 – 69.37 | 235.21 – 238.07 | 35.10 – 69.42 | 232.84 – 298.18 |
| 500 | 45.49 – 65.16 | 235.83 – 243.41 | 34.39 – 35.40 | 238.01 – 289.97 |
| 1000 | 44.93 – 46.39 | 203.34 – 232.79 | 35.16 – 68.53 | 205.34 – 274.71 |

#### Frecuencia de GPU y temperatura (tegrastats)

| N | FP32 kernel | FP32 end-to-end | FP16 kernel | FP16 end-to-end |
|---|---|---|---|---|
| 10 | 307 MHz / 30.0 °C | 76 MHz / 30.5 °C | 460 MHz / 29.0 °C | 76 MHz / 30.0 °C |
| 100 | 230 MHz / 30.0 °C | 76 MHz / 31.0 °C | 307 MHz / 29.5 °C | 76 MHz / 30.0 °C |
| 500 | 384 MHz / 30.0 °C | 76 MHz / 31.0 °C | 460 MHz / 29.5 °C | 76 MHz / 30.5 °C |
| 1000 | 460 MHz / 30.5 °C | 153 MHz / 31.5 °C | 307 MHz / 29.5 °C | 153 MHz / 30.5 °C |

#### Salidas relevantes del perfilador

Modo end-to-end, 10 000 repeticiones. En todas estas corridas la GPU estuvo a
**76.8 MHz**, así que FP16 y FP32 son directamente comparables.

Tiempos promedio por llamada (µs):

| N | Kernel FP32 | Kernel FP16 | HtoD FP32 | HtoD FP16 | DtoH FP32 | DtoH FP16 | % kernel (FP32) |
|---|---|---|---|---|---|---|---|
| 10 | 73.3 | 71.9 | 2.3 | 2.3 | 4.6 | 4.5 | 91 % |
| 100 | 73.8 | 72.4 | 3.0 | 2.6 | 5.2 | 4.9 | 90 % |
| 500 | 85.5 | 83.5 | 5.5 | 3.9 | 6.7 | 5.6 | 88 % |
| 1000 | 106.0 | 103.5 | 8.8 | 5.5 | 8.5 | 6.7 | 86 % |

Extracto de las salidas (N = 1000):

    FP32
     GPU activities:   85.93%  1.05979s     10001  105.97us  105.22us  121.05us  softmax_rows_kernel(float const *, float*, int, int)
                        7.14%  88.099ms     10001  8.8080us  8.7500us  8.8560us  [CUDA memcpy HtoD]
                        6.93%  85.456ms     10002  8.5430us  8.4380us  12.240us  [CUDA memcpy DtoH]

    FP16
     GPU activities:   89.48%  1.03485s     10001  103.47us  102.77us  122.09us  softmax_rows_kernel_fp16(__half const *, __half*, int, int)
                        5.76%  66.565ms     10002  6.6550us  6.5630us  10.313us  [CUDA memcpy DtoH]
                        4.77%  55.156ms     10001  5.5150us  5.4680us  6.9280us  [CUDA memcpy HtoD]

---

## Ejercicio D — Energía

Para determinar el consumo energético, la energía de cada llamada a la función
se calculó multiplicando la potencia por el tiempo de ejecución. La energía por
elemento se obtuvo dividiendo la energía de cada llamada entre la cantidad de
elementos (N):

    Energía (µJ) = Potencia (W) × Tiempo (µs)
    J/elem.      = Energía (J) / N

Como la NVIDIA Jetson Nano usada (2GB) no cuenta con un sensor de potencia
integrado, `tegrastats` no reporta el consumo real de la placa. Por indicación
del profesor, se asume un consumo constante de **5 W** tanto para la CPU como
para la GPU, valor que corresponde al límite del perfil de energía en el que se
ejecutaron las pruebas (`nvpmodel` modo 1).

En la GPU, el tiempo, la energía y J/elem. se muestran como **kernel /
end-to-end**; en la CPU se usa el tiempo de la función.

En la **KV260** se midió la potencia real de la placa, por lo que sus columnas
de potencia, energía y J/elem. también se reportan como **valor / valor**
(kernel / end-to-end). Ver la **Zona de potencia — KV260** más abajo para la
declaración del punto de medición.

### FP32

| Plataforma | N    | Tiempo (µs)       | Potencia (W)      | Energía (µJ)      | J/elem.                  | Error máx. |
|------------|------|-------------------|-------------------|-------------------|--------------------------|------------|
| CPU        | 10   | 0.495             | 5                 | 2.47              | 2.47e-07                 | 1.679e-08  |
| CPU        | 100  | 4.415             | 5                 | 22.08             | 2.21e-07                 | 3.492e-09  |
| CPU        | 500  | 22.007            | 5                 | 110.04            | 2.20e-07                 | 3.761e-09  |
| CPU        | 1000 | 43.986            | 5                 | 219.93            | 2.20e-07                 | 6.013e-09  |
| GPU        | 10   | 45.54 / 260.82    | 5                 | 227.7 / 1304.1    | 2.28e-05 / 1.30e-04      | 1.237e-08  |
| GPU        | 100  | 46.19 / 235.81    | 5                 | 230.9 / 1179.0    | 2.31e-06 / 1.18e-05      | 1.301e-08  |
| GPU        | 500  | 46.00 / 236.93    | 5                 | 230.0 / 1184.7    | 4.60e-07 / 2.37e-06      | 3.379e-09  |
| GPU        | 1000 | 45.75 / 206.91    | 5                 | 228.8 / 1034.5    | 2.29e-07 / 1.03e-06      | 6.155e-09  |
| KV260      | 10   | 43.2 / 49.4       | 3.547 / 3.511     | 153.3 / 173.4     | 1.53e-05 / 1.73e-05      | 1.56e-08   |
| KV260      | 100  | 49.4 / 55.0       | 3.573 / 3.532     | 176.5 / 194.4     | 1.76e-06 / 1.94e-06      | 4.18e-09   |
| KV260      | 500  | 67.0 / 73.2       | 3.629 / 3.587     | 243.2 / 262.5     | 4.86e-07 / 5.25e-07      | 1.23e-09   |
| KV260      | 1000 | 85.9 / 91.9       | 3.639 / 3.598     | 312.4 / 330.7     | 3.12e-07 / 3.31e-07      | 2.15e-09   |

### FP16

| Plataforma | N    | Tiempo (µs)       | Potencia (W)      | Energía (µJ)      | J/elem.                  | Error máx. |
|------------|------|-------------------|-------------------|-------------------|--------------------------|------------|
| CPU        | 10   | 0.530             | 5                 | 2.65              | 2.65e-07                 | 1.527e-04  |
| CPU        | 100  | 4.966             | 5                 | 24.83             | 2.48e-07                 | 6.062e-05  |
| CPU        | 500  | 24.715            | 5                 | 123.58            | 2.47e-07                 | 1.257e-05  |
| CPU        | 1000 | 49.355            | 5                 | 246.78            | 2.47e-07                 | 8.612e-06  |
| GPU        | 10   | 35.17 / 235.99    | 5                 | 175.9 / 1180.0    | 1.76e-05 / 1.18e-04      | 2.054e-04  |
| GPU        | 100  | 35.70 / 258.40    | 5                 | 178.5 / 1292.0    | 1.79e-06 / 1.29e-05      | 6.062e-05  |
| GPU        | 500  | 34.95 / 278.19    | 5                 | 174.8 / 1391.0    | 3.49e-07 / 2.78e-06      | 1.579e-05  |
| GPU        | 1000 | 35.40 / 211.35    | 5                 | 177.0 / 1056.8    | 1.77e-07 / 1.06e-06      | 8.607e-06  |
| KV260      | 10   | 44.6 / 50.6       | 3.611 / 3.573     | 161.0 / 180.9     | 1.61e-05 / 1.81e-05      | 5.27e-05   |
| KV260      | 100  | 49.2 / 55.7       | 3.631 / 3.593     | 178.5 / 200.1     | 1.79e-06 / 2.00e-06      | 5.95e-05   |
| KV260      | 500  | 68.6 / 74.5       | 3.685 / 3.648     | 252.8 / 271.9     | 5.06e-07 / 5.44e-07      | 6.08e-05   |
| KV260      | 1000 | 87.7 / 93.9       | 3.700 / 3.664     | 324.7 / 344.2     | 3.25e-07 / 3.44e-07      | 6.08e-05   |

### Compilación (host)

Requisitos: `g++` con C++17, XRT (`/usr/include/xrt`, `libxrt_coreutil`),
`pthread`.

    # Compilar FP16
    g++ -std=c++17 -O2 -DFP16 bench_softmax.cpp -o bench_fp16 \
        -I/usr/include/xrt -lxrt_coreutil -lpthread

    # Compilar FP32
    g++ -std=c++17 -O2 bench_softmax.cpp -o bench_fp32 \
        -I/usr/include/xrt -lxrt_coreutil -lpthread

### KV260

#### Paquete entregado y archivos

El paquete entregado es del **flujo Vitis/Kria (XRT)**, no el par `.bit` +
`.hwh` de PYNQ. Los offsets de registros a mano ya no aplican: XRT los maneja
con la información del `.xclbin`.

- **`.xclbin`**: bitstream empaquetado con metadata del kernel (nombre,
  argumentos, reloj). Es lo que abre XRT desde el host.
- **`.dtbo`**: overlay de device tree que el sistema necesita para cargar el
  diseño.
- **`.bin`**: depende de quién lo armó. En apps de Kria suele ser el bitstream
  en formato `.bit.bin` o el mismo `.xclbin` con otra extensión. Comparar con
  `md5sum *.bin *.xclbin` y `file *.bin`.
- **`shell.json`**: archivo de texto corto que describe el tipo de app.
  Probablemente falta; pedírselo a la otra persona.

#### Requisitos

    which xmutil
    xbutil --version

#### Barrido energético

    sudo env IDLE=30 MODES="e2e" \
      ONLY="v3_dataflow_fp16_150MHz v3_dataflow_fp32_200MHz" \
      nohup ./run_energy.sh 5 10 > run_e2e.out 2>&1 &


#### Declaración del punto de medición

- **Potencia:** medida en placa (INA260 por hwmon / `xmutil platformstats`),
  potencia absoluta de la SOM (con ~3.2 W de reposo).
- **Repeticiones:** mediana de 5 repeticiones.
- **Error:** contra FP64.
- **Modos:** FP32 a 200 MHz y FP16 a 150 MHz.
- **Formato de columnas:** `valor / valor` = kernel / end-to-end.
- **Tiempo end-to-end:** sale del ciclo de la medición de energía, no de las
  medianas de llamadas sueltas, para que Energía = Potencia × Tiempo.
- **Potencia por modo:** la del e2e es ~0.04 W menor que la del kernel, y unos
  0.016 W de esa diferencia son corrimiento de sesión.


#### Comparación end-to-end (KV260 vs GPU)

| Comparación (end-to-end)                           | N=10              | N=1000              |
|----------------------------------------------------|-------------------|---------------------|
| Tiempo KV260 contra GPU (FP32 / FP16)              | 5.3× / 4.7× menor | 2.25× / 2.25× menor |
| Energía por elemento KV260 contra GPU (FP32 / FP16)| 7.5× / 6.5× menor | 3.1× / 3.1× menor   |

Para el procesamiento end-to-end (transferencias, lanzamiento y
sincronización), la **KV260 supera a la GPU en tiempo y en energía por
elemento en todos los tamaños**: con N=1000 es **2.25× más rápida** y gasta
**3.1× menos energía por elemento**, y con N=10 la ventaja llega a **5.3× en
tiempo** y **7.5× en energía**. Entre FP16 y FP32 no cambia la conclusión de
la KV260.

> **Cautela:** las potencias de CPU y GPU son **asumidas (5 W)**, la de la
> KV260 es **medida**, así que parte de la ventaja energética de la KV260
> (~1.4×) viene de comparar 3.6 W medidos contra 5 W asumidos.

---

## Síntesis y pruebas del kernel `softmax_hls` (KV260)

Resultados de **síntesis C (csynth)** de Vitis HLS 2024.1 para el kernel
`softmax_hls` (dispositivo `xck26-sfvc784-2LV-c`, KV260), junto con los
resultados del testbench, la cosimulación y el timing post-implementación de
cada variante.

### Frecuencia objetivo

| Tipo | Frecuencia objetivo | Target Clock del reporte | Incertidumbre del reporte | Variantes con ese valor |
|---|---|---|---|---|
| FP32 | 200 MHz | 5.00 ns | 1.35 ns | las 4 (`reports/*_fp32/softmax_hls_csynth.rpt`) |
| FP16 | 150 MHz | 6.67 ns | 1.80 ns | las 4 (`reports/*_fp16/softmax_hls_csynth.rpt`) |

FP32 se sintetizó a 200 MHz y FP16 a 150 MHz. FP16 se bajó a 150 MHz porque a
200 MHz no cerraba timing.

### Recursos y frecuencia (csynth)

#### FP32

| Variante | Frec. objetivo (MHz) | II (25 / 44 / 75 / 99) | Fmax (MHz) | LUT | FF | BRAM_18K | URAM | DSP |
|---|---|---|---|---|---|---|---|---|
| v0_base | 200 | 1 / 2 / 9 / 1 | 273.97 | 4819 | 3261 | 10 | 0 | 9 |
| v1_pipeline | 200 | 1 / 1 / 5 / 1 | 248.82 | 5151 | 3473 | 10 | 0 | 9 |
| v2_partition | 200 | 1 / 1 / 5 / 4 | 248.82 | 7722 | 5399 | 16 | 0 | 9 |
| v3_dataflow | 200 | 1 / 1 / 5 / 4 | 248.82 | 7701 | 6330 | 16 | 0 | 9 |

#### FP16

| Variante | Frec. objetivo (MHz) | II (25 / 44 / 75 / 99) | Fmax (MHz) | LUT | FF | BRAM_18K | URAM | DSP |
|---|---|---|---|---|---|---|---|---|
| v0_base | 150 | 1 / 1 / 3 / 1 | 205.38 | 3910 | 2582 | 5 | 0 | 6 |
| v1_pipeline | 150 | 1 / 1 / 1 / 1 | 191.06 | 4013 | 2655 | 5 | 0 | 6 |
| v2_partition | 150 | 1 / 1 / 1 / 4 | 191.06 | 6958 | 4703 | 14 | 0 | 18 |
| v3_dataflow | 150 | 1 / 1 / 1 / 4 | 191.06 | 6985 | 5650 | 14 | 0 | 18 |

Períodos estimados (ns) de los que sale Fmax: FP32 v0 3.650, v1 4.019, v2
4.019, v3 4.019; FP16 v0 4.869, v1 5.234, v2 5.234, v3 5.234.

### II por bucle

Formato de cada celda: **II logrado (II objetivo)**. Todas las variantes
reportan los cuatro bucles con tipo de pipeline `yes` y un II objetivo de 1.

| Bucle | Variante | FP32 | FP16 |
|---|---|---|---|
| `VITIS_LOOP_25_1` | v0_base | 1 (1) | 1 (1) |
| `VITIS_LOOP_25_1` | v1_pipeline | 1 (1) | 1 (1) |
| `VITIS_LOOP_25_1` | v2_partition | 1 (1) | 1 (1) |
| `VITIS_LOOP_25_1` | v3_dataflow | 1 (1) | 1 (1) |
| `VITIS_LOOP_44_2` | v0_base | 2 (1) | 1 (1) |
| `VITIS_LOOP_44_2` | v1_pipeline | 1 (1) | 1 (1) |
| `VITIS_LOOP_44_2` | v2_partition | 1 (1) | 1 (1) |
| `VITIS_LOOP_44_2` | v3_dataflow | 1 (1) | 1 (1) |
| `VITIS_LOOP_75_2` | v0_base | 9 (1) | 3 (1) |
| `VITIS_LOOP_75_2` | v1_pipeline | 5 (1) | 1 (1) |
| `VITIS_LOOP_75_2` | v2_partition | 5 (1) | 1 (1) |
| `VITIS_LOOP_75_2` | v3_dataflow | 5 (1) | 1 (1) |
| `VITIS_LOOP_99_1` | v0_base | 1 (1) | 1 (1) |
| `VITIS_LOOP_99_1` | v1_pipeline | 1 (1) | 1 (1) |
| `VITIS_LOOP_99_1` | v2_partition | 4 (1) | 4 (1) |
| `VITIS_LOOP_99_1` | v3_dataflow | 4 (1) | 4 (1) |

- En v0–v2 cada bucle tiene su propio reporte
  (`softmax_hls_Pipeline_VITIS_LOOP_<n>_csynth.rpt`). En v3 los bucles quedan
  dentro de los módulos `read_data`, `find_max`, `comp_exp` y `norm_exp`; los de
  44, 75 y 99 conservan reporte con el mismo nombre de bucle y el 25 sale de
  `read_data_csynth.rpt`.
- Iteraciones (*Trip Count*): en v0 y v1 los bucles 44, 75 y 99 muestran
  `0 ~ 1023` y el 25 muestra `?`; en v2 y v3 los cuatro muestran `?`.

### Pruebas: testbench y cosimulación

El testbench (`tb.cpp`) llama a `softmax_hls(x, y, n)` con cuatro tamaños,
`N = 10, 100, 500, 1000`, y compara contra una referencia en `double`:

- **Entrada:** `x[i] = (i % 97) / 10.0 - 4.8` convertido al tipo del kernel
  (`float` con `USE_FLOAT32`, `half` en caso contrario).
- **Referencia:** softmax en `double` con resta del máximo, a partir de los
  mismos valores de entrada ya convertidos.
- **Métricas por tamaño:** error máximo absoluto entre `y[i]` y la referencia,
  y la suma de las salidas.
- **Tolerancia:** `1e-5` en FP32 y `1e-2` en FP16. Si el error máximo supera la
  tolerancia para algún tamaño, el testbench imprime `RESULTADO: FALLO` y
  devuelve 1; si no, imprime `RESULTADO: OK`.
- **Evidencia:** `softmax_hls_csim.log` (simulación C), `tb_resultados.txt`
  (salida del testbench) y `softmax_hls_cosim.rpt` (cosimulación RTL).

Resultado de las 8 variantes: csim **PASS** (`CSim done with 0 errors`,
`RESULTADO: OK`) y cosim **PASS** en Verilog. En `softmax_hls_cosim.rpt` la
columna VHDL aparece como `NA`.

Cada celda de las tablas siguientes es `error máximo / suma` tomada de
`softmax_hls_csim.log`.

#### FP32 (tolerancia 1e-5)

| Variante | csim | cosim (Verilog) | N=10 | N=100 | N=500 | N=1000 |
|---|---|---|---|---|---|---|
| v0_base | PASS | PASS | 5.469e-09 / 1.00000 | 8.118e-09 / 1.00000 | 8.285e-09 / 1.00000 | 2.325e-09 / 1.00000 |
| v1_pipeline | PASS | PASS | 5.469e-09 / 1.00000 | 8.118e-09 / 1.00000 | 8.285e-09 / 1.00000 | 2.325e-09 / 1.00000 |
| v2_partition | PASS | PASS | 5.469e-09 / 1.00000 | 6.783e-09 / 1.00000 | 1.028e-09 / 1.00000 | 4.688e-10 / 1.00000 |
| v3_dataflow | PASS | PASS | 5.469e-09 / 1.00000 | 6.783e-09 / 1.00000 | 1.028e-09 / 1.00000 | 4.688e-10 / 1.00000 |

#### FP16 (tolerancia 1e-2)

| Variante | csim | cosim (Verilog) | N=10 | N=100 | N=500 | N=1000 |
|---|---|---|---|---|---|---|
| v0_base | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 7.385e-05 / 1.00065 | 8.341e-05 / 1.00235 |
| v1_pipeline | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 7.385e-05 / 1.00065 | 8.341e-05 / 1.00235 |
| v2_partition | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 5.762e-05 / 0.99711 | 5.801e-05 / 0.99518 |
| v3_dataflow | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 5.762e-05 / 0.99711 | 5.801e-05 / 0.99518 |

### Timing post-implementación

| Variante | Tipo | Reloj del kernel (enlace) | `impl_1` WNS / WHS | `impl_Area_Explore` WNS / WHS |
|---|---|---|---|---|
| v0_base | FP32 | 200 MHz | 1.280 / 0.010 | N/D |
| v0_base | FP16 | 150 MHz | 0.676 / 0.012 | 0.675 / 0.011 |
| v1_pipeline | FP32 | 200 MHz | 1.011 / 0.010 | 1.027 / 0.010 |
| v1_pipeline | FP16 | 150 MHz | 0.783 / 0.010 | 0.371 / 0.010 |
| v2_partition | FP32 | 200 MHz | 1.042 / 0.011 | 0.518 / 0.012 |
| v2_partition | FP16 | 150 MHz | 0.719 / 0.010 | 0.297 / 0.012 |
| v3_dataflow | FP32 | 200 MHz | 1.143 / 0.011 | 1.129 / 0.010 |
| v3_dataflow | FP16 | 150 MHz | 0.420 / 0.011 | 0.636 / 0.010 |

- El WNS/WHS es del diseño completo (`system_wrapper`), no solo del reloj del
  kernel.

---

## Respuestas

### 1. ¿Qué etapa domina el tiempo en cada plataforma y cómo cambia con N?

**CPU (Ejercicio A).** Domina el cálculo del softmax. El programa completo tarda
apenas entre 0.25 % y 0.9 % más que la función sola, por lo que leer el vector,
convertirlo y verificar el resultado representa menos del 1 % del tiempo total.
El tiempo crece de forma lineal con N, a razón de unos 44 ns por elemento en
FP32 (cerca de 41 ciclos a 921 MHz).

**GPU (Ejercicio B).** El **kernel** (86–91 % según `nvprof`) predomina en el
tiempo de la GPU; las copias son pequeñas (≈7–17 µs por invocación total). Sin
embargo, en el end-to-end la duración total (≈210–280 µs) supera con creces a
la suma de copias y kernel (≈80–120 µs a 76.8 MHz): lo que queda son
**llamadas a la API desde la CPU y sincronización**. En el modo kernel se
solapan los lanzamientos; en el modo end-to-end cada `cudaMemcpy` se bloquea y
no hay superposiciones.

Con N a una frecuencia constante (`nvprof`, 76.8 MHz), el kernel permanece
igual entre N = 10 y N = 100 (cada hilo procesa como máximo un elemento y
predominan las barreras y reducciones, que son constantes) y aumenta
aproximadamente 16 % en N = 500 y aproximadamente 44 % en N = 1000, cuando
cada hilo maneja dos y cuatro elementos respectivamente. Las copias aumentan de
aproximadamente 7 µs a cerca de 17 µs, pero la latencia fija sigue predominando.

**KV260 (Ejercicio D).** Para el procesamiento end-to-end (transferencias,
lanzamiento y sincronización), la **KV260 supera a la GPU en tiempo y en
energía por elemento en todos los tamaños**: con N=1000 es **2.25× más rápida**
y gasta **3.1× menos energía por elemento**, y con N=10 la ventaja llega a
**5.3× en tiempo** y **7.5× en energía**. La CPU sigue siendo la mejor en
tiempo y energía por elemento en la mayoría de las configuraciones, pero mide
solo la función sobre datos ya en memoria (no es like-for-like con los
aceleradores).

### 2. ¿Qué diferencia de error aparece entre FP16 y FP32? ¿Se debe al formato, la exponencial o la reducción?

**CPU (Ejercicio A).** FP16 tiene un error absoluto entre 8.6e-6 y 1.5e-4,
frente a 3.5e-9 a 1.7e-8 en FP32, es decir, entre 1 400 y 17 000 veces mayor.
El error absoluto baja con N porque las salidas son más pequeñas, pero el error
relativo sube de 6.6e-4 con N = 10 a 1.2e-2 con N = 1000: con más elementos
muchas probabilidades quedan por debajo de 6.1e-5, donde FP16 pierde precisión.

En la versión FP16 de la CPU la exponencial y la suma se calculan en *float*,
por lo que el error proviene del **formato**: redondear la entrada, los valores
intermedios y la salida a *half*. Para medir el efecto de la reducción se usó la
variante `fp16sum`, donde la suma se acumula en *half*: con N = 1000 el error
absoluto aumenta casi 10 veces (8.5e-5 contra 8.6e-6) y la salida suma 1.011.
Esto muestra que una reducción secuencial en FP16 sí es una fuente importante
de error.

**GPU (Ejercicio B).** FP16 tiene un error entre 1 000 y 17 000 veces mayor que
FP32 (por ejemplo, con N = 100, el error de FP16 es 6.1e-5 y el de FP32 es
1.3e-8). El error absoluto de FP16 disminuye con N (2.1e-4 → 8.6e-6) porque
las salidas son menores; sin embargo, el **error relativo** se mantiene en
aproximadamente 0.1 %, que es el orden de la precisión de FP16 (10 bits de
mantisa, cerca de tres cifras decimales).

El **formato** parece ser la causa principal: cambiar la entrada a FP16 genera
errores de hasta ~1e-3 en x para valores entre [2, 4) (por ejemplo, se almacena
3.0996 en vez de 3.0993), y ese error se transmite casi por completo a
`exp(x − m)` y a la salida, lo que concuerda con el ≈0.1 % detectado. La
exponencial (`hexp`) y la reducción (suma por árbol, que restringe la
acumulación de error a alrededor de 8 niveles) tienen un menor impacto.

**KV260 (Ejercicio D).** En las 8 variantes de `softmax_hls`, csim y cosim dan
**PASS**. FP32 mantiene errores ~10⁻⁹ en todas las variantes; FP16 sube a
≈10⁻⁵ (dentro de la tolerancia 1e-2), consistente con la menor precisión del
formato. La variante `v3_dataflow` es la que mejor error logra en FP32
(4.688e-10 a N=1000) y una de las mejores en FP16 (5.801e-05 a N=1000).

### 3. ¿Cómo se comparan las tres plataformas en energía por elemento?

Con los valores de referencia (CPU/GPU con potencia asumida de 5 W; KV260 con
potencia medida):

| N    | CPU FP32   | GPU FP32 (e2e) | KV260 FP32 | CPU FP16   | GPU FP16 (e2e) | KV260 FP16 |
|------|------------|----------------|------------|------------|----------------|------------|
| 10   | 2.47e-07   | 1.30e-04       | 1.73e-05   | 2.65e-07   | 1.18e-04       | 1.81e-05   |
| 100  | 2.21e-07   | 1.18e-05       | 1.94e-06   | 2.48e-07   | 1.29e-05       | 2.00e-06   |
| 500  | 2.20e-07   | 2.37e-06       | 5.25e-07   | 2.47e-07   | 2.78e-06       | 5.44e-07   |
| 1000 | 2.20e-07   | 1.03e-06       | 3.31e-07   | 2.47e-07   | 1.06e-06       | 3.44e-07   |

- A **N pequeño (10–100)**, la CPU es la más eficiente en J/elem., porque el
  overhead de lanzamiento de kernel en GPU/KV260 domina el tiempo y la energía.
- A **N grande (500–1000)**, la GPU y KV260 superan a la CPU en eficiencia
  energética: la KV260 llega a **3.31e-07 J/elem.** en FP32 y **3.44e-07** en
  FP16, frente a **2.20e-07 / 2.47e-07** de la CPU.
- La **GPU Jetson Nano** es la **menos eficiente en J/elem. end-to-end** para N
  pequeño, por el alto overhead de transferencia/lanzamiento (206–278 µs e2e
  frente a ~45 µs de kernel).
- **KV260** mejora a la GPU en e2e porque su tiempo total (43–94 µs) es mucho
  menor y su potencia real medida (~3.5–3.7 W) es inferior a los 5 W asumidos.

> **Cautela:** las potencias de CPU y GPU son **asumidas (5 W)**, la de la
> KV260 es **medida**, así que parte de la ventaja energética de la KV260
> (~1.4×) viene de comparar 3.6 W medidos contra 5 W asumidos.

### 4. ¿Qué variante de `softmax_hls` conviene para la KV260?

A partir de csynth, II por bucle, pruebas y timing post-implementación:

- **Recursos:** `v0_base` y `v1_pipeline` son las más livianas (≈4.8–5.2k LUT,
  10 BRAM en FP32); `v2_partition` y `v3_dataflow` suben a ≈7.7k LUT y 16 BRAM
  por el particionado. En FP16, `v2`/`v3` usan 18 DSP frente a 6 de `v0`/`v1`.
- **II:** `v0_base` no logra II=1 en los bucles 44 y 75 (2 y 9 en FP32; 1 y 3
  en FP16). `v1_pipeline` mejora el 75 a 5 (FP32) y 1 (FP16). `v2`/`v3` logran
  II=1 en 44 y 75, pero bajan a II=4 en el bucle 99.
- **Fmax:** `v0_base` tiene la Fmax más alta (273.97 MHz en FP32, 205.38 MHz en
  FP16). `v1`–`v3` quedan en ≈248.82 MHz (FP32) y ≈191.06 MHz (FP16), por
  encima de la frecuencia objetivo (200 / 150 MHz), así que cierran timing.
- **Timing post-implementación:** todos los WNS son positivos. `v0_base` FP32
  tiene el mayor margen (`impl_1` WNS 1.280 ns). En FP16, `v3_dataflow` tiene
  el menor margen (`impl_1` WNS 0.420 ns), aunque sigue siendo positivo.
- **Error:** `v2_partition` y `v3_dataflow` logran el menor error en FP32
  (4.688e-10 a N=1000) y en FP16 (5.801e-05 a N=1000), mejores que `v0`/`v1`.

**Recomendación:** para la KV260 conviene **`v3_dataflow`** si se prioriza
throughput (II=1 en 44 y 75) y el mejor error, aceptando más LUT/BRAM y un
margen de timing algo menor en FP16. Si se prioriza **área y margen de
timing**, `v0_base` es la más liviana y con mayor Fmax/WNS, a costa de peor II
en los bucles 44 y 75 y peor error. `v1_pipeline` es un punto intermedio
razonable en FP32.

### 5. ¿Qué limitaciones tiene FP16 en la CPU de la Jetson Nano?

El Cortex-A57 permite convertir entre *half* y *float*, pero no soporta
aritmética directamente en *half* (faltan `fphp` y `asimdhp` en `lscpu`) y la
libm no trae exponencial para *half*. Por eso, en la CPU se almacenan los datos
en *half* pero la exponencial y la suma se hacen en *float*; la variante
`--sum-fp16` fuerza la suma en *half* y muestra que una reducción secuencial en
FP16 sí agrega error importante (suma 1.011 a N=1000, error absoluto casi 10×
mayor). En la GPU y en la KV260, en cambio, sí hay soporte nativo de FP16
(`__half` en CUDA y `half` en HLS), por lo que la comparación FP16/FP32 es más
directa.

## Sección de uso de IA
Se hace uso de herramientas de IA como apoyo para comprender conceptos, generar ideas y mejorar la redacción de la documentación. La implementación, la validación y los resultados son responsabilidad del estudiante, quien asume la responsabilidad por el uso indebido o no descrito anteriormente.

Se adjuntan los enlaces compartidos de las conversaciones como evidencia.

https://claude.ai/share/7ccbf10c-bc11-447a-807b-8573b0606242


