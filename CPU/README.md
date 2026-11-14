# Ejercicio A: CPU de la Jetson Nano

Se usó como base el código de la práctica 3, `softmax_openmp.c`, del cual se reutilizaron los tres lazos del softmax y el lazo de repeticiones. Como vector de entrada se usa `CPU/Vector_n1000_20.txt` (semilla 20), y para los tamaños menores a 1000 se toman los primeros N valores.

## Plataforma

Las mediciones fueron realizadas en la NVIDIA Jetson Nano 2GB Developer Kit en modo de potencia de 5W (`nvpmodel` modo 1), en el que solo están activos 2 de los 4 núcleos. Según `tegrastats`, mientras corrían las pruebas la CPU estuvo a 921 MHz y su temperatura se mantuvo entre 28.5 y 30 °C. El programa corre en un solo hilo, por eso en el registro se ve un núcleo al 100 % y el otro casi sin uso.

```
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
```

## Manejo de FP16

Para trabajar con FP16 se utilizó el tipo `__fp16`. Aquí apareció una limitación: el procesador de la Nano (un Cortex-A57) permite convertir números entre *half* y *float*, pero no soporta operaciones aritméticas directamente en *half*. Esto se comprobó revisando los `Flags` con un `lscpu`, donde se nota que faltan las banderas `fphp` y `asimdhp` necesarias para esto. Además, la librería matemática no incluye una función exponencial para *half*. Por estas limitaciones, la solución en esta versión fue almacenar los datos en *half*, pero realizar los cálculos de la exponencial y la suma en *float*. Adicionalmente, se agregó el parámetro `--sum-fp16` para forzar que la suma se acumule en *half*; esta variante sirve para evaluar cuánto error agrega acumular la suma en *half*, y en los resultados aparece etiquetada como `fp16sum`.

## Verificaciones

Para comprobar la validez de los resultados, el programa calcula primero el *softmax* en *double*, utilizando este resultado de mayor precisión como la referencia correcta. A partir de esto, se revisan tres condiciones principales:

- Suma de la salida: Dado que la función *softmax* devuelve probabilidades, la suma total debe ser 1. Se estableció una tolerancia de error de 1e-4 para FP32 y de 1e-2 para FP16, considerando que este último formato guarda menos decimales.
- Rango válido: Se verifica que todos los valores obtenidos estén entre 0 y 1, respetando la naturaleza de una probabilidad.
- Posición del máximo: Se comprueba que el valor más alto quede en la misma posición (índice) que en la referencia.

Si se cumplen estas tres pruebas, el programa reporta un "OK". Adicionalmente, se analiza la desviación de cada valor respecto a la referencia y se reporta el peor caso evaluado de dos maneras: mediante el error absoluto (la diferencia matemática directa) y el error relativo (la proporción de esa diferencia respecto al valor correcto).

## Comandos utilizados

Para compilar:

```bash
cd CPU
make
# o manualmente
g++ -O3 -std=c++17 -march=native softmax_cpu.cpp -o softmax_cpu -lm
```

Para ejecutar una configuración específica (desde la carpeta `CPU/`):

```bash
sudo nvpmodel -q
./softmax_cpu --precision fp32 --n 100 --reps 10000 --measure function
./softmax_cpu --precision fp16 --n 100 --reps 10000 --measure function
./softmax_cpu --precision fp32 --n 100 --reps 10000 --measure program
./softmax_cpu --precision fp16 --n 100 --reps 10000 --measure program
./softmax_cpu --precision fp16 --sum-fp16 --n 1000 --reps 10000
```

Al utilizar el parámetro `--measure function`, se mide únicamente el tiempo de ejecución de las llamadas a softmax. Por otro lado, con `--measure program` se mide el tiempo total de ejecución, lo cual incluye la lectura del vector, la conversión a FP16, las llamadas a la función, el cálculo de la referencia y la verificación del resultado. En ambos escenarios, el programa realiza primero una ejecución de calentamiento (cuyo tiempo se descarta) seguida de 5 mediciones efectivas. Al finalizar, se reportan la mediana, el valor mínimo y el máximo de los tiempos obtenidos.

Para correr todas las configuraciones de una vez se usa:

```bash
cd CPU
./run_cpu.sh
```

Este script guarda la salida de cada configuración en la carpeta `CPU/logs/` y junta todas las mediciones en `CPU/logs/cpu_results.csv`. También guarda el modo de potencia (`nvpmodel -q`), la información del procesador (`lscpu`) y el registro de `tegrastats`, de donde se obtienen la frecuencia, la temperatura y los núcleos activos.

Cada medición imprime una línea con este formato:

```
precision, n, reps, measure, total_ms, us_por_llamada, error_abs, run, error_rel, suma
```

## Resultados

Los tiempos son por llamada, usando 10 000 repeticiones y 5 mediciones por configuración. Los datos crudos están en `CPU/logs/cpu_results.csv`.

### Solo la función

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

Para N ≥ 100 la diferencia entre el mínimo y el máximo es menor a 1 %. Con N = 10 la variación es mayor (hasta 4 % en FP32 y 16 % en una medición de FP16), porque cada medición dura apenas unos 5 ms y cualquier interrupción del sistema operativo pesa más.

### Programa completo

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

### Error contra FP64

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

## Respuestas

### 1. ¿Qué etapa domina el tiempo en cada plataforma y cómo cambia con N?

En la CPU domina el cálculo del softmax. El programa completo tarda apenas entre 0.25 % y 0.9 % más que la función sola, por lo que leer el vector, convertirlo y verificar el resultado representa menos del 1 % del tiempo total. El tiempo crece de forma lineal con N, a razón de unos 44 ns por elemento en FP32 (cerca de 41 ciclos a 921 MHz).

### 2. ¿Qué diferencia de error aparece entre FP16 y FP32? ¿Se debe al formato, la exponencial o la reducción?

En la CPU, FP16 tiene un error absoluto entre 8.6e-6 y 1.5e-4, frente a 3.5e-9 a 1.7e-8 en FP32, es decir, entre 1 400 y 17 000 veces mayor. El error absoluto baja con N porque las salidas son más pequeñas, pero el error relativo sube de 6.6e-4 con N = 10 a 1.2e-2 con N = 1000. Esto ocurre porque con más elementos muchas probabilidades quedan por debajo de 6.1e-5, donde FP16 pierde precisión.

En la versión FP16 de la CPU, la exponencial y la suma se calculan en *float*, por lo que el error proviene del formato: redondear la entrada, los valores intermedios y la salida a *half*. Para medir el efecto de la reducción se usó la variante `fp16sum`, donde la suma se acumula en *half*. Con N = 1000 el error absoluto aumenta casi 10 veces (8.5e-5 contra 8.6e-6) y la salida suma 1.011. Esto muestra que una reducción secuencial en FP16 sí es una fuente importante de error.

