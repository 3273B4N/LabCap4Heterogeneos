# Ejercicio C: FPGA (KV260)

Kernel softmax sintetizado con Vitis HLS y ejecutado en la **Kria KV260** (K26 SOM) con XRT. Esta carpeta contiene el kernel, los binarios listos para cargar, el programa host y los scripts de medición de energía.

Variantes disponibles en `binaries/` (cada una trae `softmax_hls.xclbin`, `softmax_hls.bit.bin`, `softmax_hls.dtbo` y `shell.json`):

| Variante | FP32 (200 MHz) | FP16 (150 MHz) |
|---|---|---|
| v0 base | `v0_base_fp32_200MHz` | `v0_base_fp16_150MHz` |
| v1 pipeline | `v1_pipeline_fp32_200MHz` | `v1_pipeline_fp16_150MHz` |
| v2 partición | `v2_partition_fp32_200MHz` | `v2_partition_fp16_150MHz` |
| v3 dataflow | `v3_dataflow_fp32_200MHz` | `v3_dataflow_fp16_150MHz` |


## Requisitos

Todo se ejecuta **en la KV260**, con la imagen Ubuntu de Kria (no PYNQ). XRT no está instalado en la Jetson, por lo que el host no compila allí.

```bash
which xmutil
xbutil --version
ls /usr/include/xrt
```

## Compilación del host

Desde la carpeta `SW/`:

```bash
cd FPGA/SW
make
```

`make` genera dos ejecutables a partir de `bench_softmax.cpp`:

- `bench_fp32`: precisión FP32 (`float`).
- `bench_fp16`: precisión FP16 (`__fp16`, compilado con `-DFP16`).

Ambos se compilan con `g++ -std=c++17 -O2`, usando `-I/usr/include/xrt` y enlazando con `-lxrt_coreutil -lpthread`. Para borrarlos: `make clean`.

Para ver la ayuda del host:

```bash
./bench_fp32
```

## Carga de la app

Las apps ya están registradas en la KV260 con el nombre de la carpeta de `binaries/`. Solo puede haber una app cargada a la vez.

```bash
sudo xmutil listapps
sudo xmutil unloadapp                     # devuelve -1 si no había ninguna cargada
sudo xmutil loadapp v3_dataflow_fp32_200MHz
```

Para cambiar de precisión, primero `unloadapp` y luego `loadapp` de la otra:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp v3_dataflow_fp16_150MHz
```

Si `loadapp` responde `load Error: -1`, normalmente la app ya está cargada o el nombre no coincide con el de `listapps`. Para revisar el kernel, sus argumentos y el reloj:

```bash
xclbinutil --info --input ../binaries/v3_dataflow_fp32_200MHz/softmax_hls.xclbin
```

La frecuencia debe mantenerse constante en todas las mediciones (FP32 a 200 MHz y FP16 a 150 MHz).

## Ejecución

Sintaxis:

```
./bench_fp32 <xclbin> <parts 1|4> <variante> <MHz> <csv> [--reps M] [--warmup W] [--energy S] [--energy-mode full|kernel|e2e]
```

| Argumento | Descripción |
|---|---|
| `xclbin` | Ruta al `softmax_hls.xclbin` de la variante cargada |
| `parts` | Número de buffers (1 o 4); debe coincidir con la síntesis del kernel |
| `variante` | Etiqueta que se escribe en el CSV |
| `MHz` | Frecuencia del kernel, solo como etiqueta |
| `csv` | Archivo de salida con los tiempos por repetición |
| `--reps`, `--warmup` | Repeticiones medidas (mínimo 5, por defecto 30) y de calentamiento (por defecto 5) |
| `--energy S` | Repite llamadas durante `S` segundos y registra inicio y fin para integrar la energía |
| `--energy-mode` | `full`, `kernel` o `e2e` (ver abajo) |

Modos de medición:

- `full`: llamada completa, con relleno del buffer y lectura y conversión en el host.
- `kernel`: solo lanzamiento y espera del kernel, con los buffers ya cargados.
- `e2e`: sincronización a dispositivo + kernel + sincronización de vuelta, sin relleno ni conversión. Equivale al end-to-end de la GPU.

### Corrida de prueba (FP32)

Desde `FPGA/SW`, con la app cargada:

```bash
sudo xmutil loadapp v3_dataflow_fp32_200MHz
./bench_fp32 ../binaries/v3_dataflow_fp32_200MHz/softmax_hls.xclbin \
    1 v3_dataflow_fp32 200 /tmp/prueba_fp32.csv --reps 10 --warmup 2
```

### Corrida de prueba (FP16)

```bash
sudo xmutil unloadapp
sudo xmutil loadapp v3_dataflow_fp16_150MHz
./bench_fp16 ../binaries/v3_dataflow_fp16_150MHz/softmax_hls.xclbin \
    1 v3_dataflow_fp16 150 /tmp/prueba_fp16.csv --reps 10 --warmup 2
```

El programa también escribe `input_seed42.txt` (vector de entrada, semilla 42, uniforme en [-4, 4]) en la carpeta desde donde se ejecuta. Está en el `.gitignore`.

### Salida esperada

Una línea por cada N (10, 100, 500, 1000) con la mediana, mínimo y máximo del tiempo total, el desglose por etapa y la verificación:

```
== v3_dataflow_fp32 fp32 200 MHz parts=1 warmup=2 reps=10 ==
N=  10 total med/min/max = 59.0/54.5/64.2 us | fill 0.2 | h2d 3.6 | launch+wait 49.2 | d2h 3.8 | gather 1.3 | suma=1.000000 err_abs=1.56e-08 argmax=1
N= 100 total med/min/max = 72.1/71.1/75.3 us | fill 1.2 | h2d 3.5 | launch+wait 51.1 | d2h 3.8 | gather 12.4 | suma=1.000000 err_abs=4.18e-09 argmax=1
N= 500 total med/min/max = 141.1/138.9/155.5 us | fill 5.8 | h2d 3.5 | launch+wait 67.0 | d2h 3.8 | gather 60.7 | suma=1.000000 err_abs=1.23e-09 argmax=1
N=1000 total med/min/max = 224.0/223.4/226.6 us | fill 11.4 | h2d 3.7 | launch+wait 84.0 | d2h 3.7 | gather 121.3 | suma=1.000000 err_abs=2.15e-09 argmax=1
```

Verificaciones: la suma de la salida debe ser 1, el índice del máximo debe coincidir con la referencia (`argmax=1`) y el error absoluto se calcula contra FP64.

### Problemas frecuentes

- `el kernel NO termino (timeout)`: el kernel se colgó. Recuperar con `sudo xmutil unloadapp && sudo xmutil loadapp <app>`.
- `excepcion: ...`: `.xclbin` equivocado o app no cargada.
- `argmax=0` o suma lejos de 1: `parts`, precisión o `.xclbin` no coinciden con la app cargada.
- `missing separator` al hacer `make`: las recetas del Makefile deben empezar con tab.

