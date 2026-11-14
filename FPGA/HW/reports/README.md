# Resultados de síntesis y pruebas: kernel `softmax_hls` (KV260)

Resultados de **síntesis C (csynth)** de Vitis HLS 2024.1 para el kernel `softmax_hls` (dispositivo `xck26-sfvc784-2LV-c`, KV260), junto con los resultados del testbench, de la cosimulación y del timing post-implementación de cada variante.
## Frecuencia objetivo

| Tipo | Frecuencia objetivo | Target Clock del reporte | Incertidumbre del reporte | Variantes con ese valor |
|---|---|---|---|---|
| FP32 | 200 MHz | 5.00 ns | 1.35 ns | las 4 (`reports/*_fp32/softmax_hls_csynth.rpt`) |
| FP16 | 150 MHz | 6.67 ns | 1.80 ns | las 4 (`reports/*_fp16/softmax_hls_csynth.rpt`) |

- FP32 se sintetizó a 200 MHz y FP16 a 150 MHz. FP16 se bajó a 150 MHz porque a 200 MHz no cerraba timing. 


## Recursos y frecuencia

Estimaciones de csynth.

### FP32

| Variante | Frec. objetivo (MHz) | II (25 / 44 / 75 / 99) | Fmax (MHz) | LUT | FF | BRAM_18K | URAM | DSP |
|---|---|---|---|---|---|---|---|---|
| v0_base | 200 | 1 / 2 / 9 / 1 | 273.97 | 4819 | 3261 | 10 | 0 | 9 |
| v1_pipeline | 200 | 1 / 1 / 5 / 1 | 248.82 | 5151 | 3473 | 10 | 0 | 9 |
| v2_partition | 200 | 1 / 1 / 5 / 4 | 248.82 | 7722 | 5399 | 16 | 0 | 9 |
| v3_dataflow | 200 | 1 / 1 / 5 / 4 | 248.82 | 7701 | 6330 | 16 | 0 | 9 |

### FP16

| Variante | Frec. objetivo (MHz) | II (25 / 44 / 75 / 99) | Fmax (MHz) | LUT | FF | BRAM_18K | URAM | DSP |
|---|---|---|---|---|---|---|---|---|
| v0_base | 150 | 1 / 1 / 3 / 1 | 205.38 | 3910 | 2582 | 5 | 0 | 6 |
| v1_pipeline | 150 | 1 / 1 / 1 / 1 | 191.06 | 4013 | 2655 | 5 | 0 | 6 |
| v2_partition | 150 | 1 / 1 / 1 / 4 | 191.06 | 6958 | 4703 | 14 | 0 | 18 |
| v3_dataflow | 150 | 1 / 1 / 1 / 4 | 191.06 | 6985 | 5650 | 14 | 0 | 18 |

Períodos estimados (ns) de los que sale Fmax: FP32 v0 3.650, v1 4.019, v2 4.019, v3 4.019; FP16 v0 4.869, v1 5.234, v2 5.234, v3 5.234.

## II por bucle

Formato de cada celda: **II logrado (II objetivo)**. Todas las variantes reportan los cuatro bucles con tipo de pipeline `yes` y un II objetivo de 1. 

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

- En v0–v2 cada bucle tiene su propio reporte (`softmax_hls_Pipeline_VITIS_LOOP_<n>_csynth.rpt`). En v3 los bucles quedan dentro de los módulos `read_data`, `find_max`, `comp_exp` y `norm_exp`; los de 44, 75 y 99 conservan reporte con el mismo nombre de bucle y el 25 sale de `read_data_csynth.rpt`.

- Iteraciones (*Trip Count*): en v0 y v1 los bucles 44, 75 y 99 muestran `0 ~ 1023` y el 25 muestra `?`; en v2 y v3 los cuatro muestran `?`.

## Pruebas: testbench y cosimulación

### Cómo se hicieron

El testbench (`tb.cpp`) llama a `softmax_hls(x, y, n)` con cuatro tamaños, `N = 10, 100, 500, 1000`, y compara contra una referencia en `double`:

- **Entrada:** `x[i] = (i % 97) / 10.0 - 4.8` convertido al tipo del kernel (`float` con `USE_FLOAT32`, `half` en caso contrario).
- **Referencia:** softmax en `double` con resta del máximo, calculada a partir de los mismos valores de entrada ya convertidos.
- **Métricas por tamaño:** error máximo absoluto entre `y[i]` y la referencia, y la suma de las salidas.
- **Tolerancia:** `1e-5` en FP32 y `1e-2` en FP16. Si el error máximo supera la tolerancia para algún tamaño, el testbench imprime `RESULTADO: FALLO` y devuelve 1; si no, imprime `RESULTADO: OK`.
- **Evidencia:** `softmax_hls_csim.log` (simulación C), `tb_resultados.txt` (salida del testbench) y `softmax_hls_cosim.rpt` (cosimulación RTL).

Resultado de las 8 variantes: csim **PASS** (`CSim done with 0 errors`, `RESULTADO: OK`) y cosim **PASS** en Verilog. En `softmax_hls_cosim.rpt` la columna VHDL aparece como `NA`.

Cada celda de las tablas siguientes es `error máximo / suma` tomada de `softmax_hls_csim.log`.

### FP32 (tolerancia 1e-5)

| Variante | csim | cosim (Verilog) | N=10 | N=100 | N=500 | N=1000 |
|---|---|---|---|---|---|---|
| v0_base | PASS | PASS | 5.469e-09 / 1.00000 | 8.118e-09 / 1.00000 | 8.285e-09 / 1.00000 | 2.325e-09 / 1.00000 |
| v1_pipeline | PASS | PASS | 5.469e-09 / 1.00000 | 8.118e-09 / 1.00000 | 8.285e-09 / 1.00000 | 2.325e-09 / 1.00000 |
| v2_partition | PASS | PASS | 5.469e-09 / 1.00000 | 6.783e-09 / 1.00000 | 1.028e-09 / 1.00000 | 4.688e-10 / 1.00000 |
| v3_dataflow | PASS | PASS | 5.469e-09 / 1.00000 | 6.783e-09 / 1.00000 | 1.028e-09 / 1.00000 | 4.688e-10 / 1.00000 |

### FP16 (tolerancia 1e-2)

| Variante | csim | cosim (Verilog) | N=10 | N=100 | N=500 | N=1000 |
|---|---|---|---|---|---|---|
| v0_base | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 7.385e-05 / 1.00065 | 8.341e-05 / 1.00235 |
| v1_pipeline | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 7.385e-05 / 1.00065 | 8.341e-05 / 1.00235 |
| v2_partition | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 5.762e-05 / 0.99711 | 5.801e-05 / 0.99518 |
| v3_dataflow | PASS | PASS | 5.777e-05 / 0.99991 | 5.819e-05 / 0.99950 | 5.762e-05 / 0.99711 | 5.801e-05 / 0.99518 |

## Timing post-implementación


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

- El WNS/WHS es del diseño completo (`system_wrapper`), no solo del reloj del kernel.


## Análisis de resultados (TODO)



