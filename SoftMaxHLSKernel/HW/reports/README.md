# Resultados de síntesis HLS: kernel `softmax_hls` (KV260)

Estimaciones de **síntesis C (csynth)** de Vitis HLS 2024.1 (Build 5069499) para el dispositivo `xck26-sfvc784-2LV-c` (KV260). Son estimaciones previas a la implementación: no incluyen place & route.

## Variantes

Las variantes son acumulativas y salen del mismo archivo fuente, activadas con macros:

| Variante | Qué agrega | Macros activas |
|---|---|---|
| `v0_base` | Línea base | (ninguna) |
| `v1_pipeline` | `PIPELINE II=1` | `USE_PIPELINE` |
| `v2_partition` | `ARRAY_PARTITION cyclic factor=4` | `USE_PIPELINE`, `USE_PARTITION` |
| `v3_dataflow` | `DATAFLOW` | `USE_PIPELINE`, `USE_PARTITION`, `USE_DATAFLOW` |

FP32 es el valor por defecto del Makefile; FP16 se obtiene dejando `USE_FLOAT32` vacío (ver [`USO_MAKEFILE.md`](USO_MAKEFILE.md)).

## Cómo se obtuvo cada columna

| Columna | Origen |
|---|---|
| Período estimado | `softmax_hls_csynth.rpt` → *Performance Estimates* → *Timing* → *Summary* → `ap_clk`, columna *Estimated* |
| Fmax (MHz) | `1000 / período estimado (ns)`. Coincide con *Estimated Frequency* del módulo `softmax_hls` en `system_estimate_softmax_hls.xtxt` |
| LUT, FF, BRAM_18K, URAM, DSP | `softmax_hls_csynth.rpt` → *Utilization Estimates* → *Summary* → fila `Total` |
| II | `N/D` (ver [tabla de II](#ii-por-bucle)) |

**Latencia:** se omite a propósito. Depende de `N`, una variable que se pasa en tiempo de corrida, por lo que no es un valor fijo de síntesis. Además, los reportes muestran `?` para la latencia a nivel kernel en las 8 variantes.

**Recursos disponibles** (según los reportes): 117 120 LUT, 234 240 FF, 288 BRAM_18K, 64 URAM, 1 248 DSP.

## Tabla FP32

| Variante | II | Período (ns) | Fmax (MHz) | LUT | FF | BRAM_18K | URAM | DSP |
|---|---|---|---|---|---|---|---|---|
| v0_base | N/D | 3.650 | 273.97 | 4725 | 3221 | 10 | 0 | 9 |
| v1_pipeline | N/D | 4.019 | 248.82 | 5057 | 3433 | 10 | 0 | 9 |
| v2_partition | N/D | 4.544 | 220.07 | 5662 | 3801 | 16 | 0 | 9 |
| v3_dataflow | N/D | 4.544 | 220.07 | 5818 | 4799 | 16 | 0 | 9 |

## Tabla FP16

| Variante | II | Período (ns) | Fmax (MHz) | LUT | FF | BRAM_18K | URAM | DSP |
|---|---|---|---|---|---|---|---|---|
| v0_base | N/D | 3.650 | 273.97 | 3807 | 2479 | 5 | 0 | 6 |
| v1_pipeline | N/D | 4.031 | 248.08 | 3887 | 2582 | 5 | 0 | 6 |
| v2_partition | N/D | 4.031 | 248.08 | 4492 | 2950 | 14 | 0 | 6 |
| v3_dataflow | N/D | 4.031 | 248.08 | 4648 | 3947 | 14 | 0 | 6 |

## II por bucle

Los reportes **no entregan el II de ningún bucle**. Dónde se buscó:

- `softmax_hls_csynth.rpt`, sección `* Loop:` de *Performance Estimates → Latency → Detail*: dice `N/A` en las 8 variantes.
- `softmax_hls_csynth.rpt`, tabla `* Instance:` de la misma sección: solo aparecen *Interval* mín/máx del módulo y el tipo de pipeline, no el II del bucle.
- `system_estimate_softmax_hls.xtxt`, sección *Latency Information*: columna *Start Interval* del módulo, tampoco es el II del bucle.

| Bucle | Módulo en v0–v2 | Módulo en v3 | II (todas las variantes) | Tipo de pipeline según el reporte (v0–v2) |
|---|---|---|---|---|
| Bucle 18 | `softmax_hls_Pipeline_VITIS_LOOP_18_1` | `read_data` (el reporte no indica el sufijo del bucle) | N/D | `loop auto-rewind` |
| Bucle 31 | `softmax_hls_Pipeline_VITIS_LOOP_31_1` | `find_max_Pipeline_VITIS_LOOP_31_1` | N/D | `loop auto-rewind` |
| Bucle 45 | `softmax_hls_Pipeline_VITIS_LOOP_45_1` | `comp_exp_Pipeline_VITIS_LOOP_45_1` | N/D | `loop auto-rewind` |
| Bucle 58 | `softmax_hls_Pipeline_VITIS_LOOP_58_1` | `norm_exp_Pipeline_VITIS_LOOP_58_1` | N/D | `loop auto-rewind` |

Los cuatro bucles ya aparecen como `loop auto-rewind` desde v0, es decir, antes de agregar `PIPELINE II=1`. Para obtener el II real habría que consultarlo en la herramienta (por ejemplo, el *Schedule Viewer* de Vitis HLS), ya que no está en los archivos entregados.

## Análisis

Todo lo siguiente se apoya únicamente en las tablas anteriores y en el detalle por módulo de cada `csynth.rpt`. No se compara latencia (ver nota arriba).

### v0 → v1 (agrega `PIPELINE II=1`)

| Tipo | Fmax | LUT | FF | BRAM_18K | DSP |
|---|---|---|---|---|---|
| FP32 | 273.97 → 248.82 MHz (−9.2 %) | 4725 → 5057 (+332, +7.0 %) | 3221 → 3433 (+212, +6.6 %) | 10 → 10 | 9 → 9 |
| FP16 | 273.97 → 248.08 MHz (−9.5 %) | 3807 → 3887 (+80, +2.1 %) | 2479 → 2582 (+103, +4.2 %) | 5 → 5 | 6 → 6 |

- El período estimado sube (3.650 ns → 4.019 / 4.031 ns) en ambos tipos.
- El aumento de área se concentra en el módulo del bucle 45. FP32: LUT 1477 → 1850 y FF 1002 → 1248. FP16: LUT 520 → 600 y FF 374 → 477.
- En FP32, el módulo del bucle 31 baja de 283 a 242 LUT y de 146 a 112 FF.
- BRAM y DSP no cambian.

### v1 → v2 (agrega `ARRAY_PARTITION cyclic factor=4`)

| Tipo | Fmax | LUT | FF | BRAM_18K | DSP |
|---|---|---|---|---|---|
| FP32 | 248.82 → 220.07 MHz (−11.6 %) | 5057 → 5662 (+605, +12.0 %) | 3433 → 3801 (+368, +10.7 %) | 10 → 16 (+6) | 9 → 9 |
| FP16 | 248.08 → 248.08 MHz (sin cambio) | 3887 → 4492 (+605, +15.6 %) | 2582 → 2950 (+368, +14.3 %) | 5 → 14 (+9) | 6 → 6 |

- Cada uno de los tres buffers (`buf_max`, `buf_exp`, `e_buf`) pasa de 1 memoria de 1000 palabras a 4 bancos de 250 palabras. La memoria sube de 6 a 12 BRAM_18K en FP32 y de 3 a 12 en FP16: cada banco ocupa 1 BRAM_18K sin importar si es de 16 o 32 bits.
- Los incrementos absolutos de LUT (+605) y FF (+368) son idénticos en FP32 y FP16. En ambos, el multiplexor sube de 1059 a 1392 LUT (+333).
- Solo FP32 pierde Fmax en este paso.

### v2 → v3 (agrega `DATAFLOW`)

| Tipo | Fmax | LUT | FF | BRAM_18K | DSP |
|---|---|---|---|---|---|
| FP32 | 220.07 → 220.07 MHz (sin cambio) | 5662 → 5818 (+156, +2.8 %) | 3801 → 4799 (+998, +26.3 %) | 16 → 16 | 9 → 9 |
| FP16 | 248.08 → 248.08 MHz (sin cambio) | 4492 → 4648 (+156, +3.5 %) | 2950 → 3947 (+997, +33.8 %) | 14 → 14 | 6 → 6 |

- El diseño se divide en `entry_proc`, `read_data`, `find_max`, `comp_exp` y `norm_exp`, y el tipo de pipeline del kernel pasa a `dataflow`.
- El aumento de FF se reparte entre seis FIFOs (594 FF y 401 LUT en total, iguales en ambos tipos) y el módulo `read_data` (572 FF / 352 LUT en FP32; 556 FF / 352 LUT en FP16). Los registros propios de la FSM bajan de 345 (FP32) / 331 (FP16) a 20, y el multiplexor de 1392 a 153 LUT.
- BRAM, DSP y Fmax no cambian.

### FP32 vs FP16

| Variante | Fmax FP32 → FP16 | LUT | FF | BRAM_18K | DSP |
|---|---|---|---|---|---|
| v0 | 273.97 → 273.97 | −19.4 % | −23.0 % | 10 → 5 | 9 → 6 |
| v1 | 248.82 → 248.08 | −23.1 % | −24.8 % | 10 → 5 | 9 → 6 |
| v2 | 220.07 → 248.08 (+12.7 %) | −20.7 % | −22.4 % | 16 → 14 | 9 → 6 |
| v3 | 220.07 → 248.08 (+12.7 %) | −20.1 % | −17.8 % | 16 → 14 | 9 → 6 |

- FP16 usa 6 DSP en todas las variantes, frente a 9 de FP32.
- FP16 reduce LUT y FF entre un 18 % y un 25 % en todas las variantes.
- El ahorro de BRAM desaparece casi por completo con `ARRAY_PARTITION`: de 5 BRAM menos en v0/v1 a solo 2 menos en v2/v3, porque la memoria de datos queda en 12 BRAM_18K en ambos tipos.
- FP16 mantiene un período estimado de 4.031 ns desde v1 hasta v3, mientras que FP32 llega a 4.544 ns desde v2.

## Notas y limitaciones

1. **Reloj objetivo distinto al del Makefile.** Los reportes usan un objetivo de 5.00 ns (200 MHz); el `.xtxt` indica `Target Clock: 199.998 MHz`. El Makefile define `KERNEL_FREQ := 250000000` (250 MHz, 4 ns). Los reportes parecen haberse generado con otro reloj, o el Makefile cambió después. Conviene confirmarlo antes de citar estos resultados.
2. **Incertidumbre de timing.** Todos los reportes indican 1.35 ns de incertidumbre. Con un objetivo de 5.00 ns, el período útil sería 3.65 ns: v0 (3.650 ns) queda justo en ese límite y el resto de las variantes (4.019 a 4.544 ns) lo supera. Los Fmax de la tabla son `1000 / período estimado`, sin descontar la incertidumbre. No se encontró ningún mensaje de violación en los `.rpt`.
3. **DSP en los `.xtxt`.** La sección *Area Information* de `system_estimate_softmax_hls.xtxt` muestra DSP = 0 en todas las variantes, mientras que `csynth.rpt` muestra 6 (FP16) o 9 (FP32). LUT, FF y BRAM coinciden entre ambos archivos. Esta tabla usa los valores de `csynth.rpt`.
4. **Datos coincidentes entre FP32 y FP16.** Los archivos son distintos entre sí (verificado con checksums) y los buffers de FP16 son de 16 bits. Las coincidencias observadas (Fmax en v0, multiplexor de 1059 LUT en v0/v1, 12 BRAM_18K de memoria tras la partición) son consistentes con lo descrito en el análisis.
5. **Alcance.** Son estimaciones de síntesis; los valores finales tras implementación pueden diferir.