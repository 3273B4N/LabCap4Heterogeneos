# Uso del Makefile: kernel `softmax_hls` (KV260)

Guía para compilar el kernel `softmax_hls` en sus variantes `v0`–`v3` y en FP32/FP16. Los resultados obtenidos se resumen en [`RESULTADOS.md`](RESULTADOS.md).

## 1. Cargar el entorno de Vitis 2024.1

```bash
source /usr/local/Xilinx/Vitis/2024.1/settings64.sh
```

Esa es la ruta de instalación por defecto. Si Vitis está en otra ubicación, buscar el script en la computadora donde se esté corriendo:

```bash
find / -name settings64.sh -path '*Vitis*' 2>/dev/null
```

Verificar que quedó cargado:

```bash
v++ --version
```

Debe imprimir `v++ v2024.1`. Hay que repetir el `source` en cada terminal nueva.

## 2. Requisitos para `make tmp.hw/softmax_hls.xo (solo reportes) o make (reportes + bitstream)`

| Requisito | Detalle (según el Makefile) |
|---|---|
| Fuente | `softmax_hls.cpp` en el mismo directorio que el Makefile (`HLS_FILES := softmax_hls.cpp`) |
| Plataforma | `../platform/kv260_custom.xpfm` relativo al Makefile (`PLATFORM := $(ROOT_DIR)/../platform/kv260_custom.xpfm`) |
| Herramienta | `v++` en el `PATH` (lo da `settings64.sh`) |

Para el flujo completo (`make` / `make build`) hacen falta además `emconfigutil` y `bootgen`, y los archivos `../../shell.json` y `../../pl.dtbo`. Nada de eso se necesita para generar solo el `.xo`.

### Flags de v++

| Variable | Valor | Qué hace |
|---|---|---|
| `VPP_PFLAGS` | (vacía) | Flags extra para la compilación |
| `VPP_LDFLAGS` | `--vivado.impl.strategies=Area_Explore` | Flags solo del link (`v++ -l`) |
| `VPP_FLAGS` | `--save-temps --jobs 3` (+ los `-D...` de abajo) | Flags comunes a compilación y link. Conserva temporales y usa 3 jobs |
| `KERNEL_FREQ` | `250000000` | Frecuencia del reloj del kernel en Hz (`--clock.defaultFreqHz`). El comentario del Makefile dice "200 MHz", pero el valor es 250 MHz |

### Macros de diseño

Cada una se activa con `ifdef` (basta con que la variable tenga cualquier valor no vacío) y agrega un `-D` a `VPP_FLAGS`:

| Variable | Valor por defecto | Agrega | Efecto |
|---|---|---|---|
| `USE_FLOAT32` | `1` | `-DUSE_FLOAT32` | FP32. Con la variable vacía, se compila en FP16 |
| `USE_PIPELINE` | (no definida) | `-DUSE_PIPELINE` | `PIPELINE II=1` |
| `USE_PARTITION` | (no definida) | `-DUSE_PARTITION` | `ARRAY_PARTITION cyclic factor=4` |
| `USE_DATAFLOW` | (no definida) | `-DUSE_DATAFLOW` | `DATAFLOW` |

Las variables pasadas en la línea de comandos (`make VAR=valor`) tienen prioridad sobre las asignaciones del Makefile, por eso `make USE_FLOAT32= ...` (vacía) desactiva FP32.

### Reglas

| Regla | Qué hace |
|---|---|
| `tmp.hw/%.xo` | Compila `%.cpp` con `v++ -c` y genera el `.xo` (aquí `tmp.hw/softmax_hls.xo`) |
| `build` / `all` | Flujo completo: link, bitstream y empaquetado |
| `clean` | Borra `*.rpt`, `*.csv`, `system_estimate.xtxt`, `.Xil`, `emconfig.json`, `*.log`, entre otros. **No borra `tmp.hw`** |
| `cleanall` | Hace `clean` y además borra `build_dir*`, `sd_card*`, `package.*`, `tmp.*`, `_x*`, `*.xclbin`, `pl*`, entre otros |

## 4. Compilar cada variante

El comando es siempre el mismo objetivo, `make tmp.hw/softmax_hls.xo`, con las macros pasadas por la línea de comandos. Como las variantes son acumulativas, cada una activa también las macros de la anterior.

### FP32 (valor por defecto)

| Variante | Comando |
|---|---|
| `v0_base` | `make tmp.hw/softmax_hls.xo` |
| `v1_pipeline` | `make USE_PIPELINE=1 tmp.hw/softmax_hls.xo` |
| `v2_partition` | `make USE_PIPELINE=1 USE_PARTITION=1 tmp.hw/softmax_hls.xo` |
| `v3_dataflow` | `make USE_PIPELINE=1 USE_PARTITION=1 USE_DATAFLOW=1 tmp.hw/softmax_hls.xo` |

### FP16 (`USE_FLOAT32` vacío)

| Variante | Comando |
|---|---|
| `v0_base` | `make USE_FLOAT32= tmp.hw/softmax_hls.xo` |
| `v1_pipeline` | `make USE_FLOAT32= USE_PIPELINE=1 tmp.hw/softmax_hls.xo` |
| `v2_partition` | `make USE_FLOAT32= USE_PIPELINE=1 USE_PARTITION=1 tmp.hw/softmax_hls.xo` |
| `v3_dataflow` | `make USE_FLOAT32= USE_PIPELINE=1 USE_PARTITION=1 USE_DATAFLOW=1 tmp.hw/softmax_hls.xo` |


## 5. Dónde queda el reporte

**N/D.** El Makefile no define dónde queda el reporte de síntesis: no usa `--report_dir` ni menciona la ruta del `*_csynth.rpt` (se buscó en todo el archivo). Lo único relacionado es la regla `clean`, que borra `*.rpt` y `system_estimate.xtxt` del directorio actual. Como v++ se invoca con `--temp_dir ./tmp.hw` y `--save-temps`, lo esperable es que los reportes de Vitis HLS queden dentro de `tmp.hw`; la ruta exacta no se pudo verificar con los archivos entregados. Para ubicarlos:

```bash
find tmp.hw -name 'softmax_hls_csynth.rpt'
find . -name 'system_estimate_softmax_hls.xtxt' -not -path './reports/*'
```

Los reportes de este proyecto se guardan en `reports/<variante>_<tipo>/`, por ejemplo `reports/v1_pipeline_fp32/`, con estos dos archivos:

- `softmax_hls_csynth.rpt`
- `system_estimate_softmax_hls.xtxt`

Ejemplo para copiarlos al terminar una compilación:

```bash
V=v1_pipeline_fp32
mkdir -p reports/$V
cp "$(find tmp.hw -name 'softmax_hls_csynth.rpt' | head -n 1)" reports/$V/
cp "$(find . -name 'system_estimate_softmax_hls.xtxt' -not -path './reports/*' | head -n 1)" reports/$V/
```

Copiar los reportes **antes** de correr `make clean` o `make cleanall`, porque ambos pueden borrarlos.

## 6. Advertencia: borrar `tmp.hw` al cambiar de variante o de tipo de dato
> **Antes de compilar otra variante, o el mismo diseño en otro tipo de dato, borrar `tmp.hw`:**

Flujo recomendado por variante (solo para los reportes, para el flujo completo es lo mismo pero con solo make):

```bash
rm -rf tmp.hw
make <macros> tmp.hw/softmax_hls.xo
# copiar los reportes a reports/<variante>_<tipo>/
```
