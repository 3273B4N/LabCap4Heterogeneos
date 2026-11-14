#!/bin/bash
# Medicion de energia de las 8 variantes en la KV260 (INA260: potencia a nivel de placa).
# uso: sudo ./run_energy.sh [REPS=3] [SEG_POR_N=10]
# Por variante (orden aleatorio): carga la app -> idle -> REPS x {full, kernel} -> idle.
# Opcionales por entorno: IDLE=30 (segundos de idle), FW=/lib/firmware/xilinx,
#   MODES="full kernel" (modos de bench_softmax: full | kernel | e2e), ONLY="v3_dataflow_fp32_200MHz ..." (solo esas variantes)
# Necesita en esta carpeta: bench_fp16 bench_fp32 power_sampler.py energy.py energy_agregar.py y las carpetas v*/
REPS=${1:-3}
SEC=${2:-10}
IDLE=${IDLE:-30}
FW=${FW:-/lib/firmware/xilinx}
MODES=${MODES:-"full kernel"}
ONLY=${ONLY:-}
HERE=$(pwd)

if [ "$(id -u)" -ne 0 ] && [ -z "$NO_ROOT_CHECK" ]; then echo "correr con sudo"; exit 1; fi
for f in bench_fp16 bench_fp32 power_sampler.py energy.py energy_agregar.py; do
  [ -e "./$f" ] || { echo "falta ./$f"; exit 1; }
done

OUT=energy_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUT"
if [ -n "$ONLY" ]; then VARS=$(for v in $ONLY; do [ -d "$v" ] && echo "$v"; done | shuf); else VARS=$(ls -d v*/ | tr -d / | shuf); fi
NV=$(echo "$VARS" | wc -l)
NM=$(echo $MODES | wc -w)
echo "orden: $(echo $VARS)" | tee "$OUT/orden.txt"
echo "estimado: ~$(( NV * (2*IDLE + REPS*NM*(4*SEC+6)) / 60 )) min" | tee -a "$OUT/orden.txt"

for d in $VARS; do
  prec=$(echo "$d" | grep -oi 'fp16\|fp32' | tr 'A-Z' 'a-z')
  mhz=$(echo "$d" | grep -oi '[0-9]*mhz' | sed 's/[mM][hH][zZ]//')
  var=$(echo "$d" | sed 's/_[fF][pP][0-9]*_.*//')
  nargs=$(grep -a -o '"name":"[A-Za-z_0-9]*","address_qualifier"' "$d/softmax_hls.xclbin" | sort -u | wc -l)
  if   [ "$nargs" -eq 9 ]; then P=4
  elif [ "$nargs" -eq 3 ]; then P=1
  else echo "$d: $nargs argumentos, salto" | tee -a "$OUT/log.txt"; continue; fi
  if [ -z "$mhz" ] || [ -z "$prec" ]; then echo "$d: no pude leer reloj o precision, salto" | tee -a "$OUT/log.txt"; continue; fi
  X="$FW/$d/softmax_hls.xclbin"
  [ -d "$FW/$d" ] || cp -r "$d" "$FW/"
  xmutil unloadapp > /dev/null 2>&1
  if ! xmutil loadapp "$d" 2>&1 | grep -q "loaded to slot"; then echo "NO CARGO $d" | tee -a "$OUT/log.txt"; continue; fi
  sleep 10
  mkdir -p "$OUT/$d"
  echo "=== $d (prec=$prec, ${mhz} MHz, parts=$P) ===" | tee -a "$OUT/log.txt"

  python3 power_sampler.py "$OUT/$d/idle_a.csv" 0.01 "$IDLE"
  for r in $(seq 1 "$REPS"); do
    for mode in $MODES; do
      tag="${mode}_r$r"
      python3 power_sampler.py "$OUT/$d/pw_$tag.csv" 0.01 &
      SP=$!
      sleep 2
      if ! "./bench_$prec" "$X" "$P" "$var" "$mhz" "$OUT/$d/bench_$tag.csv" --energy "$SEC" --energy-mode "$mode" > "$OUT/$d/bench_$tag.log" 2>&1; then
        echo "FALLO bench en $d $tag (recargo la app)" | tee -a "$OUT/log.txt"
        xmutil unloadapp > /dev/null 2>&1; xmutil loadapp "$d" > /dev/null 2>&1; sleep 10
      fi
      sleep 1
      kill "$SP" 2> /dev/null; wait "$SP" 2> /dev/null
    done
  done
  python3 power_sampler.py "$OUT/$d/idle_b.csv" 0.01 "$IDLE"

  ( cat "$OUT/$d/idle_a.csv"; tail -n +2 "$OUT/$d/idle_b.csv" ) > "$OUT/$d/idle.csv"
  ma=$(awk -F, 'NR>1{s+=$2;n++}END{if(n)printf "%.4f", s/n/1e6}' "$OUT/$d/idle_a.csv")
  mb=$(awk -F, 'NR>1{s+=$2;n++}END{if(n)printf "%.4f", s/n/1e6}' "$OUT/$d/idle_b.csv")
  echo "$d idle_inicio=${ma}W idle_fin=${mb}W" | tee -a "$OUT/idle_deriva.txt"
  for r in $(seq 1 "$REPS"); do
    for mode in $MODES; do
      tag="${mode}_r$r"
      ( cd "$OUT/$d" && python3 "$HERE/energy.py" "pw_$tag.csv" "bench_$tag.csv_energy.csv" idle.csv > "energy_$tag.txt" 2>&1 \
        && mv energy_resumen.csv "energy_resumen_$tag.csv" )
    done
  done
done
xmutil unloadapp > /dev/null 2>&1
python3 energy_agregar.py "$OUT"
echo "listo: $OUT"