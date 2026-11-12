#!/bin/bash
# Corre todas las configuraciones del ejercicio A y guarda la salida en logs/

mkdir -p logs

# estado del sistema: modo de potencia y procesador
sudo nvpmodel -q > logs/nvpmodel.txt
lscpu > logs/lscpu.txt

# tegrastats en segundo plano para tener frecuencia, temperatura y nucleos activos
tegrastats --interval 1000 > logs/tegrastats.txt &
TEGRA_PID=$!

for measure in function program; do
  for precision in fp32 fp16; do
    for n in 10 100 500 1000; do
      ./softmax_cpu --precision $precision --n $n --reps 10000 --measure $measure \
        | tee logs/${precision}_${measure}_n${n}.txt
    done
  done
  # fp16 con la suma en half (para la pregunta 2)
  for n in 10 100 500 1000; do
    ./softmax_cpu --precision fp16 --sum-fp16 --n $n --reps 10000 --measure $measure \
      | tee logs/fp16sum_${measure}_n${n}.txt
  done
done

kill $TEGRA_PID

# datos crudos en un solo CSV
echo "precision, n, reps, measure, total_ms, us_por_llamada, error_abs, run, error_rel, suma" > logs/cpu_results.csv
grep -h "^fp" logs/*_n*.txt >> logs/cpu_results.csv
