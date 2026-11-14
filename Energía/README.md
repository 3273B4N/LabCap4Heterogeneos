# LabCap4Heterogeneos

## Ejercicio D: Energía

Para determinar el consumo energético, la energía de cada llamada a la función
se calculó multiplicando la potencia por el tiempo de ejecución. Asimismo, la
energía por elemento se obtuvo dividiendo la energía de cada llamada entre la
cantidad de elementos (N):

    Energía (µJ) = Potencia (W) × Tiempo (µs)
    J/elem.      = Energía (J) / N

Como la NVIDIA Jetson Nano usada (2GB) no cuenta con un sensor de potencia
integrado, la herramienta `tegrastats` no reporta el consumo real de la placa.
Por indicación del profesor, se asume un consumo constante de **5 W** tanto
para la CPU como para la GPU, valor que corresponde al límite del perfil de
energía en el que se ejecutaron las pruebas (`nvpmodel` modo 1).

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

---

## Compilación (host)

Requisitos: `g++` con C++17, XRT (`/usr/include/xrt`, `libxrt_coreutil`),
`pthread`.

    cd ~/grupo2/lab4/binaries

    # Verificar que el código incluye el modo end-to-end
    grep -c e2e bench_softmax.cpp        # debe dar > 0

    # Compilar FP16
    g++ -std=c++17 -O2 -DFP16 bench_softmax.cpp -o bench_fp16 \
        -I/usr/include/xrt -lxrt_coreutil -lpthread

    # Compilar FP32
    g++ -std=c++17 -O2 bench_softmax.cpp -o bench_fp32 \
        -I/usr/include/xrt -lxrt_coreutil -lpthread

---

##  KV260

Esta sección documenta la **potencia y energía de la KV260** (Kria K26 SOM,
flujo Vitis/Kria con XRT), que es donde el repo dedica la zona de potencia.

### Paquete entregado y archivos

El paquete entregado es del **flujo Vitis/Kria (XRT)**.

- **`.xclbin`**: bitstream empaquetado con metadata del kernel (nombre,
  argumentos, reloj). Es lo que abre XRT desde el host.
- **`.dtbo`**: overlay de device tree que el sistema necesita para cargar el
  diseño.
- **`.bin`**: depende de quién lo armó. En apps de Kria suele ser el bitstream
  en formato `.bit.bin` o el mismo `.xclbin` con otra extensión. Comparar con
  `md5sum *.bin *.xclbin` y `file *.bin`.
- **`shell.json`**: archivo de texto corto que describe el tipo de app.
  

### Requisitos

Requiere la **imagen Ubuntu de Kria**, no la de PYNQ:

    which xmutil
    xbutil --version

### Carga de la app

    sudo mkdir -p /lib/firmware/xilinx/softmax_fp32 
    sudo cp softmax_fp32.bit.bin softmax_fp32.dtbo shell.json \ 
        /lib/firmware/xilinx/softmax_fp32/

    sudo xmutil listapps
    sudo xmutil unloadapp
    sudo xmutil loadapp softmax_fp32

    xclbinutil --info --input softmax_fp32.xclbin

Los nombres exactos dentro de la carpeta importan (convención Kria:
`<app>.bit.bin` y `<app>.dtbo`). `xclbinutil --info` muestra el nombre del
kernel, sus argumentos y el reloj; anotar la frecuencia, que debe ser constante
en todas las mediciones (FP32 a 200 MHz y FP16 a 150 MHz en este laboratorio).

### Barrido energético

    sudo env IDLE=30 MODES="e2e" \
      ONLY="v3_dataflow_fp16_150MHz v3_dataflow_fp32_200MHz" \
      nohup ./run_energy.sh 5 10 > run_e2e.out 2>&1 &

    pgrep -af run_energy.sh

### Declaración del punto de medición

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

### Comparación end-to-end (KV260 vs GPU)

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
