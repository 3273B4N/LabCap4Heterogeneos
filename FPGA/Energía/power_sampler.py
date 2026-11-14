#!/usr/bin/env python3
"""Muestrea el INA260 (ina260_u14) a periodo fijo y, cada ~1 s, las temperaturas del AMS.
uso: python3 power_sampler.py <salida.csv> <periodo_s> [duracion_s]
Sin duracion corre hasta Ctrl+C o kill. Busca los sensores por nombre (el numero de hwmon puede cambiar).
Columnas: t_epoch_ns, power_uW, volt_mV, curr_mA, temp*_mC
Los timestamps son del reloj del sistema, igual que los intervalos que escribe bench_softmax (*_energy.csv)."""
import glob, os, sys, time

ROOT = os.environ.get("HWMON_ROOT", "/sys/class/hwmon")

def find(name):
    for d in sorted(glob.glob(os.path.join(ROOT, "hwmon*"))):
        try:
            with open(os.path.join(d, "name")) as f:
                if f.read().strip() == name:
                    return d
        except OSError:
            pass
    return None

def rd(p):
    with open(p) as f:
        return f.read().strip()

if len(sys.argv) < 3:
    sys.exit(__doc__)
ina, ams = find("ina260_u14"), find("ams")
if ina is None:
    sys.exit("no encontre ina260_u14 en " + ROOT)
temps = sorted(glob.glob(os.path.join(ams, "temp*_input"))) if ams else []
out, period = sys.argv[1], float(sys.argv[2])
dur = float(sys.argv[3]) if len(sys.argv) > 3 else None
p_pw, p_v, p_i = (os.path.join(ina, n) for n in ("power1_input", "in1_input", "curr1_input"))

t_end = time.monotonic() + dur if dur else None
nxt = time.monotonic()
last_temp_t, tvals = 0.0, [""] * len(temps)
try:
    with open(out, "w", buffering=1) as fh:
        fh.write("t_epoch_ns,power_uW,volt_mV,curr_mA" +
                 "".join("," + os.path.basename(t)[:-6] + "_mC" for t in temps) + "\n")
        while t_end is None or time.monotonic() < t_end:
            now = time.monotonic()
            if temps and now - last_temp_t >= 1.0:
                tvals = [rd(t) for t in temps]
                last_temp_t = now
            fh.write(",".join([str(time.time_ns()), rd(p_pw), rd(p_v), rd(p_i)] + tvals) + "\n")
            nxt += period
            d = nxt - time.monotonic()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.monotonic()
except KeyboardInterrupt:
    pass