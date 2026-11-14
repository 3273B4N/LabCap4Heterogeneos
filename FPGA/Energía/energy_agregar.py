#!/usr/bin/env python3
"""Resume la energia entre repeticiones (mediana, minimo y maximo), con una o varias corridas de run_energy.sh.
uso: python3 energy_agregar.py <carpeta_run> [<carpeta_run> ...]
Lee <carpeta>/*/energy_resumen_*.csv de cada carpeta; cada archivo cuenta como una repeticion.
Una carpeta  -> escribe <carpeta>/energia_resumen.csv
Varias       -> escribe energia_resumen_fusion.csv en la carpeta actual (las repeticiones se suman)
Avisa si alguna configuracion tiene menos de MIN_REPS repeticiones."""
import csv, glob, os, statistics, sys

MIN_REPS = 5
if len(sys.argv) < 2:
    sys.exit(__doc__)
roots = sys.argv[1:]
groups = {}
for root in roots:
    for p in sorted(glob.glob(os.path.join(root, "*", "energy_resumen_*.csv"))):
        for r in csv.DictReader(open(p)):
            key = (r["modo"], r["prec"], r["variante"], r["mhz"], int(r["N"]))
            groups.setdefault(key, []).append(r)

def st(rows, col):
    v = [float(r[col]) for r in rows if r[col] != ""]
    if not v:
        return ["", "", ""]
    return [round(statistics.median(v), 4), round(min(v), 4), round(max(v), 4)]

hdr = ["modo", "prec", "variante", "mhz", "N", "reps",
       "P_med_W", "P_min_W", "P_max_W",
       "E_llamada_med_uJ", "E_llamada_min_uJ", "E_llamada_max_uJ",
       "E_elem_med_uJ", "E_neta_llamada_med_uJ", "E_neta_llamada_min_uJ", "E_neta_llamada_max_uJ"]
rows = []
for key in sorted(groups):
    rs = groups[key]
    rows.append(list(key[:4]) + [key[4], len(rs)] + st(rs, "P_media_W") + st(rs, "E_llamada_uJ")
                + st(rs, "E_elem_uJ")[:1] + st(rs, "E_neta_llamada_uJ"))
out = os.path.join(roots[0], "energia_resumen.csv") if len(roots) == 1 else "energia_resumen_fusion.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(hdr)
    w.writerows(rows)
for r in rows:
    print(f"{r[0]:6s} {r[1]} {r[2]:13s} {r[3]}MHz N={r[4]:4d} reps={r[5]} "
          f"P={r[6]} W [{r[7]}..{r[8]}]  E/llamada={r[9]} uJ [{r[10]}..{r[11]}]  E/elem={r[12]} uJ  neta={r[13]} uJ")
cuentas = sorted({r[5] for r in rows})
pocas = [r for r in rows if r[5] < MIN_REPS]
print(f"repeticiones por configuracion: {cuentas}")
if pocas:
    print(f"AVISO: {len(pocas)} de {len(rows)} configuraciones tienen menos de {MIN_REPS} repeticiones")
print("escrito:", out)