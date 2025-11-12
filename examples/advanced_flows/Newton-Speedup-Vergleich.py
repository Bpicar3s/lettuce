#!/usr/bin/env python3
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import lettuce as lt
from lettuce import D3Q19
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._reporter.observable_reporter import (
    GlobalMeanUXReporter, WallQuantities, WallfunctionReporter, AdaptiveAcceleration
)
from lettuce.ext._force.Kupershtokh import ExactDifferenceForce
import argparse
import csv  # Wichtig für das Speichern

# ======================================================
# ⚙️ Argumente
# ======================================================
parser = argparse.ArgumentParser()
parser.add_argument("--Re", type=int, default=180)
parser.add_argument("--h", type=int, default=20)
parser.add_argument("--tmax", type=float, default=1)
parser.add_argument("--Precision", type=str, default="Double", choices=["Single", "Double", "Half"])
parser.add_argument("--Mach", type=float, default=0.1)
parser.add_argument("--output", type=str, default="./output/")
args = parser.parse_args()

# Parameter
Re = args.Re
h = args.h
tmax = args.tmax
Mach = args.Mach
basedir = args.output

dtype = torch.float64


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(f"Device: {device}, Dtype: {dtype}")

# ======================================================
# 💡 Kontext
# ======================================================
context = lt.Context(device=device, dtype=dtype, use_native=False)


# ======================================================
# 💡 Simulation
# ======================================================
def run_channel_simulation(newton_speedup=True, suffix=""):
    """
    Führt die Simulation aus UND speichert die Haupt-Reporterdaten
    als CSV-Dateien mit dem angegebenen Suffix.
    """
    print(f"\n=== Starte Simulation mit newton_speedup={newton_speedup} (Suffix: {suffix}) ===")

    flow = lt.ChannelFlow3D(
        context=context,
        resolution=h,
        reynolds_number=6432 if Re == 180 else Re ** (8 / 7) * (8 / 0.073) ** (4 / 7),
        stencil=D3Q19(),
        mach_number=Mach,
        bbtype=None
    )

    # Masken
    shape = flow.resolution
    mask_bottom = torch.zeros(shape, dtype=torch.bool, device=device);
    mask_bottom[:, 0, :] = True
    mask_top = torch.zeros(shape, dtype=torch.bool, device=device);
    mask_top[:, -1, :] = True

    # Force & Collision
    force = ExactDifferenceForce(flow, acceleration=[0, 0, 0])
    collision = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu, force=force)

    # Adaptive Acceleration
    adaptive_accel = AdaptiveAcceleration(
        flow=flow,
        force_obj=force,
        target_mean_ux_lu=flow.units.convert_velocity_to_lu(1.0),
        context=context,
        k_gain=1.0
    )

    # Reporter
    global_mean_ux_reporter = GlobalMeanUXReporter(flow=flow)
    wq_bottom = WallQuantities(mask=mask_bottom, wall="bottom", flow=flow, newton_speedup=newton_speedup,
                               context=context)
    wq_top = WallQuantities(mask=mask_top, wall="top", flow=flow, newton_speedup=newton_speedup, context=context)

    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
    # WICHTIG: Die Reihenfolge bestimmt die Indizes
    simulation.reporter.append(lt.ObservableReporter(global_mean_ux_reporter, interval=1, out=None))  # -> Index [0]
    simulation.reporter.append(lt.ObservableReporter(wq_bottom, interval=100, out=None))  # -> Index [1]
    simulation.reporter.append(lt.ObservableReporter(wq_top, interval=100, out=None))  # -> Index [2]
    simulation.reporter.append(lt.ObservableReporter(adaptive_accel, interval=50, out=None))  # -> Index [3]

    # WallFunction + Reporter
    collision_py = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu, force=force)
    wfb_bottom = WallFunction(mask_bottom, flow.stencil, h, context, wall="bottom", newton_speedup=newton_speedup)
    wfb_top = WallFunction(mask_top, flow.stencil, h, context, wall="top", newton_speedup=newton_speedup)

    mask_no_collision2 = torch.ones(flow.resolution, dtype=torch.bool, device=device)
    mask_no_collision2[:, 1, :] = False
    mask_no_collision2[:, -2, :] = False

    wfb_reporter = WallfunctionReporter(context, flow, collision_py, mask_no_collision2, wfb_bottom, wfb_top)
    simulation.reporter.append(lt.ObservableReporter(wfb_reporter, interval=1, out=None))  # -> Index [4]

    # Simulation starten
    steps = int(flow.units.convert_time_to_lu(tmax))
    mlups = simulation.step(num_steps=steps)

    # --- CSV-SPEICHERN (wie im alten Skript) ---
    print(f"Speichere CSV-Daten mit Suffix '{suffix}'...")
    try:
        # Daten aus den Reportern holen (Indizes 0, 1, 2, 4)
        ux_mean_arr = np.array(simulation.reporter[0].out)
        wq_bottom_arr = np.array(simulation.reporter[1].out)
        wq_top_arr = np.array(simulation.reporter[2].out)
        iters_arr = np.array(simulation.reporter[4].out)  # (enthält time, mean_it, max_it)

        # Sicherstellen, dass das Ausgabeverzeichnis existiert
        os.makedirs(basedir, exist_ok=True)

        # Speichern mit deinen Dateinamen + Suffix
        with open(os.path.join(basedir, f'uxmean{suffix}.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(ux_mean_arr)

        with open(os.path.join(basedir, f'WallQuantitiesTop{suffix}.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(wq_top_arr)

        with open(os.path.join(basedir, f'WallQuantitiesBottom{suffix}.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(wq_bottom_arr)

        # Iterationsdaten (iters_arr) auch speichern
        with open(os.path.join(basedir, f'Iterations{suffix}.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["step", "time_lu", "mean_it", "max_it"])
            writer.writerows(iters_arr)

        print("CSV-Speichern erfolgreich.")

    except Exception as e:
        print(f"FEHLER beim Speichern der CSVs: {e}")

    # --- Ergebnisse für den Plot zurückgeben ---
    # iters_arr hat die Spalten: [step, time, mean_it, max_it]
    time = iters_arr[:, 1]
    mean_it = iters_arr[:, 2]
    max_it = iters_arr[:, 3]

    print(f"Simulation beendet. MLUPS = {mlups:.2f}")
    return time, mean_it, max_it, mlups


# ======================================================
# 💡 Beide Simulationen ausführen
# ======================================================
time_true, mean_true, max_true, mlups_true = run_channel_simulation(
    newton_speedup=True, suffix="_true"
)
time_false, mean_false, max_false, mlups_false = run_channel_simulation(
    newton_speedup=False, suffix="_false"
)

# ======================================================
# 📈 Plot
# ======================================================
print("Erstelle Plot...")
plt.figure(figsize=(6, 4))
plt.plot(time_true, mean_true, "o-", label="MeanIt (Speedup=True)")
plt.plot(time_true, max_true, "s--", label="MaxIt (Speedup=True)")
plt.plot(time_false, mean_false, "x-", label="MeanIt (Speedup=False)")
plt.plot(time_false, max_false, "d--", label="MaxIt (Speedup=False)")
plt.xlabel("Zeit [LU]")
plt.ylabel("Iterationsanzahl")
plt.title("Newton-Speedup Vergleich: mean_it und max_it (mit AdaptiveAcceleration)")
plt.legend();
plt.grid();
plt.tight_layout()

# (Sicherstellen, dass das Verzeichnis existiert, falls noch nicht geschehen)
os.makedirs(basedir, exist_ok=True)
plt.savefig(os.path.join(basedir, "newton_speedup_iterations_adaptive.pdf"))
plt.close()

# ======================================================
# 💬 Zusammenfassung
# ======================================================
print("\n--- Leistungsübersicht (mit AdaptiveAcceleration) ---")
print(f"MLUPS (Speedup=True):  {mlups_true:.2f}")
print(f"MLUPS (Speedup=False): {mlups_false:.2f}")
print(f"Speedup-Faktor:        {mlups_true / mlups_false:.2f}x")

summary_path = os.path.join(basedir, "mlups_summary.txt")
print(f"Plot gespeichert unter: {os.path.join(basedir, 'newton_speedup_iterations_adaptive.pdf')}")
print(f"CSV-Daten gespeichert in: {basedir}")
print(f"Zusammenfassung gespeichert unter: {summary_path}")

# MLUPS-Werte auch in Datei speichern
with open(summary_path, "w") as f:
    f.write("--- Leistungsübersicht (mit AdaptiveAcceleration) ---\n")
    f.write(f"Re = {Re}, h = {h}, tmax = {tmax}, Precision = {args.Precision}\n")
    f.write(f"MLUPS (Speedup=True):  {mlups_true:.2f}\n")
    f.write(f"MLUPS (Speedup=False): {mlups_false:.2f}\n")
    f.write(f"Speedup-Faktor:        {mlups_true / mlups_false:.2f}x\n")