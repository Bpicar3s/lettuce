#!/usr/bin/env python3
import os, torch, numpy as np, matplotlib.pyplot as plt
import lettuce as lt
from lettuce import D3Q19
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._reporter.observable_reporter import (
    GlobalMeanUXReporter, WallQuantities, WallfunctionReporter, AdaptiveAcceleration
)
from lettuce.ext._force.Kupershtokh import ExactDifferenceForce

# ======================================================
# ⚙️ Parameter
# ======================================================
h = 20
Re = 180
Mach = 0.1
tmax = 100
dtype = torch.float64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
basedir = "./output/"

print(f"CUDA verfügbar: {torch.cuda.is_available()}")
print(f"Device: {device}, Dtype: {dtype}")

# ======================================================
# 💡 Kontext
# ======================================================
context = lt.Context(device=device, dtype=dtype, use_native=False)

# ======================================================
# 💡 Funktion für eine einzelne Simulation
# ======================================================
def run_channel_simulation(newton_speedup=True):
    print(f"\n=== Starte Simulation mit newton_speedup={newton_speedup} ===")

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
    mask_bottom = torch.zeros(shape, dtype=torch.bool, device=device); mask_bottom[:, 0, :] = True
    mask_top    = torch.zeros(shape, dtype=torch.bool, device=device); mask_top[:, -1, :] = True

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
    wq_bottom = WallQuantities(mask=mask_bottom, wall="bottom", flow=flow, newton_speedup=newton_speedup, context=context)
    wq_top    = WallQuantities(mask=mask_top,    wall="top",    flow=flow, newton_speedup=newton_speedup, context=context)

    simulation = lt.Simulation(flow=flow, collision=collision, reporter=[])
    simulation.reporter.append(lt.ObservableReporter(global_mean_ux_reporter, interval=1, out=None))
    simulation.reporter.append(lt.ObservableReporter(wq_bottom, interval=100, out=None))
    simulation.reporter.append(lt.ObservableReporter(wq_top,    interval=100, out=None))
    simulation.reporter.append(lt.ObservableReporter(adaptive_accel, interval=50, out=None))

    # WallFunction + Reporter
    collision_py = lt.BGKCollision(tau=flow.units.relaxation_parameter_lu, force=force)
    wfb_bottom = WallFunction(mask_bottom, flow.stencil, h, context, wall="bottom", newton_speedup=newton_speedup)
    wfb_top    = WallFunction(mask_top,    flow.stencil, h, context, wall="top",    newton_speedup=newton_speedup)

    mask_no_collision2 = torch.ones(flow.resolution, dtype=torch.bool, device=device)
    mask_no_collision2[:, 1, :]  = False
    mask_no_collision2[:, -2, :] = False

    wfb_reporter = WallfunctionReporter(context, flow, collision_py, mask_no_collision2, wfb_bottom, wfb_top)
    simulation.reporter.append(lt.ObservableReporter(wfb_reporter, interval=1, out=None))

    # Simulation starten
    steps = int(flow.units.convert_time_to_lu(tmax))
    mlups = simulation.step(num_steps=steps)

    # Ergebnisse
    data_wfb = np.array(simulation.reporter[-1].out)
    mean_it = data_wfb[:, 2]
    max_it  = data_wfb[:, 3]
    time    = data_wfb[:, 1]

    print(f"Simulation beendet. MLUPS = {mlups:.2f}")
    return time, mean_it, max_it, mlups

# ======================================================
# 💡 Beide Simulationen ausführen
# ======================================================
time_true, mean_true, max_true, mlups_true = run_channel_simulation(newton_speedup=True)
time_false, mean_false, max_false, mlups_false = run_channel_simulation(newton_speedup=False)

# ======================================================
# 📈 Plot
# ======================================================
plt.figure(figsize=(6,4))
plt.plot(time_true,  mean_true, "o-", label="MeanIt (Speedup=True)")
plt.plot(time_true,  max_true,  "s--", label="MaxIt (Speedup=True)")
plt.plot(time_false, mean_false,"x-", label="MeanIt (Speedup=False)")
plt.plot(time_false, max_false, "d--", label="MaxIt (Speedup=False)")
plt.xlabel("Zeit [LU]")
plt.ylabel("Iterationsanzahl")
plt.title("Newton-Speedup Vergleich: mean_it und max_it (mit AdaptiveAcceleration)")
plt.legend(); plt.grid(); plt.tight_layout()

os.makedirs(basedir, exist_ok=True)
plt.savefig(os.path.join(basedir,"newton_speedup_iterations_adaptive.pdf"))
plt.close()

# ======================================================
# 💬 Zusammenfassung
# ======================================================
print("\n--- Leistungsübersicht (mit AdaptiveAcceleration) ---")
print(f"MLUPS (Speedup=True):  {mlups_true:.2f}")
print(f"MLUPS (Speedup=False): {mlups_false:.2f}")
print(f"Speedup-Faktor:        {mlups_true / mlups_false:.2f}x")
print(f"Plot gespeichert unter: {basedir}newton_speedup_iterations_adaptive.pdf")
