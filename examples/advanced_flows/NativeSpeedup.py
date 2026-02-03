import torch
import numpy as np
import lettuce as lt
from lettuce import D3Q19
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._reporter.observable_reporter import (
    AdaptiveAcceleration,
    WallfunctionReporter
)
from lettuce.ext._force.Kupershtokh import ExactDifferenceForce
import csv
import os
import gc

# ============================================================
# Globale Parameter
# ============================================================

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PRECISIONS = {
    "single": torch.float32,
    "double": torch.float64,
}

IMPLEMENTATIONS = {
    "python": False,   # use_native = False
    "native": True,    # use_native = True
}

# Zwei Szenarien: WallFunction + Fullway-BounceBack
SCENARIOS = ["wallfunction", "fullway"]

HS = [10, 20, 30, 40, 50, 60, 70, 80]
RUNS_PER_H = 5

# Feste Schrittzahlen für Benchmark
WARMUP_STEPS = 200
MEASURE_STEPS = 1000

RE = 180
MACH = 0.1
NEWTON_SPEEDUP = True

OUTDIR = "./output_benchmark/"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Simulation-Factory: WallFunction-Szenario
# ============================================================

def make_simulation_wallfunction(h: int, dtype: torch.dtype, use_native: bool):

    context = lt.Context(
        device=DEVICE,
        dtype=dtype,
        use_native=use_native
    )

    # WICHTIG:
    # - native = False  -> bbtype="wallfunction" (Boundary im Flow)
    # - native = True   -> bbtype=None, Wandfunktion steckt im Reporter
    flow = lt.ChannelFlow3D(
        context=context,
        resolution=h,
        reynolds_number=RE**(8/7)*(8/0.073)**(4/7),
        stencil=D3Q19(),
        mach_number=MACH,
        bbtype=None if use_native else "wallfunction"
    )

    # -------------------------
    # Force
    # -------------------------
    force = ExactDifferenceForce(
        flow=flow,
        acceleration=[0.0, 0.0, 0.0]
    )

    adaptive_accel = AdaptiveAcceleration(
        flow=flow,
        force_obj=force,
        target_mean_ux_lu=flow.units.convert_velocity_to_lu(1.0),
        context=context,
        k_gain=1.0,
        Re_tau=RE
    )

    collision = lt.BGKCollision(
        tau=flow.units.relaxation_parameter_lu,
        force=force
    )

    simulation = lt.Simulation(
        flow=flow,
        collision=collision,
        reporter=[]
    )

    # -------------------------
    # Native WallFunction (als Reporter-Hack)
    # -------------------------
    if use_native:
        shape = flow.resolution

        mask_bottom = torch.zeros(shape, dtype=torch.bool, device=context.device)
        mask_bottom[:, 0, :] = True

        mask_top = torch.zeros(shape, dtype=torch.bool, device=context.device)
        mask_top[:, -1, :] = True

        collision_py = lt.BGKCollision(
            tau=flow.units.relaxation_parameter_lu,
            force=force
        )

        wfb_bottom = WallFunction(
            mask_bottom, flow.stencil, h, context,
            wall="bottom", newton_speedup=NEWTON_SPEEDUP
        )
        wfb_top = WallFunction(
            mask_top, flow.stencil, h, context,
            wall="top", newton_speedup=NEWTON_SPEEDUP
        )

        # Maske: wo KEINE Kollision stattfinden soll (Wandzellen)
        mask_no_collision = torch.ones(shape, dtype=torch.bool, device=context.device)
        mask_no_collision[:, 1, :] = False
        mask_no_collision[:, -2, :] = False

        wfb_reporter = WallfunctionReporter(
            context,
            flow,
            collision_py,
            mask_no_collision,
            wfb_bottom,
            wfb_top
        )

        simulation.reporter.append(
            lt.ObservableReporter(wfb_reporter, interval=1, out=None)
        )

    # -------------------------
    # Adaptive Force (immer!)
    # -------------------------
    simulation.reporter.append(
        lt.ObservableReporter(adaptive_accel, interval=50, out=None)
    )

    # -------------------------
    # Force-Acceleration absichern
    # -------------------------
    force.acceleration = (
        force.acceleration
        .to(device=flow.f.device, dtype=flow.f.dtype)
        .contiguous()
    )

    return simulation


# ============================================================
# Simulation-Factory: Fullway-BounceBack-Szenario
# ============================================================

def make_simulation_fullway(h: int, dtype: torch.dtype, use_native: bool):

    context = lt.Context(
        device=DEVICE,
        dtype=dtype,
        use_native=use_native
    )

    # Hier **immer** Fullway-Bounceback als Boundary im Flow
    # (python: Python-Boundary; native: selbe Boundary, aber mit native_generator)
    flow = lt.ChannelFlow3D(
        context=context,
        resolution=h,
        reynolds_number=RE**(8/7)*(8/0.073)**(4/7),
        stencil=D3Q19(),
        mach_number=MACH,
        bbtype="fullway"
    )

    # -------------------------
    # Force
    # -------------------------
    force = ExactDifferenceForce(
        flow=flow,
        acceleration=[0.0, 0.0, 0.0]
    )

    adaptive_accel = AdaptiveAcceleration(
        flow=flow,
        force_obj=force,
        target_mean_ux_lu=flow.units.convert_velocity_to_lu(1.0),
        context=context,
        k_gain=1.0,
        Re_tau=RE
    )

    collision = lt.BGKCollision(
        tau=flow.units.relaxation_parameter_lu,
        force=force
    )

    simulation = lt.Simulation(
        flow=flow,
        collision=collision,
        reporter=[]
    )

    # Optional: hier KEIN extra WallFunction-Reporter,
    # weil die Wand schon durch Fullway-BB im Flow abgedeckt ist.

    simulation.reporter.append(
        lt.ObservableReporter(adaptive_accel, interval=50, out=None)
    )

    force.acceleration = (
        force.acceleration
        .to(device=flow.f.device, dtype=flow.f.dtype)
        .contiguous()
    )

    return simulation


# ============================================================
# Benchmark
# ============================================================

for scenario in SCENARIOS:

    print(f"\n########################################")
    print(f"SCENARIO: {scenario.upper()}")
    print(f"########################################")

    # Wähle passende Factory
    if scenario == "wallfunction":
        sim_factory = make_simulation_wallfunction
    elif scenario == "fullway":
        sim_factory = make_simulation_fullway
    else:
        raise ValueError(f"Unbekanntes Szenario: {scenario}")

    for prec_name, dtype in PRECISIONS.items():
        for impl_name, use_native in IMPLEMENTATIONS.items():

            print(f"\n==============================")
            print(f"Benchmark: {scenario} | {prec_name.upper()} / {impl_name.upper()}")
            print(f"==============================")

            csv_file = os.path.join(
                OUTDIR, f"mlups_{scenario}_{prec_name}_{impl_name}.csv"
            )

            results = []

            for h in HS:
                print(f"\n--- h = {h} ---")
                mlups_runs = []

                for r in range(RUNS_PER_H):
                    print(f"  Run {r+1}/{RUNS_PER_H}")

                    sim = sim_factory(h, dtype, use_native)

                    # -------------------------
                    # Warmup (ohne Messung)
                    # -------------------------
                    _ = sim.step(num_steps=WARMUP_STEPS)

                    # GPU synchronisieren, damit Warmup wirklich fertig ist
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()

                    # -------------------------
                    # Messlauf (für MLUPS)
                    # -------------------------
                    mlups = sim.step(num_steps=MEASURE_STEPS)
                    mlups_runs.append(mlups)

                    print(f"    MLUPS = {mlups:.2f}")

                    del sim
                    gc.collect()
                    torch.cuda.empty_cache()

                mlups_runs = np.array(mlups_runs)
                mean_mlups = mlups_runs.mean()
                std_mlups = mlups_runs.std()

                print(
                    f"  → {mean_mlups:.2f} ± {std_mlups:.2f} MLUPS"
                )

                results.append([
                    h,
                    mean_mlups,
                    std_mlups,
                    prec_name,
                    impl_name,
                    scenario
                ])

            # -------------------------
            # CSV schreiben
            # -------------------------
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "h",
                    "MLUPS_mean",
                    "MLUPS_std",
                    "precision",
                    "implementation",
                    "scenario"
                ])
                writer.writerows(results)

            print(f"\n→ Ergebnisse gespeichert in: {csv_file}")

print("\nBenchmark abgeschlossen.")
