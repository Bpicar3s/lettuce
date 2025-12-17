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
    "python": False,
    "native": True,
}

HS = [8, 12, 16]
RUNS_PER_H = 2
TMAX = 1
RE = 180
MACH = 0.1
NEWTON_SPEEDUP = True

OUTDIR = "./output_benchmark/"
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# Simulation-Factory
# ============================================================

def make_simulation(h: int, dtype: torch.dtype, use_native: bool):

    context = lt.Context(
        device=DEVICE,
        dtype=dtype,
        use_native=use_native
    )

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
    # Native WallFunction
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
# Benchmark
# ============================================================

for prec_name, dtype in PRECISIONS.items():
    for impl_name, use_native in IMPLEMENTATIONS.items():

        print(f"\n==============================")
        print(f"Benchmark: {prec_name.upper()} / {impl_name.upper()}")
        print(f"==============================")

        csv_file = os.path.join(
            OUTDIR, f"mlups_{prec_name}_{impl_name}.csv"
        )

        results = []

        for h in HS:
            print(f"\n--- h = {h} ---")
            mlups_runs = []

            for r in range(RUNS_PER_H):
                print(f"  Run {r+1}/{RUNS_PER_H}")

                sim = make_simulation(h, dtype, use_native)

                steps = int(
                    sim.flow.units.convert_time_to_lu(TMAX)
                )

                mlups = sim.step(num_steps=steps)
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
                impl_name
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
                "implementation"
            ])
            writer.writerows(results)

        print(f"\n→ Ergebnisse gespeichert in: {csv_file}")

print("\nBenchmark abgeschlossen.")
