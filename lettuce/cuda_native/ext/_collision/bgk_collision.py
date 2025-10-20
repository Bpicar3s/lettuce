from typing import Optional
from lettuce.cuda_native import DefaultCodeGeneration, Parameter
from ... import NativeCollision
import torch

__all__ = ["NativeBGKCollision"]


# =====================================================================
# ✅ DummyForce (Fallback, falls keine Force übergeben wird)
# =====================================================================
class DummyForce:
    """
    Platzhalter-Force, falls keine echte Kraft übergeben wurde.
    Sie stellt sicher, dass simulation.collision.force.acceleration
    immer existiert, damit der Native-Kernel nicht crasht.
    """
    def __init__(self, flow):
        self.flow = flow
        self.acceleration = flow.context.convert_to_tensor(
            [0.0] * flow.stencil.d,
            dtype=flow.f.dtype,
            device=flow.f.device,
        )

    def native_available(self):
        return True

    def native_generator(self):
        # Kein eigener Native-Code, wird nur als Datenhalter verwendet
        return None


# =====================================================================
# ✅ NativeBGKCollision mit optionalem Forcing (Kupershtokh-kompatibel)
# =====================================================================
class NativeBGKCollision(NativeCollision):
    def __init__(self, index: int, flow=None, force: Optional["Force"] = None):
        super().__init__(index)
        self.flow = flow
        # Falls keine Force existiert → DummyForce verwenden
        self.force = force if force is not None else DummyForce(flow)

    @staticmethod
    def create(index: int, flow=None, force: Optional["Force"] = None):
        # Immer ein gültiges Objekt erzeugen
        if force is None:
            force = DummyForce(flow)
        return NativeBGKCollision(index, flow, force)

    # -----------------------------------------------------------------
    # Hook für τ⁻¹
    # -----------------------------------------------------------------
    def cuda_tau_inv(self, reg: "DefaultCodeGeneration"):
        variable = f"tau_inv{hex(id(self))[2:]}"
        return reg.cuda_hook(
            "1.0 / simulation.collision.tau", Parameter("double", variable)
        )

    def kernel_tau_inv(self, reg: "DefaultCodeGeneration"):
        variable = self.cuda_tau_inv(reg)
        return reg.kernel_hook(
            f"static_cast<scalar_t>({variable})", Parameter("scalar_t", variable)
        )

    # -----------------------------------------------------------------
    # Haupt-Codegenerator
    # -----------------------------------------------------------------
    def generate(self, reg: "DefaultCodeGeneration"):
        tau_inv = self.kernel_tau_inv(reg)

        # Hooke rho & u aus Lettuce-Makros (immer verfügbar)
        rho = reg.kernel_rho()
        u = [reg.kernel_u(d) for d in range(reg.stencil.d)]

        # -------------------------------
        # (1) Kein Forcing → klassisches BGK
        # -------------------------------
        if isinstance(self.force, DummyForce):
            for q in range(reg.stencil.q):
                f_q = reg.f_reg(q)
                f_eq_q = reg.equilibrium.f_eq(reg, q)
                reg.pipe.append(f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {f_eq_q}));")
            return

        # -------------------------------
        # (2) Forcing aktiv → Kupershtokh-Quelle hinzufügen
        # -------------------------------
        accel_name = f"acceleration_{hex(id(self))[2:]}"
        reg.cuda_hook(
            "simulation.collision.force.acceleration",
            Parameter("torch::Tensor", accel_name),
        )
        reg.kernel_hook(
            f"{accel_name}.accessor<scalar_t, 1>()",
            Parameter("const auto", accel_name),
        )

        # Konstanten aus dem Stencil
        e = reg.stencil.e
        w = reg.stencil.w
        cs2 = reg.stencil.cs**2

        # Kupershtokh: f_eq(u + 0.5a) - f_eq(u)
        for q in range(reg.stencil.q):
            f_q = reg.f_reg(q)
            f_eq_q = reg.equilibrium.f_eq(reg, q)

            ex, ey, ez = (
                float(e[q][0]),
                float(e[q][1]) if reg.stencil.d > 1 else 0.0,
                float(e[q][2]) if reg.stencil.d > 2 else 0.0,
            )
            wq = float(w[q])

            # u_plus = u + 0.5a
            ux_p = f"({u[0]} + static_cast<scalar_t>(0.5)*{accel_name}[0])"
            uy_p = f"({u[1]} + static_cast<scalar_t>(0.5)*{accel_name}[1])"
            uz_p = f"({u[2]} + static_cast<scalar_t>(0.5)*{accel_name}[2])"

            eu_p = f"(({ex})*{ux_p} + ({ey})*{uy_p} + ({ez})*{uz_p})"
            u2_p = f"({ux_p}*{ux_p} + {uy_p}*{uy_p} + {uz_p}*{uz_p})"

            # f_eq(u+0.5a)
            f_eq_plus = (
                f"(static_cast<scalar_t>({wq})*{rho}*("
                f"1 + ({eu_p})/({cs2}) "
                f"+ static_cast<scalar_t>(0.5)*(({eu_p})*({eu_p}))/(({cs2})*({cs2})) "
                f"- static_cast<scalar_t>(0.5)*({u2_p})/({cs2})"
                f"))"
            )

            # Quelle: S_i = f_eq(u+0.5a) - f_eq(u)
            s_q = f"(({f_eq_plus}) - ({f_eq_q}))"

            # Voller BGK-Schritt mit Quelle
            reg.pipe.append(
                f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {f_eq_q})) + {s_q};"
            )
