# _force.py
from abc import ABC, abstractmethod

__all__ = ["NativeForce", "NativeGuoForce"]


class NativeForce(ABC):
    @staticmethod
    @abstractmethod
    def create(*args, **kwargs):
        ...

    @abstractmethod
    def generate(self, generator: "Generator"):
        ...


class NativeGuoForce(NativeForce):
    """
    Guo-Quelle (native):
      F_i = (1 - 1/(2*tau)) * w_i * [ (e_i - u)/cs^2 + ((e_i·u) e_i)/cs^4 ] · a

    Erwartet:
      - Kernel hat f_reg, rho, u[d], e[q][d], w[q], cs (kommt aus deinem Template)
      - Skalar-Args: ax,(ay),(az) (und tau, wenn pass_tau_as_param=True)
      - Optional: force_mask (byte), um den Source-Term nur auf markierten Knoten zu addieren
    """

    def __init__(self, stencil, tau: float, use_mask: bool = False, pass_tau_as_param: bool = False):
        self.stencil = stencil
        self.tau = float(tau)
        self.use_mask = bool(use_mask)
        self.pass_tau_as_param = bool(pass_tau_as_param)

    @staticmethod
    def create(stencil, tau: float, use_mask: bool = False, pass_tau_as_param: bool = False):
        return NativeGuoForce(stencil, tau, use_mask, pass_tau_as_param)

    def generate(self, g: "Generator"):
        buf = g.append_pipeline_buffer
        d = self.stencil.d
        q = self.stencil.q

        # ---- Wrapper/Kern-Parameter: Beschleunigung ----
        g.launcher_hook("ax", "const double ax", "ax", "simulation.force_ax")
        g.kernel_hook("ax", "scalar_t ax", "static_cast<scalar_t>(ax)")
        if d >= 2:
            g.launcher_hook("ay", "const double ay", "ay", "simulation.force_ay")
            g.kernel_hook("ay", "scalar_t ay", "static_cast<scalar_t>(ay)")
        if d >= 3:
            g.launcher_hook("az", "const double az", "az", "simulation.force_az")
            g.kernel_hook("az", "scalar_t az", "static_cast<scalar_t>(az)")

        # ---- Optional: Mask (byte) ----
        if self.use_mask:
            g.launcher_hook("force_mask", "const at::Tensor force_mask", "force_mask", "simulation.force_mask")
            g.kernel_hook("force_mask", "const byte_t* force_mask", "force_mask.data<byte_t>()")

        # ---- tau: als Param oder fest eingebrannt ----
        if self.pass_tau_as_param:
            g.launcher_hook("tau", "const double tau", "tau", "simulation.force_tau")
            g.kernel_hook("tau", "scalar_t tau", "static_cast<scalar_t>(tau)")

        # ---- CUDA-Code ----
        buf("{")
        if self.pass_tau_as_param:
            buf("  const scalar_t pref = (scalar_t)1.0 - (scalar_t)0.5 / tau;")
        else:
            buf(f"  const scalar_t pref = (scalar_t)1.0 - (scalar_t)0.5 / (scalar_t){self.tau};")

        # Achtung: Dein Template definiert '#define cs {cs}'. Wir nehmen sicherheitshalber cs^2 = cs*cs:
        buf("  const scalar_t invcs2 = (scalar_t)1.0 / ((scalar_t)cs * (scalar_t)cs);")
        buf("  const scalar_t invcs4 = invcs2 * invcs2;")

        buf("  const scalar_t ax_loc = ax;")
        if d >= 2: buf("  const scalar_t ay_loc = ay;")
        if d >= 3: buf("  const scalar_t az_loc = az;")

        if self.use_mask:
            buf("  if (force_mask[node_index]) {")

        buf("  scalar_t eudot, emu_x, emu_y, emu_z, term_x, term_y, term_z, contrib;")

        for i in range(q):
            if d == 1:
                buf(f"  eudot = e[{i}][0]*u[0];")
                buf(f"  emu_x = e[{i}][0] - u[0];")
                buf(f"  term_x = emu_x*invcs2 + eudot*e[{i}][0]*invcs4;")
                buf(f"  contrib = term_x*ax_loc;")
            elif d == 2:
                buf(f"  eudot = e[{i}][0]*u[0] + e[{i}][1]*u[1];")
                buf(f"  emu_x = e[{i}][0] - u[0];")
                buf(f"  emu_y = e[{i}][1] - u[1];")
                buf(f"  term_x = emu_x*invcs2 + eudot*e[{i}][0]*invcs4;")
                buf(f"  term_y = emu_y*invcs2 + eudot*e[{i}][1]*invcs4;")
                buf(f"  contrib = term_x*ax_loc + term_y*ay_loc;")
            else:
                buf(f"  eudot = e[{i}][0]*u[0] + e[{i}][1]*u[1] + e[{i}][2]*u[2];")
                buf(f"  emu_x = e[{i}][0] - u[0];")
                buf(f"  emu_y = e[{i}][1] - u[1];")
                buf(f"  emu_z = e[{i}][2] - u[2];")
                buf(f"  term_x = emu_x*invcs2 + eudot*e[{i}][0]*invcs4;")
                buf(f"  term_y = emu_y*invcs2 + eudot*e[{i}][1]*invcs4;")
                buf(f"  term_z = emu_z*invcs2 + eudot*e[{i}][2]*invcs4;")
                buf(f"  contrib = term_x*ax_loc + term_y*ay_loc + term_z*az_loc;")

            buf(f"  f_reg[{i}] += pref * w[{i}] * contrib;")

        if self.use_mask:
            buf("  }")  # end if mask

        buf("}")
