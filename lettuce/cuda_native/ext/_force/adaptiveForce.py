# adaptive_force.py
import torch
from lettuce.ext._boundary.wallfunction import compute_wall_quantities
from ._force import NativeGuoForce

__all__ = ["AdaptiveForce"]


class AdaptiveForce:
    """
    CUDA-native Adaptive Force:
      - berechnet globale a=(Fx,0,0) aus u_tau (Spalding, y=0.5 LU) + Zielmittel u_m
      - setzt vor dem Kernellauf die Skalar-Args:
            simulation.force_ax / force_ay / force_az
        (und optional: simulation.force_mask)
      - liefert optional CPU-Fallback-Quelle (Guo-Formel) für Nicht-Native Runs
    """

    def __init__(self, flow, context, target_u_m_lu,
                 base_lbm_tau_lu, mask=None,
                 use_mask_in_native=True,
                 pass_tau_as_param=False):
        self.flow = flow
        self.context = context
        self.u_m = float(target_u_m_lu)
        self.base_lbm_tau = float(base_lbm_tau_lu)

        self.mask = None
        if mask is not None:
            self.mask = context.convert_to_tensor(mask).to(torch.bool)
        self.use_mask_in_native = bool(use_mask_in_native)

        d = flow.stencil.d
        self.last_force_lu = context.convert_to_tensor([0.0] * d)

        # Native Guo-Force, wird in die Collision injiziert
        self.native = NativeGuoForce.create(
            stencil=self.flow.stencil,
            tau=self.base_lbm_tau,
            use_mask=(self.mask is not None and self.use_mask_in_native),
            pass_tau_as_param=bool(pass_tau_as_param),
        )

    # -------------------- numerik: kraft bestimmen --------------------
    def compute_force(self):
        """
        Fx = (u_tau_mean^2)/H + (u_m - <u_x>)*(u_m/H)
        u_tau_mean von top/bottom mit Spalding, y=0.5 LU (halfway zum 1. Fluidknoten)
        """
        device = self.flow.f.device
        dtype = self.flow.f.dtype
        dy = torch.as_tensor(0.5, device=device, dtype=dtype)

        utau_top, _, _ = compute_wall_quantities(flow=self.flow, dy=dy, is_top=True)
        utau_bot, _, _ = compute_wall_quantities(flow=self.flow, dy=dy, is_top=False)
        utau_mean = 0.5 * (utau_top.mean() + utau_bot.mean())

        ux_mean = self.flow.u()[0].mean()
        Fx = (utau_mean ** 2) / self.flow.h + (self.u_m - ux_mean) * (self.u_m / self.flow.h)

        vec = [0.0] * self.flow.stencil.d
        vec[0] = float(Fx)
        self.last_force_lu = self.context.convert_to_tensor(vec)
        return self.last_force_lu

    # -------------------- CUDA-native: args setzen --------------------
    def set_on_simulation(self, simulation):
        """
        Vor jedem Kernel-Lauf aufrufen.
        Legt force_ax/ay/az (+ optional force_mask, force_tau) auf 'simulation' ab,
        damit der NativeGuoForce-Block die Werte per launcher_hook bekommt.
        """
        self.compute_force()

        simulation.force_ax = float(self.last_force_lu[0].item())
        if self.flow.stencil.d >= 2:
            simulation.force_ay = float(self.last_force_lu[1].item())
        else:
            simulation.force_ay = 0.0
        if self.flow.stencil.d >= 3:
            simulation.force_az = float(self.last_force_lu[2].item())
        else:
            simulation.force_az = 0.0

        if self.mask is not None and self.use_mask_in_native:
            simulation.force_mask = self.mask.to(torch.uint8)

        if self.native.pass_tau_as_param:
            simulation.force_tau = float(self.base_lbm_tau)

    # -------------------- CPU-fallback (optional) --------------------
    def source_term_cpu(self, u_field_lu):
        """
        Guo-Quelle in Python (falls ohne native laufen).
        Liefert Tensor [q, Nx, Ny, Nz]; mask wird angewandt, falls gesetzt.
        """
        e = self.flow.torch_stencil.e           # [d, q]
        w = self.flow.torch_stencil.w           # [q]
        cs2 = self.flow.torch_stencil.cs ** 2
        pref = 1.0 - 1.0 / (2.0 * self.base_lbm_tau)

        # (e - u)
        emu = e.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) - u_field_lu  # [d,q,...]
        emu = emu.movedim(0, 1)                                         # [q,d,...]
        eu = torch.einsum("dq,b...->q...", e, u_field_lu)               # [q,...]
        eeu = torch.einsum("dq,q...->dq...", e, eu).movedim(0, 1)       # [q,d,...]
        term = emu / cs2 + eeu / (cs2 * cs2)
        contrib = torch.einsum("qd...,d->q...", term, self.last_force_lu)
        src = pref * (w.view(-1, *([1] * self.flow.stencil.d)) * contrib)

        if self.mask is not None:
            mask4d = self.mask.to(src.dtype)[None, ...]
            src = src * mask4d
        return src

    # bequemer Alias (z. B. wenn alte Stellen source_term(u) erwarten)
    def __call__(self, u_field_lu, f=None):
        return self.source_term_cpu(u_field_lu)
