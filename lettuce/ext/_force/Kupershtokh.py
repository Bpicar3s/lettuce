import torch
from lettuce.util import append_axes
from . import Force

__all__ = ['ExactDifferenceForce']


class ExactDifferenceForce(Force):
    """
    Tau-unabhängige Kraft nach der Exact Difference Method (He et al. 1998, Kupershtokh et al. 2009).
    Optional: Maskierung des Forcings auf bestimmten Gitterzellen.
    """

    def __init__(self, flow, acceleration, mask=None):
        self.flow = flow
        # acceleration: Tensor oder Liste (wird in Context-Tensor konvertiert)
        self.acceleration = flow.context.convert_to_tensor(acceleration)
        self.mask = mask  # [Nx, Ny, Nz] oder None

    def compute_feq(self, rho, u):
        """Berechne f_eq für gegebenes ρ und u."""
        e = self.flow.torch_stencil.e  # [q, d]
        w = self.flow.torch_stencil.w  # [q]
        cs2 = self.flow.torch_stencil.cs ** 2

        eu = torch.einsum("id,d...->i...", e, u)
        u2 = torch.sum(u ** 2, dim=0)
        feq = rho * w[:, None, None, None] * (
            1 + eu / cs2 +
            0.5 * (eu ** 2) / cs2 ** 2 -
            0.5 * u2 / cs2
        )
        return feq

    def source_term(self, u):
        """Berechne Source-Term S_i = f_eq(u + Δu) - f_eq(u)."""
        rho = self.flow.rho()

        # Δu = 0.5 * a
        du = 0.5 * self.acceleration
        du_broadcasted = append_axes(du, self.flow.torch_stencil.d)
        u_plus = u + du_broadcasted

        feq_plus = self.compute_feq(rho, u_plus)
        feq = self.compute_feq(rho, u)
        source = feq_plus - feq

        # ✅ Maskierung (optional)
        if self.mask is not None:
            source = source * self.mask[None, ...].to(source.dtype)

        return source

    def u_eq(self, flow=None):
        """Korrekturgeschwindigkeit für den Strömungsstart."""
        flow = self.flow if flow is None else flow
        return append_axes(0.5 * self.acceleration, flow.torch_stencil.d) / flow.rho()

    def ueq_scaling_factor(self):
        """Nur für Kompatibilität."""
        return 0.5

    def native_available(self) -> bool:
        return False

    def native_generator(self):
        return None
