
import torch
from . import Force
from .guo import Guo
from lettuce.ext._boundary.wallfunction import compute_wall_quantities

__all__ = ['AdaptiveForce']

class AdaptiveForce:
    def __init__(self, flow, context, target_u_m_lu,
                 wall_bottom, wall_top,
                 global_ux_reporter,
                 base_lbm_tau_lu):
        self.flow = flow
        self.context = context
        self.u_m = target_u_m_lu
        self.wall_bottom = wall_bottom
        self.wall_top = wall_top
        self.global_ux = global_ux_reporter
        self.H = flow.h
        self.base_lbm_tau = base_lbm_tau_lu
        self.ueq_scaling_factor = 0.5
        self.last_force_lu = context.convert_to_tensor([0.0] * self.flow.stencil.d)

    def compute_force(self):
        utau_b_lu, y_plus, re_tau = compute_wall_quantities(flow = self.flow, dy=0.5, is_top=True)
        utau_t_lu, y_plus, re_tau = compute_wall_quantities(flow = self.flow, dy=0.5, is_top=False)

        utau_mean_lu = 0.5 * (utau_b_lu.mean() + utau_t_lu.mean())
        ux_mean_lu = self.global_ux()
        Fx_lu = (utau_mean_lu ** 2) / self.H + (self.u_m - ux_mean_lu) * (self.u_m / self.H)
        self.last_force_lu = self.context.convert_to_tensor([Fx_lu] + [0.0] * (self.flow.stencil.d - 1))
        return self.last_force_lu

    def __call__(self, u_field_lu, f):
        self.compute_force()
        guo_force = Guo(flow  =self.flow, tau=self.flow.units.relaxation_parameter_lu, acceleration=self.last_force_lu)
        return guo_force.source_term(u_field_lu)

    def source_term(self, u_field_lu):
        return self.__call__(u_field_lu, None)

    def u_eq(self, f):
        rho = self.flow.rho()
        index = [Ellipsis] + [None] * self.flow.stencil.d
        denom = torch.where(rho < 1e-10, torch.tensor(1e-10, device=rho.device, dtype=rho.dtype), rho)
        return self.ueq_scaling_factor * self.last_force_lu[index] / denom
