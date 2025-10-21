import sys
from abc import ABC, abstractmethod
from typing import Optional

import torch
import numpy as np

from ... import Reporter, Flow
from ...util import torch_gradient
from packaging import version
from lettuce.ext._boundary.wallfunction import compute_wall_quantities
__all__ = ['Observable', 'ObservableReporter', 'MaximumVelocity',
           'IncompressibleKineticEnergy', 'Enstrophy', 'EnergySpectrum',
           'Mass', 'GlobalMeanUXReporter', 'WallQuantities', 'AdaptiveAcceleration', 'WallfunctionReporter']


class Observable(ABC):
    def __init__(self, flow: 'Flow'):
        self.context = flow.context
        self.flow = flow

    @abstractmethod
    def __call__(self, f: Optional[torch.Tensor] = None):
        ...


class MaximumVelocity(Observable):
    """Maximum velocitiy"""

    def __call__(self, f: Optional[torch.Tensor] = None):
        return torch.norm(self.flow.u_pu, dim=0).max()


class IncompressibleKineticEnergy(Observable):
    """Total kinetic energy of an incompressible flow."""

    def __call__(self, f: Optional[torch.Tensor] = None):
        dx = self.flow.units.convert_length_to_pu(1.0)
        kinE = self.flow.units.convert_incompressible_energy_to_pu(
            torch.sum(self.flow.incompressible_energy()))
        kinE *= dx ** self.flow.stencil.d
        return kinE


class Enstrophy(Observable):
    """The integral of the vorticity

    Notes
    -----
    The function only works for periodic domains
    """

    def __call__(self, f: Optional[torch.Tensor] = None):
        u0 = self.flow.units.convert_velocity_to_pu(self.flow.u()[0])
        u1 = self.flow.units.convert_velocity_to_pu(self.flow.u()[1])
        dx = self.flow.units.convert_length_to_pu(1.0)
        grad_u0 = torch_gradient(u0, dx=dx, order=6)
        grad_u1 = torch_gradient(u1, dx=dx, order=6)
        vorticity = torch.sum((grad_u0[1] - grad_u1[0])
                              * (grad_u0[1] - grad_u1[0]))
        if self.flow.stencil.d == 3:
            u2 = self.flow.units.convert_velocity_to_pu(self.flow.u()[2])
            grad_u2 = torch_gradient(u2, dx=dx, order=6)
            vorticity += torch.sum(
                (grad_u2[1] - grad_u1[2]) * (grad_u2[1] - grad_u1[2])
                + ((grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
            )
        return vorticity * dx ** self.flow.stencil.d


class EnergySpectrum(Observable):
    """The kinetic energy spectrum"""

    def __init__(self, flow: Flow):
        super(EnergySpectrum, self).__init__(flow)
        self.dx = self.flow.units.convert_length_to_pu(1.0)
        self.dimensions = self.flow.resolution
        frequencies = [self.context.convert_to_tensor(
            np.fft.fftfreq(dim, d=1 / dim)
        ) for dim in self.dimensions]
        wavenumbers = torch.stack(torch.meshgrid(*frequencies, indexing='ij'))
        wavenorms = torch.norm(wavenumbers, dim=0)

        if self.flow.stencil.d == 3:
            self.norm = self.dimensions[0] * np.sqrt(2 * np.pi) / self.dx ** 2
        else:
            self.norm = self.dimensions[0] / self.dx

        self.wavenumbers = torch.arange(int(torch.max(wavenorms)))
        self.wavemask = (
                (wavenorms[..., None] > self.wavenumbers.to(
                    dtype=self.context.dtype, device=self.context.device)
                 - 0.5) &
                (wavenorms[..., None] <= self.wavenumbers.to(
                    dtype=self.context.dtype, device=self.context.device)
                 + 0.5)
        )

    def __call__(self, f: Optional[torch.Tensor] = None):
        u = self.flow.u()
        return self.spectrum_from_u(u)

    def spectrum_from_u(self, u):
        u = self.flow.units.convert_velocity_to_pu(u)
        ekin = self._ekin_spectrum(u)
        ek = ekin[..., None] * self.wavemask.to(dtype=self.context.dtype)
        ek = ek.sum(torch.arange(self.flow.stencil.d).tolist())
        return ek

    def _ekin_spectrum(self, u):
        """distinguish between different torch versions"""
        torch_ge_18 = (version.parse(torch.__version__) >= version.parse(
            "1.8.0"))
        if torch_ge_18:
            return self._ekin_spectrum_torch_ge_18(u)
        else:
            return self._ekin_spectrum_torch_lt_18(u)

    def _ekin_spectrum_torch_lt_18(self, u):
        zeros = torch.zeros(self.dimensions, dtype=self.context.dtype,
                            device=self.context.device)[..., None]
        uh = (torch.stack(
            [torch.fft(torch.cat((u[i][..., None], zeros),
                                 self.flow.stencil.d),
                       signal_ndim=self.flow.stencil.d)
             for i in range(self.flow.stencil.d)
             ]) / self.norm)
        ekin = torch.sum(0.5 * (uh[..., 0] ** 2 + uh[..., 1] ** 2), dim=0)
        return ekin

    def _ekin_spectrum_torch_ge_18(self, u):
        uh = (torch.stack([
            torch.fft.fftn(u[i], dim=tuple(torch.arange(self.flow.stencil.d)))
            for i in range(self.flow.stencil.d)
        ]) / self.norm)
        ekin = torch.sum(0.5 * (uh.imag ** 2 + uh.real ** 2), dim=0)
        return ekin


class Mass(Observable):
    """Total mass in lattice units.

    Parameters
    ----------
    no_mass_mask : torch.Tensor
        Boolean mask that defines grid points
        which do not count into the total mass (e.g. bounce-back boundary).
    """

    def __init__(self, flow: Flow, no_mass_mask=None):
        super(Mass, self).__init__(flow)
        self.mask = no_mass_mask

    def __call__(self, f: Optional[torch.Tensor] = None):
        mass = f[..., 1:-1, 1:-1].sum()
        if self.mask is not None:
            mass -= (f * self.mask.to(dtype=torch.float)).sum()
        return mass


class ObservableReporter(Reporter):
    """A reporter that prints an observable every few iterations.

    Examples
    --------
    Create an Enstrophy reporter.

    >>> from lettuce import TaylorGreenVortex3D, Enstrophy, D3Q27, Context
    >>> context = Context(device=torch.device("cpu"))
    >>> flow = TaylorGreenVortex(context, 50, 300, 0.1, D3Q27())
    >>> simulation = ...
    >>> enstrophy = Enstrophy(flow)
    >>> reporter = ObservableReporter(enstrophy, interval=10)
    >>> simulation.reporter.append(reporter)
    """

    def __init__(self, observable, interval=1, out=sys.stdout):
        super().__init__(interval)
        self.observable = observable
        self.out = [] if out is None else out
        self._parameter_name = observable.__class__.__name__

    def __call__(self, simulation: 'Simulation'):
        if simulation.flow.i % self.interval == 0:
            observed = self.observable.context.convert_to_ndarray(
                self.observable(simulation.flow.f))
            assert len(observed.shape) < 2
            if len(observed.shape) == 0:
                observed = [observed.item()]
            else:
                observed = observed.tolist()
            entry = ([simulation.flow.i,
                      simulation.units.convert_time_to_pu(simulation.flow.i)]
                     + observed)
            if isinstance(self.out, list):
                self.out.append(entry)
            else:
                print(*entry, file=self.out)


class WallQuantities(Observable):
    def __init__(self, mask, wall, flow, context = None):
        self.wall = wall
        self.mask = mask
        self.flow = flow
        self.context = context
        # Ensure the boundary object has the expected attributes, initialized to tensors
        # (This is already handled by your WallFunctionBoundaryTest __init__)

    def __call__(self, f: Optional[torch.Tensor] = None):

        u_tau, y_plus, re_tau = compute_wall_quantities(flow = self.flow, dy=torch.tensor(1, device=self.flow.f.device, dtype=self.flow.f.dtype), is_top=True if self.wall == "top" else False)
        print("y+:", y_plus.mean(),"Re_tau:", re_tau.mean())
        return torch.stack([
            u_tau.mean(),
            y_plus.mean(),
            re_tau.mean(),
        ])

class GlobalMeanUXReporter(Observable):
    def __call__(self, f: Optional[torch.Tensor] = None):
        u_field_lu = self.flow.u()  # u_field_lu: shape (3, Nx, Ny, Nz)
        u_x_spatial = u_field_lu[0]     # Nur die x-Komponente
        current_mean_ux_lu = torch.mean(u_x_spatial)
        return current_mean_ux_lu

from lettuce.ext._boundary.wallfunction import compute_wall_quantities
class AdaptiveAcceleration(Observable):
    """
    Berechnet periodisch die Beschleunigung a(t) in LU und
    schreibt sie direkt in force.acceleration (ExactDifferenceForce).
    Wird vom ObservableReporter alle 'interval' Schritte aufgerufen.
    """

    def __init__(self, flow, force_obj, target_mean_ux_lu,
                 context,
                 k_gain: float = 1.0):
        super().__init__(flow)
        self.force = force_obj
        self.target_mean_ux_lu = target_mean_ux_lu
        self.context = context
        self.k_gain = k_gain

        # interner Zustand (aktueller Beschleunigungsvektor in LU)
        self.current_accel = context.convert_to_tensor(
            [0.0] * flow.stencil.d, dtype=flow.f.dtype
        )

    @torch.no_grad()
    def compute_acceleration(self):
        """
        Neue a(t) berechnen und direkt in ExactDifferenceForce schreiben.
        """
        utau_b, _, _ = compute_wall_quantities(self.flow, dy=1, is_top=False)
        utau_t, _, _ = compute_wall_quantities(self.flow, dy=1, is_top=True)
        utau_mean = 0.5 * (utau_b.mean() + utau_t.mean())

        u_field = self.flow.u()
        ux_mean = torch.mean(u_field[0])

        H = max(float(self.flow.h), 1.0)
        Fx_base = (utau_mean ** 2) / H
        Fx_reg = self.k_gain * (self.target_mean_ux_lu - ux_mean) * (self.target_mean_ux_lu / H)
        Fx = (Fx_base + Fx_reg).to(device=self.context.device, dtype=self.flow.f.dtype)

        # ================================================================
        # DEBUG-PRINT HINZUFÜGEN:
        print(f"[Schritt {self.flow.i}] REPORTER LÄUFT. "
              f"Berechnete Fx = {Fx.item():.6e} "
              f"(Target_u: {self.target_mean_ux_lu:.3f}, "
              f"Actual_u: {ux_mean.item():.3f})")
        # ================================================================

        acc = torch.stack([Fx] + [torch.zeros_like(Fx)] * (self.flow.stencil.d - 1))
        self.current_accel = acc
        self.force.acceleration.copy_(acc)

        return acc

    def __call__(self, f=None):
        """
        Wird vom ObservableReporter aufgerufen.
        Wir berechnen hier a(t) neu und geben z. B. ax zurück (für Logging).
        """
        acc = self.compute_acceleration()
        # Rückgabe: ax als Skalar (für Reporter-CSV oder Print)
        return acc[0]

class WallfunctionReporter(Observable):
    def __init__(self, context, flow, collision_py, no_collision_mask, wfb_bottom, wfb_top):
        self.context = context
        self.flow = flow
        self.collision_py = collision_py
        self.no_collision_mask = no_collision_mask  # 🔗 übergeben von außen
        self.wfb_bottom = wfb_bottom
        self.wfb_top = wfb_top

    @torch.no_grad()
    def __call__(self, f):
        # (1) Python-Kollision nur auf Zellen ohne Maskierung (z. B. erste Fluidreihe)
        torch.where(torch.eq(self.no_collision_mask, 0),
                    self.collision_py(self.flow), self.flow.f,
                    out=self.flow.f)

        # (2) Danach Wandfunktionen aufrufen (lesen u und rho nach Collision)
        self.wfb_bottom(self.flow)
        self.wfb_top(self.flow)

        #torch.where(torch.eq(self.wfb_top.mask, 0),
        #            self.wfb_top(self.flow), self.flow.f, out=self.flow.f)
        #torch.where(torch.eq(self.wfb_bottom.mask, 0),
        #            self.wfb_bottom(self.flow), self.flow.f, out=self.flow.f)


        # (3) Optional Logging
        return torch.tensor([
            float(self.wfb_bottom.u_tau_mean),
            float(self.wfb_top.u_tau_mean)
        ], device=self.flow.f.device)
