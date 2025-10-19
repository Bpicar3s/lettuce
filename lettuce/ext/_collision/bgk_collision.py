import torch

from typing import Optional, AnyStr

from ... import Flow, Collision
from ...cuda_native.ext import NativeBGKCollision
from .. import Force

__all__ = ['BGKCollision']


class BGKCollision(Collision):
    def __init__(self, tau, force: Optional['Force'] = None):
        self.tau = tau
        self.force = force

    def __call__(self, flow: 'Flow') -> torch.Tensor:
        u_eq = 0 if self.force is None else self.force.u_eq(flow)
        u = flow.u() + u_eq
        feq = flow.equilibrium(flow, u=u)
        si = self.force.source_term(u) if self.force is not None else 0
        return flow.f - 1.0 / self.tau * (flow.f - feq) + si

    def set_on_simulation(self, simulation: 'Simulation'):
        """
        Wird vom Simulator aufgerufen. Hier registriert die Kollision
        ihre Kraft-Update-Funktion automatisch als Callback.
        """
        # Prüft, ob eine "schlaue" Kraft vorhanden ist
        if self.force is not None and hasattr(self.force, 'update_native_force_on_simulation'):
            # 1. Sagt der Kraft, welches Simulationsobjekt sie steuern soll
            self.force.set_on_simulation(simulation)
            # 2. Hängt die Update-Funktion der Kraft an die To-Do-Liste des Simulators
            simulation.callbacks.append(self.force.update_native_force_on_simulation)

    def name(self) -> AnyStr:
        if self.force is not None:
            return f"{self.__class__.__name__}_{self.force.__class__.__name__}"
        return self.__class__.__name__

    def native_available(self) -> bool:
        return self.force is None or self.force.native_available()

    def native_generator(self, index: int) -> 'NativeCollision':
        if self.force is not None:
            return NativeBGKCollision(index, self.force.native_generator())  # ✅
        return NativeBGKCollision(index)

