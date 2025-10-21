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

    def name(self) -> AnyStr:
        if self.force is not None:
            return f"{self.__class__.__name__}_{self.force.__class__.__name__}"
        return self.__class__.__name__

    def native_available(self) -> bool:
        return self.force is None or self.force.native_available()

    def native_generator(self, index: int) -> 'NativeCollision':
        native_force = None
        if self.force is not None:
            # Ruft die Methode aus Schritt 1 auf
            native_force = self.force.native_generator()

            # Übergibt 'index' und 'native_force' (das 'self' oder 'None' ist)
        return NativeBGKCollision(index, native_force)



