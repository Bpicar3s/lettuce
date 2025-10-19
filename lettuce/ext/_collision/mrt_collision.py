import torch

from ... import Flow, Collision
from typing import Union, List, Optional


__all__ = ['MRTCollision']


class MRTCollision(Collision):
    """
    Multiple relaxation time collision operator

    This is an MRT operator in the most general sense of the word.
    The transform does not have to be linear and can, e.g., be any moment or
    cumulant transform.
    """

    def __init__(self, transform: 'Transform', relaxation_parameters: list,
                 context: 'Context', force: Optional['Force'] = None):
        self.transform = transform
        self.relaxation_parameters = context.convert_to_tensor(relaxation_parameters)
        self.force = force  # ⬅️ neu

    def __call__(self, flow: 'Flow'):
        # 1. ggf. Geschwindigkeit mit u_eq verschieben
        u_eq = 0 if self.force is None else self.force.u_eq(flow)
        u = flow.u() + u_eq

        # 2. MRT-Kollision im Momentenraum
        m = self.transform.transform(flow.f)
        meq = self.transform.equilibrium(m, flow, u=u)
        m = m - torch.einsum("q...,q...->q...", [1 / self.relaxation_parameters, m - meq])
        f = self.transform.inverse_transform(m)

        # 3. Forcing-Term (Guo) addieren
        if self.force is not None:
            si = self.force.source_term(u)
            f = f + si

        return f

    def native_available(self) -> bool:
        return False

    def native_generator(self, index: int) -> 'NativeCollision':
        pass
