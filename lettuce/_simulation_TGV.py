""""Lattice Boltzmann Solver"""
from timeit import default_timer as timer
from lettuce import (torch_gradient
)
import torch
from _simulation import Simulation

__all__ = ["SimulationReducedTGV"]

from timeit import default_timer as timer
import torch
from _simulation import Simulation
from typing import List


__all__ = ["SimulationReducedTGV"]

class SimulationReducedTGV(Simulation):
    def __init__(self, flow: Flow, collision: Collision, reporter: List[Reporter]):
        super().__init__(flow, collision, reporter)
        self.u_initial = flow.u()
        self.p_initial = flow.rho()

    def __call__(self, num_steps: int):
        beg = timer()

        if self.flow.i == 0:
            self._report()

        for _ in range(num_steps):

            for boundary in self.boundaries[1:]:
                self.flow.f = boundary(self.flow)

            self._stream()
            self._collide()
            self.flow.i += 1
            self._report()

        end = timer()
        return num_steps * self.flow.rho().numel() / 1e6 / (end - beg)
