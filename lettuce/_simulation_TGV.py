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



    def initialize_f_neq(self):
        """Initialize the distribution function values. The f^(1) contributions are approximated by finite differences.
        See Krüger et al. (2017).
        """
        rho = self.lattice.rho(self.f)
        u = self.lattice.u(self.f)

        dx = self.flow.units.convert_length_to_pu(1.0)
        nges=u.size()[1]

        u_new = torch.zeros(3, nges + 6, nges + 6, nges + 6, device=self.lattice.device, dtype=self.lattice.dtype)

        u_new[:, 3:-3, 3:-3, 3:-3] = u

        u_new[:, 0:3, 3:-3, 3:-3] = torch.flip(u[:, 0:3, :, :], [1])
        u_new[0, 0:3, 3:-3, 3:-3] *= -1 #* u_new[0, 0:3, 3:-3, 3:-3]

        u_new[0, -3:, 3:-3, 3:-3] = -1 * torch.flip(torch.transpose(u[1, :, -3:, :], 0, 1), [0])
        u_new[1, -3:, 3:-3, 3:-3] = torch.flip(torch.transpose(u[0, :, -3:, :], 0, 1), [0])
        u_new[2, -3:, 3:-3, 3:-3] = torch.flip(torch.transpose(u[2, :, -3:, :], 0, 1), [0])

        u_new[:, 3:-3, 0:3, 3:-3] = torch.flip(u[:, :, 0:3, :], [2])
        u_new[1, 3:-3, 0:3, 3:-3] *= -1 #* u_new[1, 3:-3, 0:3, 3:-3]

        u_new[0, 3:-3, -3:, 3:-3] = torch.flip(torch.transpose(u[1, -3:, :, :], 0, 1), [1])
        u_new[1, 3:-3, -3:, 3:-3] = -1 * torch.flip(torch.transpose(u[0, -3:, :, :], 0, 1), [1])
        u_new[2, 3:-3, -3:, 3:-3] = torch.flip(torch.transpose(u[2, -3:, :, :], 0, 1), [1])

        u_new[:, 3:-3, 3:-3, -3:] = torch.flip(u[:, :, :, -3:], [3])
        u_new[2, 3:-3, 3:-3, -3:] *= -1 #* u_new[2, 3:-3, 3:-3, -3:]

        u_new[0, 3:-3, 3:-3, 0:3] = torch.flip(torch.transpose(u[1, :, :, 0:3], 0, 1), [2])
        u_new[1, 3:-3, 3:-3, 0:3] = torch.flip(torch.transpose(u[0, :, :, 0:3], 0, 1), [2])
        u_new[2, 3:-3, 3:-3, 0:3] = -1 * torch.flip(torch.transpose(u[2, :, :, 0:3], 0, 1), [2])

        #u_grad = torch.stack([torch_gradient(u_new[i], dx=dx, order=6) for i in range(self.lattice.D)])
        gradients = []

        # Compute gradients for each component of u_new
        for i in range(self.lattice.D):
            grad = torch_gradient(u_new[i], dx=1, order=6)[None, ...]
            # Trim the edges and add to the list
            trimmed_grad = grad[:, :, 3:-3, 3:-3, 3:-3]
            gradients.append(trimmed_grad)

        # Stack all gradients along the new dimension
        S = torch.cat(gradients)

        Pi_1 = 1.0 * self.flow.units.relaxation_parameter_lu * rho * S / self.lattice.cs ** 2
        Q = (torch.einsum('ia,ib->iab', [self.lattice.e, self.lattice.e])
             - torch.eye(self.lattice.D, device=self.lattice.device, dtype=self.lattice.dtype) * self.lattice.cs ** 2)
        Pi_1_Q = self.lattice.einsum('ab,iab->i', [Pi_1, Q])
        fneq = self.lattice.einsum('i,i->i', [self.lattice.w, Pi_1_Q])

        feq = self.lattice.equilibrium(rho, u)
        self.f = feq + fneq


