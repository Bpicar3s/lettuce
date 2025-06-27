""""Lattice Boltzmann Solver"""
from timeit import default_timer as timer
from lettuce import (
    LettuceException, get_default_moment_transform, BGKInitialization, ExperimentalWarning, torch_gradient
)
import torch
from lettuce.simulation import Simulation

__all__ = ["SimulationReducedTGV"]

class SimulationReducedTGV(Simulation):
    def __init__(self, flow, lattice, collision, streaming=None):
        super(SimulationReducedTGV, self).__init__(flow, lattice, collision, streaming)
        self.u_initial = self.lattice.u(self.f)
        self.p_initial = self.lattice.rho(self.f)

    def step(self, num_steps):
        """Take num_steps stream-and-collision steps and return performance in MLUPS."""
        start = timer()

        if self.i == 0:
            self._report()
        for _ in range(num_steps):
            self.i += 1

            for boundary in self._boundaries:
                self.f = boundary(self.f)

            self.f = self.streaming(self.f)
            # Perform the collision routine everywhere, expect where the no_collision_mask is true
            self.f = torch.where(self.no_collision_mask, self.f, self.collision(self.f))
            #for boundary in self._boundaries:
            #    self.f = boundary(self.f)

            self._report()

        end = timer()
        seconds = end - start
        num_grid_points = self.lattice.rho(self.f).numel()
        mlups = num_steps * num_grid_points / 1e6 / seconds
        return mlups

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
