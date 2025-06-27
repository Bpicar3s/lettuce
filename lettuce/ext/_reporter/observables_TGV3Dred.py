"""
Observables from the reduced TGV 3D.
The Dissipation calculated by the shear moments and enstrophy uses the torch_gradient function which only works for periodic flows.
Therefore the Velocities are swapped to calculate still "periodic".
"""

import torch
from lettuce.util import torch_gradient


__all__ = ["Dissipation_TGV"]


class Observable:
    def __init__(self, lattice, flow):
        self.lattice = lattice
        self.flow = flow

    def __call__(self, f):
        raise NotImplementedError

class Dissipation_TGV(Observable):

    def __init__(self, lattice, flow, no_grad=True):
        super(Dissipation_TGV, self).__init__(lattice, flow)

    def __call__(self, f):
        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f))
        nges = u.size()[1]

        u = self.flow.units.convert_velocity_to_pu(self.lattice.u(f)).clone()
        dx = self.flow.units.convert_length_to_pu(1.0)
        nu = self.flow.units.viscosity_pu

        u_new = torch.zeros(3, nges + 6, nges + 6, nges + 6)

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

        u_grad = torch.stack([torch_gradient(u_new[i], dx=dx, order=6) for i in range(self.lattice.D)])
        u_ij = u_grad[:, :, 3:-3, 3:-3, 3:-3]
        # dissipation=nu*1/nges**3*torch.sum(torch.square(u_ij))
        grad_u0 = u_ij[:, 0, :, :, :]
        grad_u1 = u_ij[:, 1, :, :, :]
        grad_u2 = u_ij[:, 2, :, :, :]
        vorticity = torch.sum((grad_u0[1] - grad_u1[0]) * (grad_u0[1] - grad_u1[0]) + (grad_u2[1] - grad_u1[2]) * (
                    grad_u2[1] - grad_u1[2]) + (grad_u0[2] - grad_u2[0]) * (grad_u0[2] - grad_u2[0]))
        # u_ij = torch.stack([torch_gradient(u[i], dx=dx, order=6) for i in range(self.lattice.D)])
        s_ij = 0.5 * (u_ij + torch.transpose(u_ij, 0, 1))
        dissipation = 2 * nu * torch.mean((s_ij ** 2).sum(0).sum(0))

        enstrophy = nu * vorticity * dx ** self.lattice.D
        return torch.stack([dissipation, enstrophy])