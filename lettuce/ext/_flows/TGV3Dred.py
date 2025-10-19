
from lettuce.ext._boundary.ReducedTGV_boundary import newsuperTGV3D
from lettuce.util import torch_gradient
from ... import UnitConversion
from . import ExtFlow
import torch
import numpy as np
from typing import Union, List, Optional

class SuperReducedTaylorGreenVortex3D(ExtFlow):
    def __init__(self, context, resolution, reynolds_number, mach_number,
                 stencil=None, equilibrium=None, initialize_fneq_TGV=True):
        self.initialize_fneq_TGV = initialize_fneq_TGV

        if stencil is None:
            raise ValueError("Please provide a stencil (e.g., D3Q27)")
        self.stencil = stencil() if callable(stencil) else stencil

        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, self.stencil, equilibrium)


    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * 3
        else:
            assert len(resolution) == 3
            return resolution

    def make_units(self, reynolds_number, mach_number, resolution) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=resolution[0],
            characteristic_length_pu=np.pi / 2,
            characteristic_velocity_pu=1)
    @property
    def grid(self):
        x, dx = np.linspace(0, np.pi / 2, num=self.resolution[0], endpoint=False, retstep=True)
        y, dy = np.linspace(0, np.pi / 2, num=self.resolution[1], endpoint=False, retstep=True)
        z, dz = np.linspace(np.pi / 2, np.pi, num=self.resolution[2], endpoint=False, retstep=True)
        x += dx / 2
        y += dy / 2
        z += dz / 2
        xyz = tuple(torch.tensor(arr, device=self.context.device, dtype=self.context.dtype)
                    for arr in np.meshgrid(x, y, z, indexing='ij'))
        return xyz

    def analytic_solution(self):

        grid = self.grid
        nu = self.context.convert_to_tensor(self.units.viscosity_pu)

        u = torch.stack(
            [torch.sin(grid[0])
             * torch.cos(grid[1])
             * torch.cos(grid[2]),
             -torch.cos(grid[0])
             * torch.sin(grid[1])
             * torch.cos(grid[2]),
             torch.zeros_like(grid[0])])
        p = torch.stack(
            [1 / 16. * (torch.cos(2 * grid[0]) + torch.cos(2 * grid[1]))
             * (torch.cos(2 * grid[2]) + 2)])
        return p, u

    def initial_pu(self):
        return self.analytic_solution()
    @property
    def boundaries(self) -> List['Boundary']:
        return [newsuperTGV3D(self)]

