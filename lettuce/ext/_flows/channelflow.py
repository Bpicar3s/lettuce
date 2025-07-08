from typing import Union, List, Optional
import numpy as np
import torch

from lettuce import UnitConversion, Flow, Context
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._flows import ExtFlow

class ChannelFlow3D(ExtFlow):
    def __init__(self, context: Context,
                 resolution: Union[int, List[int]],
                 reynolds_number: float,
                 mach_number: float,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):
        self.h = resolution if isinstance(resolution, int) else resolution[1] // 2
        self._mask = None  # erst nach resolution verfügbar
        super().__init__(context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)
        self.mask_top = None
        self.mask_bottom = None

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            h = resolution
            return [int(2 * np.pi * h), 2 * h, int(np.pi * h)]
        assert len(resolution) == 3, "ChannelFlow3D erwartet 3D-Auflösung!"
        return resolution

    def make_units(self, reynolds_number, mach_number, resolution: List[int]) -> UnitConversion:
        h = resolution[1] // 2
        return UnitConversion(
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=h,
            characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    @property
    def mask(self):
        if self._mask is None:
            self._mask = np.zeros(shape=tuple(self.resolution), dtype=bool)
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray)
        assert m.shape == tuple(self.resolution)
        self._mask = m.astype(bool)

    @property
    def grid(self):
        x = np.linspace(0, self.resolution[0], self.resolution[0], endpoint=False)
        y = np.linspace(0, self.resolution[1], self.resolution[1], endpoint=False)
        z = np.linspace(0, self.resolution[2], self.resolution[2], endpoint=False)
        return np.meshgrid(x, y, z, indexing='ij')

    def initial_pu(self):
        xg, yg, zg = self.grid
        nx, ny, nz = self.resolution

        p = np.ones_like(xg)[None, ...]
        u = np.zeros((3, nx, ny, nz))

        y_norm = yg / yg.max()
        u_base = y_norm * (1 - y_norm)
        u[0] = u_base * (1 - self.mask.astype(float))

        # Störung
        A_sin = 0.5
        Lx, Ly, Lz = xg.max(), yg.max(), zg.max()
        modes = [(1, 1, 1), (2, 2, 3), (3, 2, 1)]

        for kx, ky, kz in modes:
            phase = 2 * np.pi * np.random.rand()
            mode = np.sin(2 * np.pi * (kx * xg / Lx + ky * yg / Ly + kz * zg / Lz) + phase)
            envelope = y_norm * (1 - y_norm)
            u[0] += A_sin * mode * envelope

        # Divergenzfreie Störung (wie in deiner Version)
        # (Optional: hier kannst du psi-Feld, Gewichtung, Filter etc. hinzufügen)

        u[:, :, 0, :] = 0.0
        u[:, :, -1, :] = 0.0

        return torch.tensor(p), torch.tensor(u)

    @property
    def boundaries(self):
        shape = self.resolution
        self.mask_bottom = torch.zeros(shape, dtype=torch.bool, device=self.context.device)
        self.mask_bottom[:, 0, :] = True
        self.mask_top = torch.zeros(shape, dtype=torch.bool, device=self.context.device)
        self.mask_top[:, -1, :] = True



        wfb_bottom = WallFunction(mask=self.mask_bottom, stencil = self.stencil, h=self.h, context=self.context, wall='bottom')
        wfb_top = WallFunction(mask=self.mask_top, stencil = self.stencil, h=self.h, context=self.context, wall='top')
        return [wfb_bottom, wfb_top]
