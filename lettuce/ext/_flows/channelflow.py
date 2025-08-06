from typing import Union, List, Optional
import numpy as np
import torch

from lettuce import UnitConversion, Flow, Context
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._boundary.bounce_back_boundary import BounceBackBoundary

from lettuce.ext._flows import ExtFlow

class ChannelFlow3D(ExtFlow):
    def __init__(self, context: Context,
                 resolution: Union[int, List[int]],
                 reynolds_number: float,
                 mach_number: float,
                 bbtype,
                 stencil: Optional['Stencil'] = None,
                 equilibrium: Optional['Equilibrium'] = None):

        self.h = resolution if isinstance(resolution, int) else resolution[1] // 2
        self._mask = None  # erst nach resolution verfügbar
        super().__init__(context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)
        self.mask_top = None
        self.mask_bottom = None
        self.bbtype = bbtype

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
            characteristic_length_lu=2*h,
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

    # In Ihrer neuen Klasse "ChannelFlow3D"

    def initial_pu(self):
        """
        Initialisierung des Kanalflusses nach Nathen et al. (2018):
        - u(y) = u_char * (y/H)^{1/7} als Anfangsprofil (nur u-Komponente)
        - plus normalverteilte Störungen auf u, v, w (alle Richtungen)
        """


        # Charakteristische Geschwindigkeit aus physikalischen Einheiten
        # Kanalhöhe
        # Kanalparameter
        ny = self.resolution[1]
        nx, _, nz = self.resolution
        device = self.context.device
        dtype = self.context.dtype

        # Gitterkoordinaten von 0 bis 1 (unten bis oben)
        y = torch.linspace(0, 1, ny, device=device, dtype=dtype)

        # Symmetrisches 1/7-Gesetz: max. in der Mitte, 0 an den Rändern
        u_char = 1.0
        u_y = u_char * torch.pow(y * (1 - y), 1 / 7)

        # Geschwindigkeit u(y) in x-Richtung setzen
        u = torch.zeros((3, nx, ny, nz), device=device, dtype=dtype)
        u[0] = u_y.view(1, ny, 1).expand(nx, ny, nz)

        # Normalverteilte Störung auf alle drei Komponenten hinzufügen
        u += 0.05 * torch.randn_like(u)

        # Randgeschwindigkeit (Wände) auf 0 setzen (no-slip)
        u[:, :, 0, :] = 0.0
        u[:, :, -1, :] = 0.0

        # Dichtefeld: überall rho = 1
        rho = torch.ones((nx, ny, nz), device=device, dtype=dtype)

        # Rückgabe im Format: shape(f) = [q, nx, ny, nz]
        return rho[None, ...], u

    @property
    def boundaries(self):

        shape = self.resolution
        self.mask_bottom = torch.zeros(shape, dtype=torch.bool, device=self.context.device)
        self.mask_bottom[:, 0, :] = True
        self.mask_top = torch.zeros(shape, dtype=torch.bool, device=self.context.device)
        self.mask_top[:, -1, :] = True

        if self.bbtype == "wallfunction":

            wfb_bottom = WallFunction(mask=self.mask_bottom, stencil=self.stencil, h=self.h, context=self.context,
                                      wall='bottom')
            wfb_top = WallFunction(mask=self.mask_top, stencil=self.stencil, h=self.h, context=self.context, wall='top')

        elif self.bbtype == "fullway":


            wfb_bottom=BounceBackBoundary(mask = self.mask_top)
            wfb_top=BounceBackBoundary(mask = self.mask_bottom)
        return [wfb_bottom, wfb_top]
