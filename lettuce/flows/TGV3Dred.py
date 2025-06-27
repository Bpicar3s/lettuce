"""
Taylor-Green vortex in 2D and 3D.
"""

import numpy as np

from lettuce.unit import UnitConversion

from lettuce.boundary_TGV import newsuperTGV3D

from lettuce.flows import TaylorGreenVortex3D

class SuperReducedTaylorGreenVortex3D(TaylorGreenVortex3D):
    def __init__(self, resolution, reynolds_number, mach_number, lattice):
        self.resolution = resolution
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=resolution / (1/2*np.pi), characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )

    '''Initialization of the reduced TGV-Grid '''
    @property
    def grid(self):
        x,dx = np.linspace(0, np.pi/2, num=self.resolution, endpoint=False, retstep=True)
        y,dy = np.linspace(0, np.pi/2, num=self.resolution, endpoint=False, retstep=True)
        z,dz = np.linspace(np.pi/2, np.pi, num=self.resolution, endpoint=False, retstep=True)
        x += dx / 2
        y += dy / 2
        z += dz / 2
        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        boundary=newsuperTGV3D(lattice=self.units.lattice)
        return [boundary]
