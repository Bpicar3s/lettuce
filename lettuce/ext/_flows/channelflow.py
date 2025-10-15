from typing import Union, List, Optional
import numpy as np
import torch

from lettuce import UnitConversion, Flow, Context, Stencil, Equilibrium
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._boundary.bounce_back_boundary import BounceBackBoundary

from lettuce.ext._flows import ExtFlow

class ChannelFlow3D(ExtFlow):
    def __init__(self, context: Context,
                 resolution: Union[int, List[int]],
                 reynolds_number: float,
                 mach_number: float,
                 bbtype: str,
                 stencil: Optional[Stencil] = None,
                 equilibrium: Optional[Equilibrium] = None,
                 random_seed: int = 42):  # <-- NEU: Seed als Parameter
        """
        Initialisiert den 3D-Kanalfluss.

        Args:
            ... (andere Parameter)
            random_seed: Seed für den Zufallszahlengenerator, um eine
                         reproduzierbare Initialisierung zu gewährleisten.
        """
        self.h = resolution if isinstance(resolution, int) else resolution[1] // 2
        self._mask = None
        self.random_seed = random_seed  # <-- NEU: Seed speichern
        super().__init__(context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)
        self.mask_top = None
        self.mask_bottom = None
        self.bbtype = bbtype

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional[Stencil] = None) -> List[int]:
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

    def initial_pu(self):
        """
        Erzeugt eine komplexe Anfangsströmung, um die Transition zur Turbulenz
        zu beschleunigen. Verwendet einen Seed für reproduzierbare Ergebnisse.
        """
        # --- NEU: Zufallszahlengenerator mit dem Seed initialisieren ---
        rng = np.random.default_rng(self.random_seed)

        # Gitter und Auflösung aus der Klasse holen
        xg, yg, zg = self.grid
        nx, ny, nz = self.resolution

        # --- 1. Basisprofil & Dichte ---
        p = np.ones_like(xg)[None, ...]
        u = np.zeros((3, nx, ny, nz))
        y_normalized = yg / (ny - 1)
        u_base = 4 * y_normalized * (1 - y_normalized)
        u[0] = u_base * (1 - self.mask.astype(float))

        # --- 2. Sinusmoden-Störung (deterministisch) ---
        A_sin = 0.05  # 5% Amplitude
        Lx, Ly, Lz = xg.max(), yg.max(), zg.max()
        sinus_modes = [(1, 1, 1), (2, 2, 3), (3, 2, 1)]

        for kx, ky, kz in sinus_modes:
            # --- GEÄNDERT: rng.random() statt np.random.rand() ---
            phase = 2 * np.pi * rng.random()
            mode = np.sin(2 * np.pi * (kx * xg / Lx + ky * yg / Ly + kz * zg / Lz) + phase)
            envelope = y_normalized * (1 - y_normalized)
            u[0] += A_sin * mode * envelope

        # --- 3. Divergenzfreie Störung mit Vektorpotential ψ (stochastisch) ---
        A_psi = 0.1  # Amplitude der Störung
        # --- GEÄNDERT: rng.random() statt np.random.rand() ---
        # Beachten Sie die leicht andere Syntax: rng.random() erwartet die Shape als Tupel
        random_psi = (rng.random((3, nx, ny, nz)) - 0.5) * 2

        # FFT-Filterung für glatte Wirbel
        k0 = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        psi_filtered = np.empty_like(random_psi)
        for d in range(3):
            psi_hat = np.fft.fftn(random_psi[d])
            kx = np.fft.fftfreq(nx).reshape(-1, 1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1, 1)
            kz = np.fft.fftfreq(nz).reshape(1, 1, -1)
            kabs = np.sqrt((kx * nx) ** 2 + (ky * ny) ** 2 + (kz * nz) ** 2)

            filter_mask = np.exp(-kabs / (0.15 * k0))
            psi_hat *= filter_mask
            psi_hat[0, 0, 0] = 0
            psi_filtered[d] = np.real(np.fft.ifftn(psi_hat))

        # Curl(ψ) berechnen: u_psi = ∇ × ψ
        u_psi = np.zeros_like(u)
        u_psi[0] = np.gradient(psi_filtered[2], axis=1) - np.gradient(psi_filtered[1], axis=2)
        u_psi[1] = np.gradient(psi_filtered[0], axis=2) - np.gradient(psi_filtered[2], axis=0)
        u_psi[2] = np.gradient(psi_filtered[1], axis=0) - np.gradient(psi_filtered[0], axis=1)

        # Normieren und mit Amplitude skalieren
        umax_psi = np.max(np.sqrt(np.sum(u_psi ** 2, axis=0)))
        if umax_psi > 0:
            u_psi *= A_psi / umax_psi

        # --- 4. Überlagerung & Randbedingungen ---
        u += u_psi
        u[:, :, 0, :] = 0.0
        u[:, :, -1, :] = 0.0

        # --- 5. Konvertierung zu PyTorch Tensoren ---
        p_tensor = torch.tensor(p, dtype=self.context.dtype)
        u_tensor = torch.tensor(u, dtype=self.context.dtype)

        return p_tensor, u_tensor

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
            boundary = [wfb_bottom, wfb_top]
        elif self.bbtype == "fullway":
            wfb_bottom = BounceBackBoundary(mask=self.mask_top)
            wfb_top = BounceBackBoundary(mask=self.mask_bottom)
            boundary = [wfb_bottom, wfb_top]
        elif self.bbtype is None:
            boundary = []
        return boundary
