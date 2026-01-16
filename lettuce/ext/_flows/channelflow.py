from typing import Union, List, Optional
import numpy as np
import torch

from lettuce import UnitConversion, Flow, Context, Stencil, Equilibrium
from lettuce.ext._boundary.wallfunction import WallFunction
from lettuce.ext._boundary.bounce_back_boundary import BounceBackBoundary

from lettuce.ext._flows import ExtFlow


import lettuce as lt
from lettuce import Boundary
from lettuce.cuda_native import NativeBoundary, DefaultCodeGeneration


# --- ANFANG: Code-Block zum Einfügen ---

class NativeDummyBoundary(NativeBoundary):
    """
    Eine leere Cuda-Boundary, die nichts tut.
    Sie existiert nur, um den Generator in den "Mask-Mode" zu zwingen.
    """

    def generate(self, reg: 'DefaultCodeCodeGeneration'):
        # Fügt einen C++-Kommentar hinzu, damit wir wissen, dass es funktioniert hat
        reg.pipe.append(f"// NativeDummyBoundary (Index {self.index}) - Aktiv.")

    @staticmethod
    def create(index: int):
        return NativeDummyBoundary(index)


class DummyBoundary(Boundary):
    """
    Die Python-Seite der Dummy-Boundary.
    Sie existiert nur, um flow.pre_boundaries zu füllen.
    """

    def __call__(self, flow):
        return flow.f  # Tut nichts

    def native_available(self) -> bool:
        return True  # Sagt "Ich habe eine Cuda-Version!"

    def native_generator(self, index: int) -> 'NativeBoundary':
        return NativeDummyBoundary.create(index)  # Gibt die Cuda-Klasse zurück

    # WICHTIG: Diese Funktion muss auch da sein
    def make_no_collision_mask(self, shape, context):
        """
        ERZEUGT DIE BOOLEAN MASKE FÜR DEINEN "TRICK"
        Gibt eine *boolesche* Maske zurück, die True ist,
        wo der Cuda-Kernel NICHT kollidieren soll.
        Simulation.__init__ wandelt das in die uint8 Index-Maske um.
        """
        print(">>> DummyBoundary.make_no_collision_mask() wird aufgerufen.")
        mask = torch.zeros(shape, dtype=torch.bool, device=context.device)

        # Dein "Trick": y=0,1 und y=-1,-2 überspringen
        mask[:, 0, :] = True
        mask[:, 1, :] = True
        mask[:, -1, :] = True
        mask[:, -2, :] = True

        return mask

    def make_no_streaming_mask(self, shape, context):
        """
        ERZEUGT DEN DUMMY-TENSOR FÜR NO-STREAMING
        Gibt einen uint8 Tensor zurück, der mit 0 gefüllt ist
        ("überall streamen"). Wird benötigt, weil der Cuda-Kernel
        im Mask-Mode diesen Tensor erwartet.
        """
        print(">>> DummyBoundary.make_no_streaming_mask() wird aufgerufen.")
        # Shape ist [q, *resolution]
        q = shape[0]  # Erster Dimension von f ist q
        resolution = shape[1:]

        # Erzeuge den Null-Tensor direkt hier
        dummy_mask = torch.zeros([q, *resolution], dtype=torch.bool, device = context.device)
        print("      Dummy 'no_streaming_mask' (uint8 Tensor) erstellt.")
        return dummy_mask
# --- ENDE: Code-Block zum Einfügen ---


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
            # Originale theoretische Werte
            lx = 2 * np.pi * h
            ly = 2 * h
            lz = np.pi * h

            # Aufrunden auf das nächste Vielfache von 8
            def round8(x):
                return int(np.ceil(x / 8) * 8)

            return [int(lx), int(ly), int(lz)]
              # [round8(lx), round8(ly), round8(lz)]
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

    def initial_pu2(self):
        """
        Init wie in Nathen et al. / Bespalko-Setup:
          u(z) = u_char * (z/H)^(1/7)
        + Gaussian Perturbations u', v', w' ~ N(0, sigma) mit sigma=5% (relativ zu u_char)
        """

        rng = np.random.default_rng(self.random_seed)

        xg, yg, zg = self.grid
        nx, ny, nz = self.resolution

        # Druck/Dichte initial
        p = np.ones_like(xg)[None, ...]
        u = np.zeros((3, nx, ny, nz), dtype=np.float64)

        # -----------------------------
        # 1) 1/7 power law Profil
        # -----------------------------
        # Halbhöhe H in "Index-Länge" (Zellenabstand = 1)
        # ny = 2H -> H ~ (ny-1)/2 (weil y=0..ny-1)
        H = 0.5 * (ny - 1)

        # Abstand zur nächsten Wand (symmetrisch)
        y = yg  # 0..ny-1
        dist_to_wall = np.minimum(y, (ny - 1) - y)  # 0..H

        # normierter Abstand z/H in [0,1]
        z_over_H = np.clip(dist_to_wall / H, 0.0, 1.0)

        # u_char in PU
        u_char = 1.0  # <- falls du willst: u_char = float(self.units.characteristic_velocity_pu)

        # Power-law, exponent 1/7
        u_base = u_char * (z_over_H ** (1.0 / 7.0))

        # Setze Basisprofil in x-Richtung
        u[0, :, :, :] = u_base

        # -----------------------------
        # 2) Zufallsstörungen (Normalverteilung)
        # -----------------------------
        sigma = 0.05  # 5%
        # Gaussian noise
        noise = rng.normal(loc=0.0, scale=sigma, size=(3, nx, ny, nz))

        # Störungen relativ zu u_char
        u += u_char * noise

        # -----------------------------
        # 3) No-Slip an Wänden erzwingen
        # -----------------------------
        u[:, :, 0, :] = 0.0
        u[:, :, -1, :] = 0.0

        # (falls du irgendwo eine Maske für feste Zellen nutzt)
        if self._mask is not None:
            u *= (1.0 - self.mask.astype(float))[None, ...]

        # Tensoren zurückgeben
        p_tensor = torch.tensor(p, dtype=self.context.dtype)
        u_tensor = torch.tensor(u, dtype=self.context.dtype)

        return p_tensor, u_tensor

    def initial_pu(self):
        rng = np.random.default_rng(self.random_seed)

        xg, yg, zg = self.grid
        nx, ny, nz = self.resolution

        # --- 1) Basisprofil & Dichte ---
        u_char = 1.0
        p = np.ones_like(xg)[None, ...]
        u = np.zeros((3, nx, ny, nz), dtype=np.float64)

        # KANAL-KORREKT: Abstand zur nächsten Wand
        H = 0.5 * (ny)
        dist_to_wall = np.minimum(yg, (ny) - yg)  # 0 an beiden Wänden, H in der Mitte
        z_over_H = np.clip(dist_to_wall / H, 0.0, 1.0)  # 0..1

        # 1/7 power law
        u_base = u_char * (z_over_H ** (1.0 / 7.0))
        u[0] = u_base * (1 - self.mask.astype(float))

        # --- 2) Gaussian Noise Trigger (Nathen et al. Stil) ---
        sigma = 0.10 * u_char  # 5% von u_char
        noise = rng.normal(loc=0.0, scale=sigma, size=(3, nx, ny, nz))

        # Envelope: 0 an Wänden, 1 in der Mitte (weicher als z_over_H)
        envelope = z_over_H * (1.0 - z_over_H) * 2
        envelope /= envelope.max() + 1e-30
        envelope = 1
        # Auf alle Komponenten anwenden
        u += noise * envelope[None, :, :, :]

        # Optional: zusätzlich u' etwas kleiner machen, falls u'u' zu hoch bleibt:
        # u[0] += 0.8 * noise[0] * envelope
        # u[1] += 1.0 * noise[1] * envelope
        # u[2] += 1.0 * noise[2] * envelope

        u[:, :, 0, :] = 0.0
        u[:, :, -1, :] = 0.0

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
            boundary = [DummyBoundary()]
        return boundary

