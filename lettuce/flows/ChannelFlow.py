import numpy as np
from lettuce.unit import UnitConversion
from lettuce.boundary import EquilibriumBoundaryPU, BounceBackBoundary, AntiBounceBackOutlet,\
    HalfwayBounceBackBoundary, WallFunctionBoundary, FreeSlipBoundary


class ChannelFlow2D(object):
    """
    Flow class to simulate the flow around an object (mask) in 2D.
    It consists off one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive x direction.

    Parameters
    ----------
    resolution_x : int
        Grid resolution in streamwise direction.
    resolution_y : int
        Grid resolution in spanwise direction.
    char_length_lu : float
        The characteristic length in lattice units; usually the number of grid points for the obstacle in flow direction

    Attributes
    ----------
    mask : np.array with dtype = bool
        Boolean mask to define the obstacle. The shape of this object is the shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------
    Initialization of flow around a cylinder:

    >>> from lettuce import Lattice, D2Q9
    >>> flow = Obstacle2D(
    >>>     resolution_x=101,
    >>>     resolution_y=51,
    >>>     reynolds_number=100,
    >>>     mach_number=0.1,
    >>>     lattice=lattice,
    >>>     char_length_lu=10
    >>> )
    >>> x, y = flow.grid
    >>> x = flow.units.convert_length_to_lu(x)
    >>> y = flow.units.convert_length_to_lu(y)
    >>> condition = np.sqrt((x-25)**2+(y-25)**2) < 5.0001
    >>> flow.mask[np.where(condition)] = 1
   """

    def __init__(self, resolution_x, resolution_y, reynolds_number, mach_number, lattice, char_length_lu, boundary):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=char_length_lu, characteristic_length_pu=1,
            characteristic_velocity_pu=1
        )
        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y), dtype=bool)
        self._boundary = boundary

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y)
        self._mask = m.astype(bool)

    def initial_solution(self, grid):
        xg, yg = grid
        p = np.ones_like(xg)[None, ...]
        nx, ny = self.resolution_x, self.resolution_y

        # --- 📐 Basisströmung: Poiseuille-Profil ---
        y_normalized = yg / yg.max()
        u_base = y_normalized * (1 - y_normalized) * 4

        u = np.zeros((2, nx, ny))
        u[0] = u_base * (1 - self.mask.astype(float))

        # --- 🎛️ Sinus-Moden-Störung ---
        A_sin = 1  # 5 % Störung
        Lx = xg.max()
        Ly = yg.max()
        sinus_modes = [(1, 1), (2, 3), (3, 2)]

        for kx, ky in sinus_modes:
            phase = 2 * np.pi * np.random.rand()
            mode = np.sin(2 * np.pi * (kx * xg / Lx + ky * yg / Ly) + phase)
            envelope = y_normalized * (1 - y_normalized)  # nur in der Mitte stark
            u[0] += A_sin * mode * envelope

        # --- 🌪️ Vektorpotential-Störung ---
        A_psi = 1  # separat skalierbar
        random_psi = ((np.random.rand(2, nx, ny) - 0.5) * 2)

        # Wandgewichtung (nur vertikal)
        weight_y = np.exp(-((y_normalized - 0.0) / 0.2) ** 2) + np.exp(-((y_normalized - 1.0) / 0.2) ** 2)
        weight_y /= weight_y.max()
        random_psi *= weight_y[None, :, :]

        # Weichfilterung im Spektralraum
        k0 = np.sqrt(nx ** 2 + ny ** 2)
        psi_filtered = np.empty_like(random_psi)
        for d in range(2):
            psi_hat = np.fft.fft2(random_psi[d])
            kx = np.fft.fftfreq(nx).reshape(-1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1)
            kabs = np.sqrt((kx * nx) ** 2 + (ky * ny) ** 2)
            filter_mask = np.exp(-kabs / (0.3 * k0))  # sanfter Filter
            psi_hat *= filter_mask
            psi_hat[0, 0] = 0
            psi_filtered[d] = np.real(np.fft.ifft2(psi_hat))

        # Geschwindigkeit via Curl(ψ)
        u_psi = np.zeros_like(psi_filtered)
        u_psi[0] = np.gradient(psi_filtered[1], axis=0) - np.gradient(psi_filtered[0], axis=1)
        u_psi[1] = 0.0  # kann auch u_y mit aufnehmen

        # Normierung und Überlagerung
        umax_psi = np.max(np.sqrt(np.sum(u_psi ** 2, axis=0)))
        if umax_psi > 0:
            u_psi *= A_psi / umax_psi

        u += u_psi  # überlagern

        u[:, :, 0] = 0.0  # untere Wand y=0
        u[:, :, -1] = 0.0  # obere Wand y=Ny-1
        return p, u

    @property
    def grid(self):
        stop_x = self.resolution_x / self.units.characteristic_length_lu
        stop_y = self.resolution_y / self.units.characteristic_length_lu

        x = np.linspace(0, stop_x, num=self.resolution_x, endpoint=True)
        y = np.linspace(0, stop_y, num=self.resolution_y, endpoint=True)

        return np.meshgrid(x, y, indexing='ij')

    @property
    def boundaries(self):
        x, y = self.grid
        Ny = y.shape[1]

        # Bounce-Back-Maske (Wände bei y=0 und y=Ny-1)
        mask_bb = np.zeros_like(x, dtype=bool)
        mask_bb[:, 0] = True  # untere Wand
        mask_bb[:, Ny - 1] = True  # obere Wand

        # Wall-Function-Masken (erste Fluidzellen direkt an der Wand)
        mask_bottom = np.zeros_like(x, dtype=bool)
        mask_bottom[:, 1] = True

        mask_top = np.zeros_like(x, dtype=bool)
        mask_top[:, Ny - 2] = True

        if self._boundary == "halfway":
            bb = HalfwayBounceBackBoundary(mask_bb, self.units.lattice)
        else:
            bb = BounceBackBoundary(mask_bb, self.units.lattice)

        return [
            bb,
            WallFunctionBoundary(mask_bottom, self.units.lattice, self.units.viscosity_lu, wall='bottom'),
            WallFunctionBoundary(mask_top, self.units.lattice, self.units.viscosity_lu, wall='top')
        ]


class ChannelFlow3D(object):
    """Flow class to simulate the flow around an object (mask) in 3D.
    See documentation for :class:`~Obstacle2D` for details.
    """

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu, boundary):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=1,
            characteristic_velocity_pu=1)

        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=bool)
        self._boundary = boundary
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(bool)

    def initial_solution(self, grid):
        xg, yg, zg = grid
        p = np.ones_like(xg)[None, ...]
        nx, ny, nz = self.resolution_x, self.resolution_y, self.resolution_z

        u = np.zeros((3, nx, ny, nz))

        # --- 📐 Poiseuille-Profil (in x-Richtung) ---
        y_normalized = yg / yg.max()
        u_base = y_normalized * (1 - y_normalized)
        u[0] = u_base * (1 - self.mask.astype(float))  # u_x = Basisströmung

        # --- 🎛️ Sinusmoden-Störung (3D) ---
        A_sin = 0.5  # 5% Amplitude
        Lx, Ly, Lz = xg.max(), yg.max(), zg.max()
        sinus_modes = [(1, 1, 1), (2, 2, 3), (3, 2, 1)]

        for kx, ky, kz in sinus_modes:
            phase = 2 * np.pi * np.random.rand()
            mode = np.sin(2 * np.pi * (kx * xg / Lx + ky * yg / Ly + kz * zg / Lz) + phase)
            envelope = y_normalized * (1 - y_normalized)
            u[0] += A_sin * mode * envelope  # nur u_x gestört, kannst du erweitern

        # --- 🌪️ Vektorpotential ψ (3 Komponenten für Curl in 3D) ---
        A_psi = 1
        random_psi = ((np.random.rand(3, nx, ny, nz) - 0.5) * 2)

        # Wandgewichtung in y und z
        y_weight = np.exp(-((y_normalized - 0.0) / 0.2) ** 2) + np.exp(-((y_normalized - 1.0) / 0.2) ** 2)
        y_weight /= y_weight.max()

        z_normalized = zg / zg.max()
        z_weight = np.exp(-((z_normalized - 0.5) / 0.3) ** 2)
        z_weight /= z_weight.max()

        weight = y_weight * z_weight
        random_psi *= weight[None, :, :, :]

        # FFT-Filterung (3D)
        k0 = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        psi_filtered = np.empty_like(random_psi)
        for d in range(3):
            psi_hat = np.fft.fftn(random_psi[d])
            kx = np.fft.fftfreq(nx).reshape(-1, 1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1, 1)
            kz = np.fft.fftfreq(nz).reshape(1, 1, -1)
            kabs = np.sqrt((kx * nx) ** 2 + (ky * ny) ** 2 + (kz * nz) ** 2)
            filter_mask = np.exp(-kabs / (0.3 * k0))
            psi_hat *= filter_mask
            psi_hat[0, 0, 0] = 0
            psi_filtered[d] = np.real(np.fft.ifftn(psi_hat))

        # --- 🌀 Curl(ψ): u = ∇ × ψ ---
        u_psi = np.zeros_like(u)
        u_psi[0] = np.gradient(psi_filtered[2], axis=1) - np.gradient(psi_filtered[1], axis=2)  # u_x
        u_psi[1] = np.gradient(psi_filtered[0], axis=2) - np.gradient(psi_filtered[2], axis=0)  # u_y
        u_psi[2] = np.gradient(psi_filtered[1], axis=0) - np.gradient(psi_filtered[0], axis=1)  # u_z

        # Normierung
        umax_psi = np.max(np.sqrt(np.sum(u_psi ** 2, axis=0)))
        if umax_psi > 0:
            u_psi *= A_psi / umax_psi

        # --- Überlagerung: Basis + Sine + Curl ---
        u += u_psi
        # 2D: Nullsetzen der Wandgeschwindigkeit

        u[:, :, 0, :] = 0.0  # untere Wand (y=0)
        u[:, :, -1, :] = 0.0  # obere Wand (y=Ny-1)

        return p, u

    @property
    def grid(self):
        stop_x = self.resolution_x / self.units.characteristic_length_lu
        stop_y = self.resolution_y / self.units.characteristic_length_lu
        stop_z = self.resolution_z / self.units.characteristic_length_lu

        x = np.linspace(0, stop_x, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, stop_y, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, stop_z, num=self.resolution_z, endpoint=False)

        return np.meshgrid(x, y, z, indexing='ij')

    @property
    def boundaries(self):
        x, y, z = self.grid  # Jetzt auch z
        Ny = y.shape[1]  # Höhe des Kanals in y-Richtung

        # Bounce-Back-Maske (Wände bei y=0 und y=Ny-1)
        # In 3D müssen wir die Maske über alle x- und z-Koordinaten ausdehnen.
        mask_bb = np.zeros_like(x, dtype=bool)
        mask_bb[:, 0, :] = True  # untere Wand (y=0)
        mask_bb[:, Ny - 1, :] = True  # obere Wand (y=Ny-1)

        # Wall-Function-Masken (erste Fluidzellen direkt an der Wand)
        # Auch hier über alle x- und z-Koordinaten ausdehnen.
        mask_bottom = np.zeros_like(x, dtype=bool)
        mask_bottom[:, 0, :] = True  # Erste Fluidzelle über der unteren Wand (y=1)

        mask_top = np.zeros_like(x, dtype=bool)
        mask_top[:, Ny - 1, :] = True  # Erste Fluidzelle unter der oberen Wand (y=Ny-2)

        # Das Boundary-Objekt für Bounce-Back
        if self._boundary == "halfway":
            bb = [HalfwayBounceBackBoundary(mask_bb, self.units.lattice)]
        elif self._boundary == "fullway":
            bb = [BounceBackBoundary(mask_bb, self.units.lattice)]
        elif self._boundary == "freeslip":
            bb = [FreeSlipBoundary(mask_bb, self.units.lattice)]
        elif self._boundary == "wallfunction":
            bb = [

            WallFunctionBoundary(mask_bottom, self.units.lattice, self.units.viscosity_lu, wall='bottom'),
            WallFunctionBoundary(mask_top, self.units.lattice, self.units.viscosity_lu, wall='top')]
        # Rückgabe der Liste von Boundary-Objekten
        # Wichtig: Die WallFunctionBoundary-Objekte müssen später (nach der Simulation-Initialisierung)
        # mit dem 'collision_model' aktualisiert werden, wie wir es besprochen haben.
        return bb


class ChannelFlow3DV2(object):
    # ... (Ihre __init__, mask Property und grid Property bleiben unverändert) ...

    def __init__(self, resolution_x, resolution_y, resolution_z, reynolds_number, mach_number, lattice, char_length_lu,
                 boundary):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

        self.units = UnitConversion(
            lattice,
            reynolds_number=reynolds_number,
            mach_number=mach_number,
            characteristic_length_lu=char_length_lu,
            characteristic_length_pu=1,
            characteristic_velocity_pu=1)

        self._mask = np.zeros(shape=(self.resolution_x, self.resolution_y, self.resolution_z), dtype=bool)
        self._boundary = boundary

        # KORREKTUR HIER: self.ndim definieren
        # Die Dimension (2 oder 3) ist im Lattice-Objekt als Attribut D verfügbar.
        self.ndim = lattice.D

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert isinstance(m, np.ndarray) and m.shape == (self.resolution_x, self.resolution_y, self.resolution_z)
        self._mask = m.astype(bool)

    def initial_solution(self, grid):
        xg, yg, zg = grid
        p = np.ones_like(xg)[None, ...]
        nx, ny, nz = self.resolution_x, self.resolution_y, self.resolution_z

        u = np.zeros((3, nx, ny, nz))

        # --- 📐 Power-law Profil (in x-Richtung) für turbulente Kanalströmung ---
        # Umrechnung von y in y+
        # y_coords (physical units) von yg (shape: Nx, Ny, Nz) -> y_pu (Ny,)
        # Wir brauchen nur die y-Achse als 1D-Array, also z.B. yg[0, :, 0]
        # self.ndim ist jetzt im __init__ definiert.
        y_coords_pu = yg[0, :, 0] if self.ndim == 3 else yg[0, :]

        # Holen der molekularen Viskosität in PU
        nu_pu = self.units.viscosity_pu

        # Ziel-Reibgeschwindigkeit aus dem Paper (m/s)
        u_tau_target_pu = 0.0509

        # Umrechnung der y-Koordinaten in y+
        y_plus_coords = (y_coords_pu * u_tau_target_pu) / nu_pu

        # Implementierung des Power-law Profils
        u_x_profile_pu = np.zeros_like(y_plus_coords)

        # Parameter für Power-law (Werner and Wengle)
        y_c_plus = 11.81
        m_exponent = 1 / 7
        C_m_constant = 8.3

        # Wenden Sie das Power-law Profil auf y_plus_coords an
        for i, yp_val in enumerate(y_plus_coords):
            if yp_val < y_c_plus:
                u_x_profile_pu[i] = yp_val
            else:
                u_x_profile_pu[i] = C_m_constant * (yp_val ** m_exponent)

                # Skalieren zu tatsächlicher Geschwindigkeit in LU
        u_base_pu = u_x_profile_pu * u_tau_target_pu
        u_base_lu = self.units.convert_velocity_to_lu(u_base_pu)

        # Setzen der Basis-Geschwindigkeit für u_x-Komponente
        # u_base_lu hat Form (Ny,). Wir müssen es auf (Nx, Ny, Nz) erweitern.
        if self.ndim == 3:
            # u[0] hat (Ny,) -> np.newaxis macht es (1,Ny) -> np.newaxis macht es (1,Ny,1)
            # Dann tile auf (Nx, 1, Nz)
            u[0] = u_base_lu[np.newaxis, :, np.newaxis] * np.ones((nx, ny, nz), dtype=np.float64)
            # Alternativ: Broadcast mit unsqueeze in torch, dann zu numpy
            # u_base_lu_3d = u_base_lu.reshape(1, ny, 1).repeat(nx, 1, nz)
            # u[0] = u_base_lu_3d.numpy() * (1 - self.mask.astype(float))
            # ODER den broadcast so nutzen:
            # u[0] = (u_base_lu[np.newaxis, :, np.newaxis] * (1 - self.mask.astype(float))) # Original
            # Der Fehler war hier, dass u[0] schon (nx,ny,nz) ist.
            # Richtig so: u[0][:,:,:] = u_base_lu[np.newaxis, :, np.newaxis] * (1 - self.mask.astype(float))
            u[0][:, :, :] = u_base_lu[np.newaxis, :, np.newaxis]  # broadcast to (nx, ny, nz)
            u[0] *= (1 - self.mask.astype(float))  # Apply mask
        elif self.ndim == 2:
            u[0][:, :] = u_base_lu[np.newaxis, :]  # broadcast to (nx, ny)
            u[0] *= (1 - self.mask.astype(float))  # Apply mask

        # --- 🎛️ Sinusmoden-Störung (3D, 15% Zufallsfluktuationen) ---
        rand_noise_amplitude = 0.15 * np.max(u_base_lu)
        random_perturbation = (np.random.rand(3, nx, ny, nz) - 0.5) * 2 * rand_noise_amplitude

        # Wandgewichtung anwenden (wie in Ihrem Code)
        y_normalized = yg / yg.max()
        z_normalized = zg / zg.max()
        y_weight = np.exp(-((y_normalized - 0.0) / 0.2) ** 2) + np.exp(-((y_normalized - 1.0) / 0.2) ** 2)
        y_weight /= np.maximum(y_weight.max(), 1e-10)
        z_weight = np.exp(-((z_normalized - 0.5) / 0.3) ** 2)
        z_weight /= np.maximum(z_weight.max(), 1e-10)
        weight = y_weight * z_weight
        random_perturbation *= weight[None, :, :, :]

        # FFT-Filterung (wie Ihr Code es tut)
        k0 = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
        psi_filtered = np.empty_like(random_perturbation)
        for d in range(3):
            psi_hat = np.fft.fftn(random_perturbation[d])
            kx = np.fft.fftfreq(nx).reshape(-1, 1, 1)
            ky = np.fft.fftfreq(ny).reshape(1, -1, 1)
            kz = np.fft.fftfreq(nz).reshape(1, 1, -1)
            kabs = np.sqrt((kx * nx) ** 2 + (ky * ny) ** 2 + (kz * nz) ** 2)
            filter_mask = np.exp(-kabs / (0.3 * k0))
            psi_hat *= filter_mask
            psi_hat[0, 0, 0] = 0
            psi_filtered[d] = np.real(np.fft.ifftn(psi_hat))

        u_psi_turbulent_like = np.zeros_like(u)
        u_psi_turbulent_like[0] = np.gradient(psi_filtered[2], axis=1) - np.gradient(psi_filtered[1], axis=2)
        u_psi_turbulent_like[1] = np.gradient(psi_filtered[0], axis=2) - np.gradient(psi_filtered[2], axis=0)
        u_psi_turbulent_like[2] = np.gradient(psi_filtered[1], axis=0) - np.gradient(psi_filtered[0], axis=1)

        # Normierung der Störung
        umax_psi = np.max(np.sqrt(np.sum(u_psi_turbulent_like ** 2, axis=0)))
        if umax_psi > 1e-10:
            u_psi_turbulent_like *= (rand_noise_amplitude / umax_psi)

        u += u_psi_turbulent_like

        # --- Nullsetzen der Wandgeschwindigkeiten (No-Slip für die Initialisierung) ---
        u[:, :, 0, :] = 0.0  # untere Wand (y=0)
        u[:, :, -1, :] = 0.0  # obere Wand (y=Ny-1)

        return p, u

    @property
    def grid(self):
        # ... (bleibt unverändert) ...
        stop_x = self.resolution_x / self.units.characteristic_length_lu
        stop_y = self.resolution_y / self.units.characteristic_length_lu
        stop_z = self.resolution_z / self.units.characteristic_length_lu

        x = np.linspace(0, stop_x, num=self.resolution_x, endpoint=False)
        y = np.linspace(0, stop_y, num=self.resolution_y, endpoint=False)
        z = np.linspace(0, stop_z, num=self.resolution_z, endpoint=False)

        return np.meshgrid(x, y, z, indexing='ij')
    @property
    def boundaries(self):
        # ... (bleibt unverändert) ...
        x, y, z = self.grid
        Ny = y.shape[1]

        mask_bottom_wall = np.zeros_like(x, dtype=bool)
        mask_bottom_wall[:, 0, :] = True

        mask_top_wall = np.zeros_like(x, dtype=bool)
        mask_top_wall[:, Ny - 1, :] = True

        smagorinsky_constant = 0.17
        delta_x = 1.0

        wffsb_bottom = WallFunctionBoundary(  # Class name as you use it
            mask=mask_bottom_wall,
            lattice=self.units.lattice,
            viscosity=self.units.viscosity_lu,
            y_lattice=1.0,  # Fullway BB base
            wall='bottom',
            smagorinsky_constant=smagorinsky_constant,
            delta_x=delta_x,
            apply_wfb_correction=True
        )

        wffsb_top = WallFunctionBoundary(  # Class name as you use it
            mask=mask_top_wall,
            lattice=self.units.lattice,
            viscosity=self.units.viscosity_lu,
            y_lattice=1.0,  # Fullway BB base
            wall='top',
            smagorinsky_constant=smagorinsky_constant,
            delta_x=delta_x,
            apply_wfb_correction=True
        )

        return [wffsb_bottom, wffsb_top]