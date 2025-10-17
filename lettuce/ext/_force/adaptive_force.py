import torch
from lettuce import Force
from lettuce.ext import Guo
from lettuce.ext._boundary.wallfunction import compute_wall_quantities
from ...cuda_native.ext._force._force import NativeGuoForce

__all__ = ["AdaptiveForce"]


class AdaptiveForce(Force):
    """
    Korrigierte dual-mode adaptive Kraft.
    Funktioniert sowohl im Python-Modus als auch im nativen CUDA-Modus.
    """

    def __init__(self, flow, context, target_u_m_lu, base_lbm_tau_lu, mask=None):
        self.flow = flow
        self.context = context
        self.u_m = float(target_u_m_lu)
        self.base_lbm_tau = float(base_lbm_tau_lu)
        self.mask = mask

        # Simulation-Referenz für native Pfad
        self.simulation = None

        d = flow.stencil.d
        self.last_force_lu = context.convert_to_tensor([0.0] * d)

        # Für den Python-Pfad: Guo-Instanz
        self.python_guo = Guo(self.flow, self.base_lbm_tau, self.last_force_lu)

        # Für den Native-Pfad: Lazy erstellt
        self.native = None

    def update_native_force_on_simulation(self, i, t, f):
        """Callback-Funktion: Berechnet Kraft neu und aktualisiert Simulation."""
        if self.simulation is None:
            return

        # Kraft berechnen
        current_force = self.compute_force()

        # Simulation aktualisieren
        self.simulation.force_ax = float(current_force[0].item())
        if self.flow.stencil.d > 1:
            self.simulation.force_ay = float(current_force[1].item()) if len(current_force) > 1 else 0.0
        if self.flow.stencil.d > 2:
            self.simulation.force_az = float(current_force[2].item()) if len(current_force) > 2 else 0.0

    def set_on_simulation(self, simulation):
        """Wird vom Simulator aufgerufen: Initialisiert native Kraft."""
        self.simulation = simulation

        # Initiale Kraft berechnen
        initial_force = self.compute_force()
        simulation.force_ax = float(initial_force[0].item())
        simulation.force_ay = float(initial_force[1].item()) if len(initial_force) > 1 else 0.0
        simulation.force_az = float(initial_force[2].item()) if len(initial_force) > 2 else 0.0

        # Mask setzen (falls vorhanden)
        if self.mask is not None:
            simulation.force_mask = self.mask.to(torch.uint8)

        # WICHTIG: Callback registrieren
        if hasattr(simulation, 'callbacks'):
            simulation.callbacks.append(self.update_native_force_on_simulation)

    def compute_force(self):
        """Berechnet adaptive Kraft basierend auf Wandschubspannung."""
        device = self.flow.f.device
        dtype = self.flow.f.dtype
        dy = torch.as_tensor(0.5, device=device, dtype=dtype)

        # Wandschubspannung berechnen
        utau_top, _, _ = compute_wall_quantities(flow=self.flow, dy=dy, is_top=True)
        utau_bot, _, _ = compute_wall_quantities(flow=self.flow, dy=dy, is_top=False)
        utau_mean = 0.5 * (utau_top.mean() + utau_bot.mean())

        # Mittlere Geschwindigkeit
        ux_mean = self.flow.u()[0].mean()

        # Adaptive Kraft berechnen
        Fx = (utau_mean ** 2) / self.flow.h + (self.u_m - ux_mean) * (self.u_m / self.flow.h)

        # Kraft-Vektor erstellen
        vec = [0.0] * self.flow.stencil.d
        vec[0] = float(Fx)
        self.last_force_lu = self.context.convert_to_tensor(vec)

        return self.last_force_lu

    # --- Python-Pfad Methoden ---
    def source_term(self, u):
        """Source-Term für Python-Simulation."""
        current_force = self.compute_force()
        self.python_guo.acceleration = current_force
        source = self.python_guo.source_term(u)

        # Maske anwenden falls vorhanden
        if self.mask is not None:
            source *= self.mask.to(source.dtype)[None, ...]

        return source

    def u_eq(self, flow):
        """Gleichgewichts-Velocity-Shift für Python-Pfad."""
        current_force = self.compute_force()  # Kraft aktualisieren
        rho = flow.rho()
        # Division durch Null vermeiden
        denom = torch.where(rho < 1e-10, torch.tensor(1e-10, device=rho.device, dtype=rho.dtype), rho)
        index = [Ellipsis] + [None] * flow.stencil.d
        return self.ueq_scaling_factor * current_force[index] / denom

    @property
    def ueq_scaling_factor(self):
        return 0.5

    # --- Native-Pfad Methoden ---
    def native_available(self) -> bool:
        return True

    def native_generator(self):
        """Erstellt nativen Generator für CUDA-Pfad."""
        if self.native is None:
            self.native = NativeGuoForce.create(
                stencil=self.flow.stencil,
                tau=self.base_lbm_tau,
                use_mask=(self.mask is not None)
            )
        return self.native