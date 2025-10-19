__all__ = ['NativeExactDifferenceForce']

class NativeExactDifferenceForce:
    """
    Generiert den nativen C++/CUDA Code für die Exact Difference Method.
    Diese Klasse braucht keine Basisklasse, nur eine __init__- und eine generate-Methode.
    """

    def __init__(self, python_force_instance):
        """
        Speichert eine Referenz auf die Python-Instanz, um an die Daten (Beschleunigung) zu kommen.
        """
        self.python_force = python_force_instance

    def generate(self, generator: 'Generator'):
        """
        Diese Methode wird von NativeBGKCollision aufgerufen.
        Sie fügt den C++ Code für den Source Term in den Kernel ein.
        """
        # 1. Beschleunigungs-Tensor für den C++ Kernel verfügbar machen
        accel_name = f"acceleration_{hex(id(self))[2:]}"
        if not generator.launcher_hooked(accel_name):
            # Übergibt den Python-Tensor an den C++ Launcher
            generator.launcher_hook(
                accel_name,
                f"torch::Tensor {accel_name}",
                accel_name,
                # Pfad zum Tensor im Python-Simulations-Objekt
                'simulation.collision.force.acceleration'
            )
            # Macht den Tensor im Kernel als "Accessor" verfügbar
            generator.kernel_hook(
                accel_name,
                f"const auto {accel_name}",
                f"{accel_name}.accessor<scalar_t, 1>()"
            )

        buffer = generator.append_pipeline_buffer

        # 2. C++ Code zur Berechnung des Source Terms generieren

        # Lokale C++-Variablen im Kernel deklarieren
        # Lokale C++-Variablen deklarieren (sichtbar im Kernel)
        # Lokale C++-Variablen im Kernel deklarieren
        buffer('scalar_t u_plus[d];')
        buffer('scalar_t f_eq_plus[q];')
        buffer('scalar_t source_term[q];')

        # u_plus = u + 0.5 * a
        buffer('// Berechne u_plus = u + 0.5 * a')
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < d; ++i) {')
        buffer(f'    u_plus[i] = u[i] + static_cast<scalar_t>(0.5) * {accel_name}[i];')
        buffer('}')

        # --- NEU: f_eq(u) sichern, u -> u_plus setzen, f_eq(u_plus) generieren, alles zurücksetzen ---

        # 1) f_eq(u) sichern
        buffer('scalar_t f_eq_base[q];')
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < q; ++i) {')
        buffer('    f_eq_base[i] = f_eq[i];')
        buffer('}')

        # 2) u sichern und temporär mit u_plus überschreiben
        buffer('scalar_t u_save[d];')
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < d; ++i) {')
        buffer('    u_save[i] = u[i];')
        buffer('    u[i] = u_plus[i];')
        buffer('}')

        # 3) f_eq(u_plus) in den Standard-Namen f_eq schreiben lassen
        buffer('// f_eq für (rho, u_plus) generieren (schreibt in f_eq)')
        generator.equilibrium.generate_f_eq(generator)  # keine eigenen Namen -> nutzt u & f_eq

        # 4) f_eq(u_plus) nach f_eq_plus kopieren
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < q; ++i) {')
        buffer('    f_eq_plus[i] = f_eq[i];')
        buffer('}')

        # 5) u und f_eq(u) wiederherstellen
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < d; ++i) {')
        buffer('    u[i] = u_save[i];')
        buffer('}')
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < q; ++i) {')
        buffer('    f_eq[i] = f_eq_base[i];')
        buffer('}')

        # 6) Source-Term S_i = f_eq(u_plus) - f_eq(u)
        buffer('// Source-Term S_i = f_eq_plus[i] - f_eq[i]')
        buffer('#pragma unroll')
        buffer('for (index_t i = 0; i < q; ++i) {')
        buffer('    source_term[i] = f_eq_plus[i] - f_eq[i];')
        buffer('}')

        # Ergebnis-Arrayname zurückgeben
        return "source_term"
