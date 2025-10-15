# Du kannst diese Klasse in einer neuen Datei (z.B. native/force.py)
# oder neben der ExactDifferenceForce-Klasse platzieren.

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
        buffer('scalar_t u_plus[d];')
        buffer('scalar_t f_eq_plus[q];')
        buffer('scalar_t source_term[q];')

        # u_plus = u + 0.5 * a berechnen
        buffer('// Berechne u_plus = u + 0.5 * a')
        buffer('#pragma unroll')
        buffer('for(index_t i = 0; i < d; ++i) {')
        buffer(f'    u_plus[i] = u[i] + static_cast<scalar_t>(0.5) * {accel_name}[i];')
        buffer('}')

        # f_eq für u_plus berechnen (genial: wir nutzen den existierenden f_eq-Generator!)
        buffer('// Berechne f_eq(rho, u_plus)')
        generator.equilibrium.generate_f_eq(generator, u_var='u_plus', f_eq_var='f_eq_plus')

        # Source-Term S_i = f_eq(u+Δu) - f_eq(u) berechnen
        # (f_eq für das normale u wurde bereits vorher von NativeBGKCollision generiert)
        buffer('// Berechne den Source-Term S_i = f_eq_plus[i] - f_eq[i]')
        buffer('#pragma unroll')
        buffer('for(index_t i = 0; i < q; ++i) {')
        buffer('    source_term[i] = f_eq_plus[i] - f_eq[i];')
        buffer('}')

        # 3. Den Namen der C++ Variable zurückgeben, die das Ergebnis enthält
        return "source_term"