__all__ = ['NativeExactDifferenceForce']
from lettuce.cuda_native import DefaultCodeGeneration, Parameter
from lettuce.cuda_native import DefaultCodeGeneration, Parameter

class NativeExactDifferenceForce:
    def __init__(self, python_force_instance):
        self.python_force = python_force_instance

    def generate(self, reg: 'DefaultCodeGeneration'):
        accel_name = f"acceleration_{hex(id(self))[2:]}"

        # einmalige "hook"-Sets (da reg.cuda_hooked/... nicht existieren)
        if not hasattr(reg, "_cuda_hooked_names"):
            reg._cuda_hooked_names = set()
        if not hasattr(reg, "_kernel_hooked_names"):
            reg._kernel_hooked_names = set()

        def cuda_hook_once(symbol_name: str, expr: str, param: Parameter):
            if symbol_name in reg._cuda_hooked_names:
                return
            reg.cuda_hook(expr, param)
            reg._cuda_hooked_names.add(symbol_name)

        def kernel_hook_once(symbol_name: str, expr: str, param: Parameter):
            if symbol_name in reg._kernel_hooked_names:
                return
            reg.kernel_hook(expr, param)
            reg._kernel_hooked_names.add(symbol_name)

        # 1) Acceleration aus Python verfügbar machen
        cuda_hook_once(accel_name,
                       'simulation.collision.force.acceleration',
                       Parameter('torch::Tensor', accel_name))
        kernel_hook_once(accel_name,
                         f'{accel_name}.accessor<scalar_t, 1>()',
                         Parameter('const auto', accel_name))

        buf = reg.pipe

        # 2) Hilfsfelder
        buf.append('scalar_t u_plus[d];')
        buf.append('scalar_t u_save[d];')
        buf.append('scalar_t source_term[q];')

        # u_plus = u + 0.5 * a
        buf.append('#pragma unroll')
        buf.append('for (index_t i = 0; i < d; ++i) {')
        buf.append(f'  u_plus[i] = u[i] + static_cast<scalar_t>(0.5) * {accel_name}[i];')
        buf.append('}')

        # f_eq(u) pro Richtung (als Skalar) sichern
        for q in range(reg.stencil.q):
            f_eq_base_q = reg.equilibrium.f_eq(reg, q)   # Ausdruck für f_eq(u)
            buf.append(f'scalar_t f_eq_base_{q} = {f_eq_base_q};')

        # u sichern und temporär u = u_plus setzen
        buf.append('#pragma unroll')
        buf.append('for (index_t i = 0; i < d; ++i) { u_save[i] = u[i]; u[i] = u_plus[i]; }')

        # f_eq(u_plus) pro Richtung berechnen
        for q in range(reg.stencil.q):
            f_eq_plus_q = reg.equilibrium.f_eq(reg, q)  # Ausdruck für f_eq(u_plus), weil u==u_plus
            buf.append(f'scalar_t f_eq_plus_{q} = {f_eq_plus_q};')

        # u zurücksetzen
        buf.append('#pragma unroll')
        buf.append('for (index_t i = 0; i < d; ++i) { u[i] = u_save[i]; }')

        # Source-Term: S_i = f_eq(u_plus) - f_eq(u)
        for q in range(reg.stencil.q):
            buf.append(f'source_term[{q}] = f_eq_plus_{q} - f_eq_base_{q};')

        return "source_term"
