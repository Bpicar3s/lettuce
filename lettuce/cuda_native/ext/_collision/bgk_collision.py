from typing import Optional

from lettuce.cuda_native import DefaultCodeGeneration, Parameter
from ... import NativeCollision
# Importiere NativeForce, um den Typ-Hint zu korrigieren


__all__ = ['NativeBGKCollision']


class NativeBGKCollision(NativeCollision):

    def __init__(self, index: int, force: Optional['NativeForce'] = None):
        NativeCollision.__init__(self, index)
        self.force = force

    @staticmethod
    def create(index: int, force: Optional['NativeForce'] = None):
        """
        Korrigierte Factory-Methode.
        Gibt *immer* eine Instanz zurück, entweder mit oder ohne force.
        """
        return NativeBGKCollision(index, force)

    def cuda_tau_inv(self, reg: 'DefaultCodeGeneration'):
        variable = f"tau_inv{hex(id(self))[2:]}"
        # Greift auf simulation.collision.tau zu (was BGKCollision.tau ist)
        return reg.cuda_hook('1.0 / simulation.collision.tau', Parameter('double', variable))

    def kernel_tau_inv(self, reg: 'DefaultCodeGeneration'):
        variable = self.cuda_tau_inv(reg)
        return reg.kernel_hook(f"static_cast<scalar_t>({variable})", Parameter('scalar_t', variable))

    def generate(self, reg: 'DefaultCodeGeneration'):

        # ================================================================
        print("\n--- NATIVE KERNEL GENERATOR WIRD AUSGEFÜHRT ---")
        if self.force is None:
            print("Status: Compiling WITHOUT forcing. (self.force war None)")
        else:
            print("Status: Compiling WITH forcing. (self.force war VORHANDEN)")
        print("--------------------------------------------------\n")
        # ================================================================

        tau_inv = self.kernel_tau_inv(reg)

        # --- Fall 1: Standard BGK (kein Forcing) ---
        if self.force is None:
            for q in range(reg.stencil.q):
                f_q = reg.f_reg(q)
                f_eq_q = reg.equilibrium.f_eq(reg, q)
                reg.pipe.append(f"{f_q}={f_q}-({tau_inv}*({f_q}-{f_eq_q}));")
            return

        # --- Fall 2: BGK mit Python-identischer EDM-Logik ---

        force_id = hex(id(self.force))[2:]

        # 1. Hole makroskopische Größen (u_macro)
        rho = reg.kernel_rho()
        rho_inv = reg.kernel_rho_inv()
        u_macro = [reg.kernel_u(d) for d in range(reg.stencil.d)]

        # 2. Hooke UNMASKIERTE Beschleunigung 'a'
        accel_tensor_var = f"accel_tensor_{force_id}"
        kernel_accel_ptr = f"k_ptr_accel_{force_id}"
        cuda_accel = reg.cuda_hook(
            'simulation.collision.force.acceleration',
            Parameter('at::Tensor', accel_tensor_var)
        )
        reg.kernel_hook(
            f"{cuda_accel}.data_ptr<scalar_t>()",
            Parameter('scalar_t*', kernel_accel_ptr)
        )
        a = [f"{kernel_accel_ptr}[{d}]" for d in range(reg.stencil.d)]

        # 3. Berechne u = u_macro + 0.5*a/rho (UNMASKIERT)
        # (entspricht ExactDifferenceForce.u_eq)
        u_eq = [
            reg.pipes.variable('scalar_t', f'u_eq_{d}_{force_id}',
                               f"static_cast<scalar_t>(0.5) * {a[d]} * {rho_inv}")
            for d in range(reg.stencil.d)
        ]
        u = [
            reg.pipes.variable('scalar_t', f'u_shifted_{d}_{force_id}',
                               f"{u_macro[d]} + {u_eq[d]}")
            for d in range(reg.stencil.d)
        ]

        # 4. Berechne feq = feq(rho, u)
        # (entspricht BGKCollision.__call__)
        feq = [
            reg.equilibrium.f_eq(reg, q, rho=rho, u=u)
            for q in range(reg.stencil.q)
        ]

        # 5. Berechne u_plus = u + 0.5*a (UNMASKIERT)
        # (entspricht ExactDifferenceForce.source_term)
        du = [
            reg.pipes.variable('scalar_t', f'du_{d}_{force_id}',
                               f"static_cast<scalar_t>(0.5) * {a[d]}")
            for d in range(reg.stencil.d)
        ]
        u_plus = [
            reg.pipes.variable('scalar_t', f'u_plus_{d}_{force_id}',
                               f"{u[d]} + {du[d]}")  # Basiert auf u, nicht u_macro!
            for d in range(reg.stencil.d)
        ]

        # 6. Berechne feq_plus = feq(rho, u_plus)
        feq_plus = [
            reg.equilibrium.f_eq(reg, q, rho=rho, u=u_plus)
            for q in range(reg.stencil.q)
        ]

        # 7. Berechne UNMASKIERTEN si = feq_plus - feq
        si_unmasked = [
            reg.pipes.variable('scalar_t', f'si_unmasked_{q}_{force_id}',
                               f"{feq_plus[q]} - {feq[q]}")
            for q in range(reg.stencil.q)
        ]

        # 8. Hooke die Maske
        mask_check_var = f"has_mask_{force_id}"
        kernel_has_mask = f"k_{mask_check_var}"
        reg.python_pre.append(f"{mask_check_var} = simulation.collision.force.mask is not None")
        cuda_has_mask = reg.cuda_hook(mask_check_var, Parameter('bool', mask_check_var))
        reg.kernel_hook(cuda_has_mask, Parameter('bool', kernel_has_mask))

        mask_tensor_var = f"mask_tensor_{force_id}"
        kernel_mask_ptr = f"k_ptr_mask_{force_id}"
        cuda_mask = reg.cuda_hook(
            f'simulation.collision.force.mask if {mask_check_var} else simulation.no_collision_mask',
            Parameter('at::Tensor', mask_tensor_var)
        )
        reg.kernel_hook(f"{cuda_mask}.data_ptr<bool>()",  # KORREKTUR: bool
                        Parameter('bool*', kernel_mask_ptr))  # KORREKTUR: bool

        base_index = reg.kernel_base_index()
        # KORREKTUR: 'bool' und 'true'. Wenn keine Maske da ist, ist mask_value = true (1.0).
        mask_value = reg.pipes.variable('bool', f'mask_val_{force_id}',
                                        f"{kernel_has_mask} ? {kernel_mask_ptr}[{base_index}] : true")

        # 9. Wende Maske auf 'si' an (wie in Python)
        si_masked = [
            reg.pipes.variable('scalar_t', f'si_masked_{q}_{force_id}',
                               f"{si_unmasked[q]} * static_cast<scalar_t>({mask_value})")
            for q in range(reg.stencil.q)
        ]

        # 10. Führe den finalen Kollisionsschritt durch
        # f_new = f - tau_inv * (f - feq) + si_masked
        for q in range(reg.stencil.q):
            f_q = reg.f_reg(q)
            reg.pipe.append(f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {feq[q]})) + {si_masked[q]};")