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
        print("\n--- NATIVE KERNEL GENERATOR (KORREKTE, VEREINFACHTE VERSION) ---")
        if self.force is None:
            print("Status: Compiling WITHOUT forcing.")
            reg.pipe.append("// Native BGK (no force)")
        else:
            print("Status: Compiling WITH forcing (Python-Algorithmus).")
            reg.pipe.append("// Native BGK-EDM (Python-Algorithmus)")
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

        # --- Fall 2: BGK mit Forcing (Python-identisch) ---
        # Dieser Code ist ein 1:1-Mapping von:
        # BGKCollision.__call__ + ExactDifferenceForce.source_term

        force_id = hex(id(self.force))[2:]

        # 1. Hole makroskopische Größen
        rho = reg.kernel_rho()
        rho_inv = reg.kernel_rho_inv()
        u_macro = [reg.kernel_u(d) for d in range(reg.stencil.d)]

        # 2. Hooke Beschleunigung 'a'
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

        # 3. Berechne u_shift = u_macro + 0.5*a/rho
        # (Entspricht: u_eq = force.u_eq(flow); u = flow.u() + u_eq)
        u_eq = [
            reg.pipes.variable('scalar_t', f'u_eq_{d}_{force_id}',
                               f"static_cast<scalar_t>(0.5) * {a[d]} * {rho_inv}")
            for d in range(reg.stencil.d)
        ]
        u_shift = [
            reg.pipes.variable('scalar_t', f'u_shifted_{d}_{force_id}',
                               f"{u_macro[d]} + {u_eq[d]}")
            for d in range(reg.stencil.d)
        ]

        # 4. Berechne feq = feq(u_shift)
        # (Entspricht: feq = flow.equilibrium(flow, u=u))
        feq_shift = [
            reg.equilibrium.f_eq(reg, q, rho=rho, u=u_shift)
            for q in range(reg.stencil.q)
        ]

        # 5. Berechne u_plus = u_shift + 0.5*a
        # (Entspricht: u_plus = u + du in source_term)
        du = [
            reg.pipes.variable('scalar_t', f'du_{d}_{force_id}',
                               f"static_cast<scalar_t>(0.5) * {a[d]}")
            for d in range(reg.stencil.d)
        ]
        u_plus = [
            reg.pipes.variable('scalar_t', f'u_plus_{d}_{force_id}',
                               f"{u_shift[d]} + {du[d]}")
            for d in range(reg.stencil.d)
        ]

        # 6. Berechne feq_plus = feq(u_plus)
        # (Entspricht: feq_plus = self.compute_feq(rho, u_plus))
        feq_plus = [
            reg.equilibrium.f_eq(reg, q, rho=rho, u=u_plus)
            for q in range(reg.stencil.q)
        ]

        # 7. Berechne si = feq_plus - feq_shift
        # (Entspricht: source = feq_plus - feq in source_term)
        si = [
            reg.pipes.variable('scalar_t', f'si_{q}_{force_id}',
                               f"{feq_plus[q]} - {feq_shift[q]}")
            for q in range(reg.stencil.q)
        ]

        # 8. Finales Schema: f = f - relax*(f - feq_shift) + si
        for q in range(reg.stencil.q):
            f_q = reg.f_reg(q)
            reg.pipe.append(f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {feq_shift[q]})) + {si[q]};")

