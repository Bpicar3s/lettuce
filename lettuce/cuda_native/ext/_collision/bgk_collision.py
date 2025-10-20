from typing import Optional

from lettuce.cuda_native import DefaultCodeGeneration, Parameter
from ... import NativeCollision

__all__ = ['NativeBGKCollision']


class NativeBGKCollision(NativeCollision):

    def __init__(self, index: int, force: Optional['NativeForce'] = None):
        super().__init__(index)
        self.force = force

    @staticmethod
    def create(index: int, force: Optional['NativeForce'] = None):
        if force is None:
            return None
        return NativeBGKCollision(index, force)

    def cuda_tau_inv(self, reg: 'DefaultCodeGeneration'):
        variable = f"tau_inv{hex(id(self))[2:]}"
        # erzeugt im Launcher (Host) eine double-Variable und liefert deren Namen zurück
        return reg.cuda_hook('1.0 / simulation.collision.tau', Parameter('double', variable))

    def kernel_tau_inv(self, reg: 'DefaultCodeGeneration'):
        variable = self.cuda_tau_inv(reg)
        # mapped dieselbe Variable als scalar_t in den Kernel
        return reg.kernel_hook(f"static_cast<scalar_t>({variable})", Parameter('scalar_t', variable))

    def generate(self, reg: 'DefaultCodeGeneration'):
        # tau^{-1}
        tau_inv = self.kernel_tau_inv(reg)

        # --- acceleration aus Python hooken ---
        accel_name = f"acceleration_{hex(id(self))[2:]}"
        if not hasattr(reg, "_cuda_hooked_names"):
            reg._cuda_hooked_names = set()
        if not hasattr(reg, "_kernel_hooked_names"):
            reg._kernel_hooked_names = set()

        from lettuce.cuda_native import Parameter

        def cuda_hook_once(sym, expr, param):
            if sym in reg._cuda_hooked_names: return
            reg.cuda_hook(expr, param)
            reg._cuda_hooked_names.add(sym)

        def kernel_hook_once(sym, expr, param):
            if sym in reg._kernel_hooked_names: return
            reg.kernel_hook(expr, param)
            reg._kernel_hooked_names.add(sym)

        cuda_hook_once(accel_name,
                       'simulation.collision.force.acceleration',
                       Parameter('torch::Tensor', accel_name))
        kernel_hook_once(accel_name,
                         f'{accel_name}.accessor<scalar_t, 1>()',
                         Parameter('const auto', accel_name))

        # ---------- rho(u) & u aus f_q und e_{qk} aufbauen (ohne reg.macro) ----------
        e = reg.stencil.e  # numpy-like: shape [q, d]
        w = reg.stencil.w  # [q]
        cs2 = reg.stencil.cs ** 2  # scalar
        qn = reg.stencil.q
        dn = reg.stencil.d

        # rho = sum_q f_q
        rho_terms = [f"{reg.f_reg(q)}" for q in range(qn)]
        rho = "(" + " + ".join(rho_terms) + ")"

        # momentum_k = sum_q e[q,k] * f_q
        def mom_comp(k: int) -> str:
            terms = []
            for qi in range(qn):
                coef = float(e[qi][k]) if k < dn else 0.0
                if coef == 0.0:
                    continue
                terms.append(f"({coef})*({reg.f_reg(qi)})")
            return "(0)" if not terms else "(" + " + ".join(terms) + ")"

        # u_k = momentum_k / rho
        ux = f"({mom_comp(0)})/({rho})"
        uy = "static_cast<scalar_t>(0)" if dn < 2 else f"({mom_comp(1)})/({rho})"
        uz = "static_cast<scalar_t>(0)" if dn < 3 else f"({mom_comp(2)})/({rho})"

        # u_plus = u + 0.5 * a   (GENAU wie in deiner Python-EDM, KEIN /rho!)
        ux_p = f"(({ux}) + static_cast<scalar_t>(0.5) * {accel_name}[0])"
        uy_p = "static_cast<scalar_t>(0)" if dn < 2 else f"(({uy}) + static_cast<scalar_t>(0.5) * {accel_name}[1])"
        uz_p = "static_cast<scalar_t>(0)" if dn < 3 else f"(({uz}) + static_cast<scalar_t>(0.5) * {accel_name}[2])"

        # Helper: f_eq-Ausdrücke (rein aus Literalen/Ausdrücken, kein 'u[]' im Scope)
        def f_eq_expr(q: int, ux_s: str, uy_s: str, uz_s: str) -> str:
            ex = float(e[q][0])
            ey = float(e[q][1]) if dn > 1 else 0.0
            ez = float(e[q][2]) if dn > 2 else 0.0
            eu = f"(({ex})*({ux_s}) + ({ey})*({uy_s}) + ({ez})*({uz_s}))"
            u2 = f"(({ux_s})*({ux_s}) + ({uy_s})*({uy_s}) + ({uz_s})*({uz_s}))"
            return (
                f"static_cast<scalar_t>({w[q]})*({rho})*("
                f"1 + ({eu})/({cs2}) + static_cast<scalar_t>(0.5)*(({eu})*({eu}))/(({cs2})*({cs2}))"
                f" - static_cast<scalar_t>(0.5)*({u2})/({cs2}))"
            )

        # ---------- BGK + harter EDM-Source je Richtung ----------
        for q in range(qn):
            f_q = reg.f_reg(q)
            f_eq_q = reg.equilibrium.f_eq(reg, q)  # f_eq(rho, u) (bestehende API)
            f_eq_p = f_eq_expr(q, ux_p, uy_p, uz_p)  # f_eq(rho, u + 0.5 a)
            s_q = f"(({f_eq_p}) - ({f_eq_q}))"  # Kupershtokh-Quelle

            reg.pipe.append(f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {f_eq_q})) + {s_q};")
