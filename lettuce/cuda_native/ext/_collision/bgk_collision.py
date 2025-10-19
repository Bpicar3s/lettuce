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
        tau_inv = self.kernel_tau_inv(reg)

        # Falls ein Forcing vorhanden ist, generiere zuerst den Source-Term.
        # Konvention: self.force.generate(reg) gibt den Namen eines device-Arrays zurück,
        # das pro Diskretisierungsrichtung indexiert werden kann (z. B. source[i] / source[q]).
        source_array = None
        if self.force is not None:
            # häufige API-Variante: generate(reg) -> "source_var_name"
            if hasattr(self.force, "generate"):
                source_array = self.force.generate(reg)
            # defensive Fallbacks, falls deine Force andere Namen nutzt:
            elif hasattr(self.force, "generate_array"):
                source_array = self.force.generate_array(reg)
            elif hasattr(self.force, "variable_name"):
                source_array = self.force.variable_name

        # Kollisionsschritt je Richtung zusammenbauen
        for q in range(reg.stencil.q):
            f_q    = reg.f_reg(q)
            f_eq_q = reg.equilibrium.f_eq(reg, q)

            if source_array:
                # BGK + Forcing: f = f - tau_inv * (f - f_eq) + S[q]
                reg.pipe.append(f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {f_eq_q})) + {source_array}[{q}];")
            else:
                # Standard-BGK ohne Forcing
                reg.pipe.append(f"{f_q} = {f_q} - ({tau_inv} * ({f_q} - {f_eq_q}));")
