from ... import NativeBoundary, Generator
import torch

__all__ = ['Wallfunction']


class Wallfunction(NativeBoundary):
    """
    CUDA-native Wall Function Boundary (D3Q19).
    """

    def __init__(self, mask: torch.Tensor, stencil, h, context: 'Context',
                 wall='bottom', kappa=0.4187, B=5.5, max_iter=20, tol=1e-8):
        self.mask = mask
        self.stencil = stencil
        self.h = h
        self.context = context
        self.wall = wall
        self.kappa = float(kappa)
        self.B = float(B)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    @staticmethod
    def create(index, **kwargs):
        instance = Wallfunction(**kwargs)
        instance.index = index
        return instance

    def generate(self, g: Generator):
        if not hasattr(g, '_math_headers_added'):
            g.append_global_buffer("#include <cmath>")
            g.append_global_buffer("#include <algorithm>")
            g._math_headers_added = True

        # 💡 KORREKTUR HIER: Alle Datenpfade zeigen jetzt auf simulation.flow
        if not g.launcher_hooked("rho_tensor"):
            g.launcher_hook("rho_tensor", "const at::Tensor rho_tensor", "rho_tensor", "simulation.flow.rho()")
        if not g.kernel_hooked("rho_tensor"):
            g.kernel_hook("rho_tensor", "const scalar_t* rho_tensor", "rho_tensor.data<scalar_t>()")

        if not g.launcher_hooked("ux_tensor"):
            g.launcher_hook("ux_tensor", "const at::Tensor ux_tensor", "ux_tensor", "simulation.flow.u()[0]")
        if not g.kernel_hooked("ux_tensor"):
            g.kernel_hook("ux_tensor", "const scalar_t* ux_tensor", "ux_tensor.data<scalar_t>()")

        if not g.launcher_hooked("uz_tensor"):
            g.launcher_hook("uz_tensor", "const at::Tensor uz_tensor", "uz_tensor", "simulation.flow.u()[2]")
        if not g.kernel_hooked("uz_tensor"):
            g.kernel_hook("uz_tensor", "const scalar_t* uz_tensor", "uz_tensor.data<scalar_t>()")

        if not g.launcher_hooked("nu"):
            g.launcher_hook("nu", "const double nu", "nu", "simulation.units.viscosity_lu")
        if not g.kernel_hooked("nu"):
            g.kernel_hook("nu", "scalar_t nu", "static_cast<scalar_t>(nu)")

        if not g.launcher_hooked("stride_y"):
            g.launcher_hook("stride_y", "const int stride_y", "stride_y", "simulation.flow.resolution[0]")
        if not g.kernel_hooked("stride_y"):
            g.kernel_hook("stride_y", "int stride_y", "stride_y")

        g.append_pipeline_buffer(f"if (no_collision_mask[node_index] == {self.index}) {{")
        g.append_pipeline_buffer(f"""
            // ---- Native WallFunction Boundary ({self.wall}, index={self.index}) ----
            int i_fluid;
            {'i_fluid = node_index + stride_y;' if self.wall == 'bottom' else 'i_fluid = node_index - stride_y;'}

            const scalar_t rho_loc = rho_tensor[i_fluid];
            const scalar_t ux_loc  = ux_tensor[i_fluid];
            const scalar_t uz_loc  = uz_tensor[i_fluid];
            const scalar_t u_mag   = sqrt(ux_loc * ux_loc + uz_loc * uz_loc);

            const scalar_t y = 1.0; 
            scalar_t u_tau = 0.0;

            if (u_mag > 1e-12) {{
                const scalar_t KAPPA = {self.kappa};
                const scalar_t B = {self.B};
                const scalar_t A = exp(-KAPPA * B);
                const scalar_t eps = 1e-14;
                u_tau = sqrt(u_mag * nu / y);

                for (int it = 0; it < {self.max_iter}; ++it) {{
                    scalar_t u_plus = u_mag / (u_tau + eps);
                    scalar_t ku = KAPPA * u_plus;
                    scalar_t rhs = u_plus + A * (exp(ku) - 1.0 - ku - 0.5 * ku * ku - (1.0/6.0) * ku * ku * ku);
                    scalar_t lhs = y * u_tau / nu;
                    scalar_t F = lhs - rhs;
                    scalar_t drhs_duplus = 1.0 + A * (KAPPA * exp(ku) - KAPPA - KAPPA*KAPPA * u_plus - 0.5 * KAPPA*KAPPA*KAPPA * u_plus*u_plus);
                    scalar_t duplus_dutau = -u_mag / ((u_tau + eps) * (u_tau + eps));
                    scalar_t dF = (y / nu) - drhs_duplus * duplus_dutau;
                    if (fabs(dF) < eps) {{ dF = eps; }}
                    scalar_t delta = F / dF;
                    u_tau = std::max(eps, u_tau - delta);
                    if (fabs(delta) < {self.tol}) {{ break; }}
                }}
            }}

            const scalar_t tau_w = rho_loc * u_tau * u_tau;
            scalar_t tau_x_field = 0.0;
            scalar_t tau_z_field = 0.0;

            if (u_mag > 1e-12) {{
                tau_x_field = -(ux_loc / u_mag) * 0.5 * tau_w;
                tau_z_field = -(uz_loc / u_mag) * 0.5 * tau_w;
            }}
        """)

        if self.wall == 'bottom':
            g.append_pipeline_buffer(f"""
                const scalar_t f17_old = f_reg[17]; const scalar_t f16_old = f_reg[16];
                const scalar_t f10_old = f_reg[10]; const scalar_t f8_old  = f_reg[8];
                f_reg[15] = f17_old + tau_x_field; f_reg[16] = f17_old + tau_x_field;
                f_reg[18] = f16_old - tau_x_field; f_reg[17] = f16_old - tau_x_field;
                f_reg[7]  = f10_old + tau_z_field; f_reg[8]  = f10_old + tau_z_field;
                f_reg[9]  = f8_old - tau_z_field; f_reg[10] = f8_old - tau_z_field;
            """)
        elif self.wall == 'top':
            g.append_pipeline_buffer(f"""
                const scalar_t f15_old = f_reg[15]; const scalar_t f18_old = f_reg[18];
                const scalar_t f7_old  = f_reg[7]; const scalar_t f9_old  = f_reg[9];
                f_reg[17] = f15_old + tau_x_field; f_reg[18] = f15_old + tau_x_field;
                f_reg[16] = f18_old - tau_x_field; f_reg[15] = f18_old - tau_x_field;
                f_reg[10] = f7_old + tau_z_field; f_reg[9]  = f7_old + tau_z_field;
                f_reg[8]  = f9_old - tau_z_field; f_reg[7]  = f9_old - tau_z_field;
            """)
        g.append_pipeline_buffer("}")