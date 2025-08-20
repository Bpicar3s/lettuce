from ... import NativeBoundary

__all__ = ['Wallfunction']
class Wallfunction(NativeBoundary):
    """
    CUDA-native Wall Function Boundary (D3Q19) mit Spalding-Newton auf dem Device.
    Macht:
      1) Opposite-Swap auf Wandzellen
      2) u_tau Device-Newton (Spalding) bei y=0.5
      3) tau_w = rho_nb * u_tau^2, Zerlegung in tx/tz entlang (ux,uz)
      4) Exakte f-Zuweisungen wie im Python-Pfad (bottom/top)
    Erwartete Kernel-Args:
      - const scalar_t* rho, ux, uz     [Nx*Ny*Nz]
      - scalar_t nu                     (viscosity in LU)
      - int stride_y                    (= nx)
      - int ny                          (Gitterpunkte in y)
    Optional (auskommentiert): Metrik-Accus für u_tau/y+/_Re_tau via atomicAdd.
    """
    def __init__(self, mask, stencil, h, context: 'Context',
                 wall='bottom', kappa=0.4187, B=5.5, max_iter=10, tol=1e-8):
        self.context = context
        self.mask = self.context.convert_to_tensor(mask)
        self.stencil = stencil
        self.h = h
        self.wall = wall  # 'bottom' oder 'top'
        self.kappa = float(kappa)
        self.B = float(B)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Monitoring (optional, nicht im Kernel berechnet)
        import torch
        self.u_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.y_plus_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.Re_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)

    # Fahrt nur native
    def native_available(self) -> bool:
        return True

    def make_no_collision_mask(self, f_shape, context):
        # Wird für den if-Guard im Kernel verwendet:
        return self.mask

    def make_no_streaming_mask(self, f_shape, context):
        return None

    def native_generator(self):
        wall = self.wall
        KAPPA = self.kappa
        B = self.B
        MAX_IT = self.max_iter
        TOL = self.tol

        def gen(g):
            buf = g.append_pipeline_buffer
            ncm = g.support_no_collision_mask

            # --- Kernel-Argumente (Passe diese calls an dein Generator-API an) ---
            # Felder: rho, ux, uz (ganze Domäne, SoA), Viskosität nu (Skalar), stride_y (=nx), ny (int)
            g.require_field_arg("rho")   # const scalar_t* rho
            g.require_field_arg("ux")    # const scalar_t* ux
            g.require_field_arg("uz")    # const scalar_t* uz
            g.require_scalar_arg("nu")   # scalar_t nu
            g.require_int_arg("stride_y")
            g.require_int_arg("ny")

            # --- Device-Helper (einmalig) ---
            g.emit_once("wfb_helpers", r"""
            __device__ __forceinline__ scalar_t wfb_safe_hypot(scalar_t a, scalar_t b){
              scalar_t s = a*a + b*b;
              const scalar_t eps = (scalar_t)1e-30;
              return sqrt(s < eps ? eps : s);
            }

            __device__ __forceinline__ scalar_t wfb_clamp_min(scalar_t x, scalar_t m){
              return x < m ? m : x;
            }

            // Spalding-Newton für u_tau (y=distance to wall in LU, U=tangential speed magnitude).
            __device__ __forceinline__
            scalar_t wfb_solve_utau_spalding(scalar_t y, scalar_t U, scalar_t nu,
                                             int max_it, scalar_t tol,
                                             scalar_t KAPPA, scalar_t B) {
              const scalar_t eps = (scalar_t)1e-14;
              U  = wfb_clamp_min(U,  eps);
              y  = wfb_clamp_min(y,  eps);
              nu = wfb_clamp_min(nu, eps);

              const scalar_t A = expf(-KAPPA * B); // A = exp(-kappa*B)

              // Start: viskose Näherung, dann falls y+ groß -> log-Kick
              scalar_t ut = sqrtf(wfb_clamp_min((U * nu) / y, eps));
              scalar_t yplus0 = (y * ut) / nu;
              if (yplus0 >= (scalar_t)11.0f) {
                scalar_t denom = (1.0f/KAPPA) * logf(wfb_clamp_min((y * ut) / nu, eps)) + B;
                denom = wfb_clamp_min(denom, (scalar_t)1e-8f);
                ut = wfb_clamp_min(U / denom, eps);
              }

              for (int it=0; it<max_it; ++it){
                scalar_t uplus = U / wfb_clamp_min(ut, eps);
                scalar_t ku = KAPPA * uplus;

                // numerisch stabil
                ku = fminf(fmaxf(ku, -50.0f), 50.0f);
                scalar_t expku = expf(ku);

                scalar_t rhs = uplus + A * (expku - 1.0f - ku - 0.5f*ku*ku - (1.0f/6.0f)*ku*ku*ku);
                scalar_t lhs = (y * ut) / nu;
                scalar_t F   = lhs - rhs;

                scalar_t drhs_duplus =
                    1.0f + A * (KAPPA*expku - KAPPA - (KAPPA*KAPPA)*uplus - 0.5f*(KAPPA*KAPPA*KAPPA)*uplus*uplus);
                scalar_t duplus_dut = -U / (ut*ut + eps);
                scalar_t dF = (y/nu) - drhs_duplus * duplus_dut;
                if (fabsf(dF) < (scalar_t)1e-14f) dF = copysignf((scalar_t)1e-14f, dF) + (scalar_t)1e-14f;

                scalar_t delta = F / dF;
                ut = wfb_clamp_min(ut - delta, eps);

                if (fabsf(delta) < tol) break;
              }
              return ut;
            }
            """)

            # --- IF-Guard: nur Wandzellen ---
            # wenn du Indexe nutzt: f"if (no_collision_mask[node_index] == {self.index})"
            buf(f"if (no_collision_mask[node_index])", cond=ncm)
            buf("{")

            # --- Nachbar-Index (erste Fluidzelle neben der Wand) ---
            if wall == 'bottom':
                buf("  const int nb = node_index + stride_y;")
            else:  # 'top'
                buf("  const int nb = node_index - stride_y;")

            # --- Nachbarwerte laden ---
            buf("  const scalar_t rho_nb = rho[nb];")
            buf("  const scalar_t ux_nb  = ux[nb];")
            buf("  const scalar_t uz_nb  = uz[nb];")
            buf("  const scalar_t Umag   = wfb_safe_hypot(ux_nb, uz_nb);")

            # --- u_tau an der Wand (y=0.5 LU), y+, Re_tau ---
            buf(f"  const scalar_t dy = (scalar_t)0.5f;")
            buf(f"  const scalar_t KAPPA = (scalar_t){KAPPA}f;")
            buf(f"  const scalar_t Bc    = (scalar_t){B}f;")
            buf(f"  const int MAXIT      = {MAX_IT};")
            buf(f"  const scalar_t TOL   = (scalar_t){TOL}f;")

            buf("  const scalar_t utau = wfb_solve_utau_spalding(dy, Umag, nu, MAXIT, TOL, KAPPA, Bc);")
            buf("  const scalar_t yplus = (dy * utau) / nu;")
            buf("  const scalar_t retau = ((scalar_t)0.5f * (scalar_t)ny) * utau / nu;")

            # --- Scherspannung und Zerlegung ---
            buf("  const scalar_t tauw = rho_nb * utau * utau;")
            buf("  const scalar_t mag  = wfb_safe_hypot(ux_nb, uz_nb);")
            buf("  const scalar_t tx   = -(ux_nb / mag) * (scalar_t)0.5f * tauw;")
            buf("  const scalar_t tz   = -(uz_nb / mag) * (scalar_t)0.5f * tauw;")

            # --- Alte f_i cachen (genau deine Indizes) ---
            if wall == 'bottom':
                for comp, nm in ((17,"f17_old"), (16,"f16_old"), (10,"f10_old"), (8,"f8_old")):
                    buf(f"  const scalar_t {nm} = f_reg[{comp}];")
            else:
                for comp, nm in ((15,"f15_old"), (18,"f18_old"), (7,"f7_old"), (9,"f9_old")):
                    buf(f"  const scalar_t {nm} = f_reg[{comp}];")

            # --- Opposite-Swap wie im Python-Pfad ---
            buf("  scalar_t tmp[q];")
            for i in range(g.stencil.q):
                buf(f"  tmp[{i}] = f_reg[{g.stencil.opposite[i]}];")
            for i in range(g.stencil.q):
                buf(f"  f_reg[{i}] = tmp[{i}];")

            # --- Korrekturzuweisungen (IDENTISCH zu deinem Code) ---
            if wall == 'bottom':
                # x-Scherung
                buf("  f_reg[15] = f17_old + tx;")
                buf("  f_reg[16] = f17_old + tx;")
                buf("  f_reg[18] = f16_old - tx;")
                buf("  f_reg[17] = f16_old - tx;")
                # z-Scherung
                buf("  f_reg[7]  = f10_old + tz;")
                buf("  f_reg[8]  = f10_old + tz;")
                buf("  f_reg[9]  = f8_old  - tz;")
                buf("  f_reg[10] = f8_old  - tz;")
            else:
                # x-Scherung
                buf("  f_reg[17] = f15_old + tx;")
                buf("  f_reg[18] = f15_old + tx;")
                buf("  f_reg[16] = f18_old - tx;")
                buf("  f_reg[15] = f18_old - tx;")
                # z-Scherung
                buf("  f_reg[10] = f7_old + tz;")
                buf("  f_reg[9]  = f7_old + tz;")
                buf("  f_reg[8]  = f9_old - tz;")
                buf("  f_reg[7]  = f9_old - tz;")

            # --- Optional: Atomics für Mittelwerte (wenn gewünscht) ---
            # g.require_field_arg("acc_u_tau"); g.require_field_arg("acc_y_plus");
            # g.require_field_arg("acc_re_tau"); g.require_field_arg("acc_count");
            # buf("  atomicAdd(acc_u_tau, utau);")
            # buf("  atomicAdd(acc_y_plus, yplus);")
            # buf("  atomicAdd(acc_re_tau, retau);")
            # buf("  atomicAdd(acc_count, 1u);")

            buf("}")  # end if
        return gen
