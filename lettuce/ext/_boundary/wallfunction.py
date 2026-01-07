from ... import Boundary, Flow, Context
import torch
from ...cuda_native.ext import Wallfunction

__all__ = ["WallFunction"]

import torch

# 🔧 Konstanten global festgelegt (nur einmal ändern nötig)



def solve_u_tau_exact(y, u, nu,
                      max_iter=10, tol=1e-6,
                      KAPPA=0.41, B=5.5,
                      damping=1.0, utau_prev=None, newton_speedup = False):
    device = u.device
    dtype = u.dtype
    eps = torch.finfo(dtype).eps

    y = torch.as_tensor(y, device=device, dtype=dtype)
    u = torch.as_tensor(u, device=device, dtype=dtype)
    nu = torch.as_tensor(nu, device=device, dtype=dtype)

    A = torch.exp(torch.as_tensor(-KAPPA * B, device=device, dtype=dtype))

    # --- Startwertwahl ---
    if utau_prev is not None and newton_speedup and torch.isfinite(utau_prev).all():
        utau = utau_prev.clone()
    else:
        utau = torch.sqrt((u * nu / y).clamp_min(eps))
        yplus0 = y * utau / nu
        mask_log = yplus0 >= 11.81
        if mask_log.any():
            utau = torch.where(mask_log,
                               (u * nu ** (1 / 7) / y ** (1 / 7) / 8.3) ** (7 / 8),
                               utau)

    iters = torch.zeros_like(utau, dtype=torch.int32)
    active = torch.ones_like(utau, dtype=torch.bool)

    for _ in range(max_iter):
            if not active.any():
                break

            u_plus = u / utau
            ku = KAPPA * u_plus
            exp_ku = torch.exp(ku)

            rhs = u_plus + A * (exp_ku - 1.0 - ku - 0.5 * ku ** 2 - (1.0 / 6.0) * ku ** 3)
            lhs = y * utau / nu
            F = lhs - rhs

            drhs_duplus = 1.0 + A * (KAPPA * exp_ku - KAPPA - (KAPPA ** 2) * u_plus - 0.5 * (KAPPA ** 3) * u_plus ** 2)
            duplus_dutau = -u / utau**2
            dF = (y / nu) - drhs_duplus * duplus_dutau

            delta = F / dF
            utau_new = (utau - damping * delta)

            utau = torch.where(active, utau_new, utau)
            conv = (delta/utau).abs() < tol
            just = active & conv
            active = active & (~conv)
            iters = iters + just.to(iters.dtype) + active.to(iters.dtype)

    mean_iters = iters.float().mean()
    max_iters = iters.max()

    return utau, mean_iters, max_iters

def compute_wall_quantities(flow, dy, is_top: bool, acceleration = 0, newton_speedup = False, utau_prev = None):
    """
    Berechnet Wandgrößen wie u_tau, y+, Re_tau für eine Wand.

    :param u: Geschwindigkeitstensor [3, Nx, Ny, Nz]
    :param rho: Dichte-Tensor [Nx, Ny, Nz]
    :param viscosity: Skalar (dynamische Viskosität)
    :param dy: Gitterabstand in y-Richtung (float)
    :param is_top: True für obere Wand, sonst untere
    :return: (u_tau, y+, Re_tau) als Tensors
    """
    method = "Spalding"

    u = flow.u()
    if is_top == True:
        mask = torch.zeros_like(u[0], dtype=torch.bool)
        mask[:, -2, :] = True
    elif is_top == False:
        mask = torch.zeros_like(u[0], dtype=torch.bool)
        mask[:, 1, :] = True



    viscosity = flow.units.viscosity_lu
    ny = flow.resolution[1]


    if method == "Spalding":

        utau, mean_it, max_it = solve_u_tau_exact(
            y=dy,
            u=torch.sqrt((u[0,mask])**2+u[2,mask]**2),
            nu=viscosity,
            newton_speedup = newton_speedup,
            utau_prev = utau_prev,
        )

    elif method == "Log-Visc":
        utau = torch.sqrt(torch.sqrt(u[0, mask] ** 2 + u[2, mask] ** 2) * viscosity / dy)
        yplus = dy * utau / viscosity

        # Maske für log-law Bereich
        loglaw_mask = yplus >= 11.81

        # Log-law utau nur für die betroffenen Stellen berechnen
        utau_log = ((u[0, mask][loglaw_mask] ** 2 + u[2, mask][loglaw_mask] ** 2) / 8.3 * (viscosity / dy) ** (1 / 7)) ** (
                    8 / 7)

        # Alte utau-Werte an diesen Stellen ersetzen
        utau[loglaw_mask] = utau_log
        mean_it = torch.tensor(1.0, device=utau.device, dtype=utau.dtype)
        max_it = torch.tensor(1.0, device=utau.device, dtype=utau.dtype)

    # yplus entsprechend neu berechnen
    yplus = dy * utau / viscosity

    re_tau = (ny / 2) * utau / viscosity
    rho = flow.rho()
    rho_wall = rho[0, mask]

    tau_w = rho_wall * (utau ** 2)
    u_tau_ref = torch.sqrt(tau_w.mean() / rho_wall.mean())
    re_tau_ref = (ny / 2) * utau.mean() / viscosity

    return utau, yplus, re_tau, u_tau_ref, re_tau_ref, mean_it, max_it





class WallFunction(Boundary):
    def __init__(self, mask, stencil, h, context: 'Context', wall = 'bottom',  kappa=0.41, B=5.5, max_iter = 10, tol = 1e-6, force=None, newton_speedup = False):
        self.context = context

        self.mask = self.context.convert_to_tensor(mask)
        self.stencil = stencil
        self.h = h
        self.wall = wall
        self.kappa = kappa
        self.B = B
        self.max_iter = max_iter
        self.tol = tol
        self.force=force
        self.utau_start = None
        self.tau_x = None
        self.tau_z = None
        self.newton_speedup = newton_speedup
        self.mean_it=None
        self.max_it=None
        self.u_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.y_plus_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.Re_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.previous_u_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)

    def __call__(self, flow: Flow):

        if self.wall == 'bottom':
            f17_old = flow.f[17, self.mask].clone()
            f16_old = flow.f[16, self.mask].clone()
            f10_old = flow.f[10, self.mask].clone()
            f8_old  = flow.f[8, self.mask].clone()

        elif self.wall == 'top':
            f15_old = flow.f[15, self.mask].clone()
            f18_old = flow.f[18, self.mask].clone()
            f7_old  = flow.f[7, self.mask].clone()
            f9_old  = flow.f[9, self.mask].clone()
        else:
            raise ValueError("wall must be 'bottom' or 'top'")

        if self.wall == 'bottom':
            mask_fluidcell = torch.zeros_like(self.mask, dtype=torch.bool)
            mask_fluidcell[:, 1, :] = True
        elif self.wall == 'top':
            mask_fluidcell = torch.zeros_like(self.mask, dtype=torch.bool)
            mask_fluidcell[:, -2, :] = True


        rho = flow.rho()
        u = flow.u()
        if self.force is None:
            u_x = u[0][mask_fluidcell]
            acceleration = 0
        else:
            u_x = u[0][mask_fluidcell]
            acceleration = self.force.acceleration[0]


        u_z = u[2][mask_fluidcell]
        safe_u = torch.sqrt(u_x**2 + u_z**2)

        y = torch.tensor(1, device=flow.f.device, dtype=flow.f.dtype)

        u_tau, yplus, re_tau, _, _, self.mean_it, self.max_it = compute_wall_quantities(flow, y,
                                                                   is_top=True if self.wall == "top" else False,
                                                                   acceleration = acceleration,
                                                                   newton_speedup = self.newton_speedup,
                                                                   utau_prev=self.utau_start
                                                                   )
        self.utau_start = u_tau
        tau_w = rho[:,mask_fluidcell] * u_tau**2

        if torch.isnan(tau_w).any() or torch.isinf(tau_w).any():
            self.previous_u_tau_mean = self.u_tau_mean.clone().detach()
            self.u_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.y_plus_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.Re_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            return flow.f

        tau_x_field = - (u_x / safe_u) * 0.5 * tau_w
        tau_z_field = - (u_z / safe_u) * 0.5 * tau_w

        flow.f = torch.where(self.mask, flow.f[self.stencil.opposite], flow.f)

        if self.wall == 'bottom':
            flow.f[15, self.mask] = f17_old + tau_x_field
            flow.f[16, self.mask] = f17_old + tau_x_field
            flow.f[18, self.mask] = f16_old - tau_x_field
            flow.f[17,  self.mask] = f16_old - tau_x_field
            flow.f[7,  self.mask] = f10_old + tau_z_field
            flow.f[8, self.mask] = f10_old + tau_z_field
            flow.f[9,  self.mask] = f8_old - tau_z_field
            flow.f[10, self.mask] = f8_old - tau_z_field
        elif self.wall == 'top':
            flow.f[17, self.mask] = f15_old + tau_x_field
            flow.f[18, self.mask] = f15_old + tau_x_field
            flow.f[16, self.mask] = f18_old - tau_x_field
            flow.f[15,  self.mask] = f18_old - tau_x_field
            flow.f[10, self.mask] = f7_old + tau_z_field
            flow.f[9, self.mask] = f7_old + tau_z_field
            flow.f[8,  self.mask] = f9_old - tau_z_field
            flow.f[7,  self.mask] = f9_old - tau_z_field

        self.u_tau_mean = u_tau.mean()
        # Lokales y_plus wie gehabt
        self.y_plus_mean = (y * u_tau / flow.units.viscosity_lu).mean()

        # Korrektes Re_tau

        self.Re_tau_mean = re_tau.mean()

        if torch.isnan(self.u_tau_mean) or torch.isinf(self.u_tau_mean) or \
           torch.isnan(self.y_plus_mean) or torch.isinf(self.y_plus_mean) or \
           torch.isnan(self.Re_tau_mean) or torch.isinf(self.Re_tau_mean):
            self.u_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.y_plus_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.Re_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)

        return flow.f



    def make_no_collision_mask(self, f_shape, context):
        return self.mask.to(torch.bool)

    def make_no_streaming_mask(self, f_shape, context):
        return None

    def native_available(self) -> bool:
        return False

    # ext/_boundary/wallfunction.py
    # In your high-level Python WallFunction class file...

    def native_generator(self, index: int) -> 'NativeBoundary':
        # Import the native class
        from lettuce.cuda_native.ext._boundary.wallfunction import Wallfunction

        # Create an instance of the native Wallfunction class
        native_instance = Wallfunction(
            mask=self.mask,
            stencil=self.stencil,
            h=self.h,
            context=self.context,
            wall=getattr(self, "wall", "bottom"),
            kappa=self.kappa,
            B=self.B,
            max_iter=self.max_iter,
            tol=self.tol
        )

        # --- THIS IS THE CRUCIAL FIX ---
        # Manually assign the index that lettuce provided.
        native_instance.index = index

        # Return the fully configured native instance
        return native_instance



class WallFunction2(Boundary):
    def __init__(self, mask, stencil, h, context: 'Context', wall='bottom', kappa=0.41, B=5.5, max_iter=10,
                 tol=1e-6):
        self.context = context

        self.mask = self.context.convert_to_tensor(mask)
        self.stencil = stencil
        self.h = h
        self.wall = wall
        self.kappa = kappa
        self.B = B
        self.max_iter = max_iter
        self.tol = tol

        self.tau_x = None
        self.tau_z = None

        self.u_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.y_plus_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.Re_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)
        self.previous_u_tau_mean = torch.tensor(0.0, device=self.context.device, dtype=self.context.dtype)

    def __call__(self, flow: Flow):

        if self.wall == 'bottom':
            f17_old = flow.f[17, self.mask].clone()
            f16_old = flow.f[16, self.mask].clone()
            f10_old = flow.f[10, self.mask].clone()
            f8_old = flow.f[8, self.mask].clone()

        elif self.wall == 'top':
            f15_old = flow.f[15, self.mask].clone()
            f18_old = flow.f[18, self.mask].clone()
            f7_old = flow.f[7, self.mask].clone()
            f9_old = flow.f[9, self.mask].clone()
        else:
            raise ValueError("wall must be 'bottom' or 'top'")

        if self.wall == 'bottom':
            mask_fluidcell = torch.zeros_like(self.mask, dtype=torch.bool)
            mask_fluidcell[:, 0, :] = True
        elif self.wall == 'top':
            mask_fluidcell = torch.zeros_like(self.mask, dtype=torch.bool)
            mask_fluidcell[:, -1, :] = True

        rho = flow.rho()
        u = flow.u()

        u_x = u[0][mask_fluidcell]
        u_z = u[2][mask_fluidcell]
        safe_u = torch.sqrt(u_x ** 2 + u_z ** 2)

        y = torch.tensor(1, device=flow.f.device, dtype=flow.f.dtype)

        u_tau, yplus, re_tau = compute_wall_quantities(flow, y, is_top=True if self.wall == "top" else False)
        tau_w = rho[:, mask_fluidcell] * u_tau ** 2

        if torch.isnan(tau_w).any() or torch.isinf(tau_w).any():
            self.previous_u_tau_mean = self.u_tau_mean.clone().detach()
            self.u_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.y_plus_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.Re_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            return flow.f

        tau_x_field = - (u_x / safe_u) * 0.5 * tau_w
        tau_z_field = - (u_z / safe_u) * 0.5 * tau_w

        flow.f = torch.where(self.mask, flow.f[self.stencil.opposite], flow.f)

        if self.wall == 'bottom':
            flow.f[15, self.mask] = f17_old + tau_x_field
            flow.f[16, self.mask] = f17_old + tau_x_field
            flow.f[18, self.mask] = f16_old - tau_x_field
            flow.f[17, self.mask] = f16_old - tau_x_field
            flow.f[7, self.mask] = f10_old + tau_z_field
            flow.f[8, self.mask] = f10_old + tau_z_field
            flow.f[9, self.mask] = f8_old - tau_z_field
            flow.f[10, self.mask] = f8_old - tau_z_field
        elif self.wall == 'top':
            flow.f[17, self.mask] = f15_old + tau_x_field
            flow.f[18, self.mask] = f15_old + tau_x_field
            flow.f[16, self.mask] = f18_old - tau_x_field
            flow.f[15, self.mask] = f18_old - tau_x_field
            flow.f[10, self.mask] = f7_old + tau_z_field
            flow.f[9, self.mask] = f7_old + tau_z_field
            flow.f[8, self.mask] = f9_old - tau_z_field
            flow.f[7, self.mask] = f9_old - tau_z_field

        self.u_tau_mean = u_tau.mean()
        # Lokales y_plus wie gehabt
        self.y_plus_mean = (y * u_tau / flow.units.viscosity_lu).mean()

        # Korrektes Re_tau

        self.Re_tau_mean = re_tau.mean()

        if torch.isnan(self.u_tau_mean) or torch.isinf(self.u_tau_mean) or \
                torch.isnan(self.y_plus_mean) or torch.isinf(self.y_plus_mean) or \
                torch.isnan(self.Re_tau_mean) or torch.isinf(self.Re_tau_mean):
            self.u_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.y_plus_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.Re_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)

        return flow.f

    def make_no_collision_mask(self, f_shape, context):
        return self.mask

    def make_no_streaming_mask(self, f_shape, context):
        return None

    def native_available(self) -> bool:
        return True

    def native_generator(self, index: int) -> 'NativeBoundary':
        return NativeBounceBackBoundary(index)

