from ... import Boundary, Flow, Context
import torch
from ...cuda_native.ext import Wallfunction

__all__ = ["WallFunction"]

import torch

# 🔧 Konstanten global festgelegt (nur einmal ändern nötig)



def solve_u_tau_exact(y, u, nu,
                                max_iter=10, tol=1e-7,
                                KAPPA=0.4187, B=5.5, damping=1.0, eps=1e-14):
        """
        Löst für einen festen Wandabstand y (Skalar) und einen Vektor u (LU)
        elementweise parallel auf der GPU:

            y+ = u+ + A [exp(k u+) - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6]
            u+ = u / u_tau,   y+ = y * u_tau / nu,   A = exp(-KAPPA*B)

        Rückgabe:
            u_tau_vec  (shape wie u)
            iter_vec   (int, benötigte Schritte je Element)
        """
        device = u.device
        dtype = u.dtype

        y = torch.as_tensor(y, device=device, dtype=dtype).clamp_min(eps)
        u = torch.as_tensor(u, device=device, dtype=dtype).clamp_min(eps)
        nu = torch.as_tensor(nu, device=device, dtype=dtype).clamp_min(eps)

        A = torch.exp(torch.as_tensor(-KAPPA * B, device=device, dtype=dtype))

        # Startwert: viskose Näherung + Log-Kick für große y+
        utau = torch.sqrt((u * nu / y).clamp_min(eps))
        yplus0 = y * utau / nu
        mask_log = yplus0 >= 11.81
        if mask_log.any():
            denom = (1.0 / KAPPA) * torch.log((y * utau / nu).clamp_min(eps)) + B
            utau = torch.where(mask_log, (u / denom.clamp_min(1e-8)).clamp_min(eps), utau)

        iters = torch.zeros_like(utau, dtype=torch.int32)
        active = torch.ones_like(utau, dtype=torch.bool)

        for _ in range(max_iter):
            if not active.any():
                break

            u_plus = u / utau.clamp_min(eps)
            ku = KAPPA * u_plus
            exp_ku = torch.exp(ku.clamp(-50, 50))

            rhs = u_plus + A * (exp_ku - 1.0 - ku - 0.5 * ku ** 2 - (1.0 / 6.0) * ku ** 3)
            lhs = y * utau / nu
            F = lhs - rhs

            drhs_duplus = 1.0 + A * (KAPPA * exp_ku - KAPPA - (KAPPA ** 2) * u_plus - 0.5 * (KAPPA ** 3) * u_plus ** 2)
            duplus_dutau = -u / utau.clamp_min(eps).pow(2)
            dF = (y / nu) - drhs_duplus * duplus_dutau
            dF = torch.where(dF.abs() < 1e-14, dF.sign() * 1e-14 + (dF == 0) * 1e-14, dF)

            delta = F / dF
            utau_new = (utau - damping * delta).clamp_min(eps)

            utau = torch.where(active, utau_new, utau)
            conv = delta.abs() < tol
            just = active & conv
            active = active & (~conv)
            iters = iters + just.to(iters.dtype) + active.to(iters.dtype)

        return utau

def compute_wall_quantities(flow, dy, is_top: bool):
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


    if is_top == True:
        mask = torch.zeros(flow.resolution, dtype=torch.bool)
        mask[:, -2, :] = True
    elif is_top == False:
        mask = torch.zeros(flow.resolution, dtype=torch.bool)
        mask[:, 1, :] = True


    u = flow.u()
    viscosity = flow.units.viscosity_lu
    ny = u.shape[2]


    if method == "Spalding":

        utau = solve_u_tau_exact(
            y=dy,
            u=torch.sqrt(u[0,mask]**2+u[2,mask]**2),
            nu=viscosity,
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

    # yplus entsprechend neu berechnen
    yplus = dy * utau / viscosity

    re_tau = (ny / 2) * utau / viscosity
    return utau, yplus, re_tau





class WallFunction(Boundary):
    def __init__(self, mask, stencil, h, context: 'Context', wall = 'bottom',  kappa=0.4187, B=5.5, max_iter = 100, tol = 1e-8):
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

        u_x = u[0][mask_fluidcell]
        u_z = u[2][mask_fluidcell]
        safe_u = torch.sqrt(u_x**2 + u_z**2)

        y = torch.tensor(1, device=flow.f.device, dtype=flow.f.dtype)

        u_tau, yplus, re_tau = compute_wall_quantities(flow, y, is_top=True if self.wall == "top" else False)

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
        return self.mask

    def make_no_streaming_mask(self, f_shape, context):
        return None

    def native_available(self) -> bool:
        return True

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
    def __init__(self, mask, stencil, h, context: 'Context', wall='bottom', kappa=0.4187, B=5.5, max_iter=100,
                 tol=1e-8):
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

        y = torch.tensor(0.5, device=flow.f.device, dtype=flow.f.dtype)

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

