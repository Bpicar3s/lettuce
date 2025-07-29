from ... import Boundary, Flow, Context
import torch
__all__ = ["WallFunction"]

import torch

# 🔧 Konstanten global festgelegt (nur einmal ändern nötig)



def solve_u_tau_exact(y, u, nu, max_iter=100, tol=1e-8):

    KAPPA = 0.4187
    B = 5.5
    E = 9.793
    A = torch.exp(torch.tensor(-KAPPA * B, device=u.device, dtype=u.dtype))
    u_tau= torch.sqrt(u * y / nu)
    u_tau = 0.05 * u
    u_tau = torch.sqrt((u * nu) / y)
    for i in range(max_iter):
        u_plus = u / (u_tau + 1e-12)
        ku = KAPPA * u_plus
        exp_ku = torch.exp(ku)

        f_rhs = u_plus + A * (exp_ku - 1 - ku - 0.5 * ku ** 2 - (1/6) * ku ** 3)
        lhs = y * u_tau / nu
        residual = lhs - f_rhs

        d_f_rhs_duplus = 1 + A * (KAPPA * exp_ku - KAPPA - KAPPA**2 * u_plus - 0.5 * KAPPA**3 * u_plus**2)
        d_uplus_du_tau = -u / (u_tau + 1e-12)**2
        df_du_tau = d_f_rhs_duplus * d_uplus_du_tau
        d_lhs_du_tau = y / nu
        total_derivative = d_lhs_du_tau - df_du_tau

        total_derivative = torch.where(torch.abs(total_derivative) < 1e-10,
                                       torch.full_like(total_derivative, 1e-10),
                                       total_derivative)

        delta = residual / total_derivative
        delta = torch.clamp(delta, min=-0.1, max=0.1)

        u_tau_new = u_tau - delta

        if torch.max(torch.abs(delta)) < tol:
            break
        u_tau = u_tau_new

    return u_tau

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
    if is_top == True:
        mask = torch.zeros(flow.resolution, dtype=torch.bool)
        mask[:, -2, :] = True
    elif is_top == False:
        mask = torch.zeros(flow.resolution, dtype=torch.bool)
        mask[:, 1, :] = True


    u = flow.u()
    rho = flow.rho()
    viscosity = flow.units.viscosity_lu
    ny = u.shape[1]


    utau = solve_u_tau_exact(
        y=dy,
        u=torch.sqrt(u[0,mask]**2+u[2,mask]**2),
        nu=viscosity,
    )

    yplus = dy * utau / viscosity
    re_tau = (dy * ny / 2) * utau / viscosity

    return utau, yplus, re_tau





class WallFunction(Boundary):
    def __init__(self, mask, stencil, h, context: 'Context', wall = 'bottom',  kappa=0.41, B=5.2, max_iter = 100, tol = 1e-8):
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
        rho = rho[:,mask_fluidcell]
        u = flow.u()

        u_x = u[0][mask_fluidcell]
        u_z = u[2][mask_fluidcell]
        safe_u = torch.sqrt(u_x**2 + u_z**2)

        y = torch.tensor(0.5, device=flow.f.device, dtype=flow.f.dtype)

        u_tau, yplus, re_tau = compute_wall_quantities(flow, y , is_top=True if self.wall == "top" else False)

        tau_w = rho * u_tau**2

        if torch.isnan(tau_w).any() or torch.isinf(tau_w).any():
            self.previous_u_tau_mean = self.u_tau_mean.clone().detach()
            self.u_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.y_plus_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            self.Re_tau_mean = torch.tensor(0.0, device=flow.f.device, dtype=flow.f.dtype)
            return flow.f

        tau_x = - (u_x / safe_u) * tau_w
        tau_z = - (u_z / safe_u) * tau_w

        tau_x_field = torch.zeros_like(u[0])
        tau_z_field = torch.zeros_like(u[2] if self.stencil.d == 3 else u[0])
        tau_x_field[self.mask] = 0.5*tau_x
        tau_z_field[self.mask] = 0.5*tau_z

        flow.f = torch.where(self.mask, flow.f[self.stencil.opposite], flow.f)

        if self.wall == 'bottom':
            flow.f[15, self.mask] = f17_old + tau_z_field[self.mask]
            flow.f[16, self.mask] = f17_old + tau_z_field[self.mask]
            flow.f[18, self.mask] = f16_old - tau_z_field[self.mask]
            flow.f[8,  self.mask] = f16_old - tau_z_field[self.mask]
            flow.f[7,  self.mask] = f10_old + tau_x_field[self.mask]
            flow.f[17, self.mask] = f10_old + tau_x_field[self.mask]
            flow.f[9,  self.mask] = f8_old - tau_x_field[self.mask]
            flow.f[10, self.mask] = f8_old - tau_x_field[self.mask]
        elif self.wall == 'top':
            flow.f[17, self.mask] = f15_old + tau_z_field[self.mask]
            flow.f[18, self.mask] = f15_old + tau_z_field[self.mask]
            flow.f[16, self.mask] = f18_old - tau_z_field[self.mask]
            flow.f[9,  self.mask] = f18_old - tau_z_field[self.mask]
            flow.f[10, self.mask] = f7_old + tau_x_field[self.mask]
            flow.f[15, self.mask] = f7_old + tau_x_field[self.mask]
            flow.f[8,  self.mask] = f9_old - tau_x_field[self.mask]
            flow.f[7,  self.mask] = f9_old - tau_x_field[self.mask]

        self.tau_x = tau_x_field
        self.tau_z = tau_z_field
        self.previous_u_tau_mean = self.u_tau_mean.clone().detach()
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
        return False

    def native_generator(self):
        pass


