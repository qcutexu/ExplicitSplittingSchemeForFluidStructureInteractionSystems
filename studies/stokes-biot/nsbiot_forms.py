"""Shared BDF2/EX2 forms, identical to the independently checked reference."""
from fenics import *

def epsilon(u):
    return sym(nabla_grad(u))

def fluid_form(V_f,dt,old,remote,g,material,L,gamma,source):
    u,p=TrialFunctions(V_f);v,q=TestFunctions(V_f)
    u_old,u_old1=old['u'],old['u1']
    ksi_old,ksi_old1=remote['xi'],remote['xi1']
    phi_old,phi_old1=remote['phi'],remote['phi1']
    n_f,tor_f,dx_f,ds_f=g['n'],g['tangent'],g['dx'],g['ds']
    rho_f,mu_f=material['rho_f'],material['mu_f']
    L_1=L;C_BJS=gamma;f_f_source=source
    F_f3 = rho_f / (2.0 * dt) * dot(3.0 * u - 4.0 * u_old + u_old1, v) * dx_f + 2.0 * mu_f * inner(epsilon(u), epsilon(v)) * dx_f - p * div(v) * dx_f + div(u) * q * dx_f + L_1 * dot(u, n_f) * dot(v, n_f) * ds_f(5) + C_BJS * dot(u, tor_f) * dot(v, tor_f) * ds_f(5) - dot(f_f_source, v) * dx_f - C_BJS * dot(2.0 * ksi_old - ksi_old1, tor_f) * dot(v, tor_f) * ds_f(5) - L_1 * dot(2.0 * u_old - u_old1, n_f) * dot(v, n_f) * ds_f(5) + (2.0 * phi_old - phi_old1) * dot(v, n_f) * ds_f(5)
    return F_f3

def biot_forms(V_b,dt,old,remote,g,material,L,gamma,beta,source_e,source_p):
    eta,ksi,phi=TrialFunctions(V_b);vp,zeta,psi=TestFunctions(V_b)
    eta_old,eta_old1=old['eta'],old['eta1']
    ksi_old,ksi_old1=old['xi'],old['xi1']
    phi_old,phi_old1=old['phi'],old['phi1']
    u_old,u_old1=remote['u'],remote['u1']
    n_b,tor_b,dx_b,ds_b=g['n'],g['tangent'],g['dx'],g['ds']
    rho_b,mu_b,lambda_b,alpha,C0,K=[material[k] for k in ('rho_b','mu_b','lambda_b','alpha','C0','K')]
    C_BJS=gamma;L_3=1./L;f_e_source=source_e;f_p_source=source_p
    F_pe3 = rho_b / (2.0 * dt) * dot(3.0 * ksi - 4.0 * ksi_old + ksi_old1, zeta) * dx_b + 2.0 * mu_b * inner(epsilon(eta), epsilon(zeta)) * dx_b + lambda_b * div(eta) * div(zeta) * dx_b - alpha * phi * div(zeta) * dx_b + dot(ksi, vp) * dx_b - 1.0 / (2.0 * dt) * dot(3.0 * eta - 4.0 * eta_old + eta_old1, vp) * dx_b + C_BJS * dot(ksi, tor_b) * dot(zeta, tor_b) * ds_b(5) + phi * dot(zeta, n_b) * ds_b(5) + beta * dot(ksi, n_b) * dot(zeta, n_b) * ds_b(5) - beta * dot(2.0 * ksi_old - ksi_old1, n_b) * dot(zeta, n_b) * ds_b(5) - dot(f_e_source, zeta) * dx_b - C_BJS * dot(2.0 * u_old - u_old1, tor_b) * dot(zeta, tor_b) * ds_b(5)
    F_pd3 = C0 / (2.0 * dt) * (3.0 * phi - 4.0 * phi_old + phi_old1) * psi * dx_b + alpha * div(ksi) * psi * dx_b + K * dot(grad(phi), grad(psi)) * dx_b + L_3 * phi * psi * ds_b(5) - dot(ksi, n_b) * psi * ds_b(5) - f_p_source * psi * dx_b + dot(2.0 * u_old - u_old1, n_b) * psi * ds_b(5) - L_3 * (2.0 * phi_old - phi_old1) * psi * ds_b(5)
    return F_pe3,F_pd3
