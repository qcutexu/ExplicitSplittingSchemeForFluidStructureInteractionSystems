from fenics import *
import numpy as np

"""
Spatial sweep driver for the monolithic BDF2 + AB2-extrapolated coupling code.

It runs several mesh sizes (via nm_list), chooses dt ~ h^2 so that temporal
error is negligible, and prints max-in-time L2 errors and spatial rates.

Run:
  python3 nsbiot_2nd_BDF2_AB2_spatial_sweep.py
"""

# -----------------------
# Physical constants
# -----------------------
rho_f    = 1.0
mu_f     = 1.0
rho_b    = 1.0
mu_b     = 1.0
lambda_b = 1.0
alpha    = 1.0
C0       = 1.0
K        = 1.0

C_BJS    = 1.0
L_1      = 1.0
L_2      = 1.0
L_3      = 1.0

def epsilon(u):
    return sym(nabla_grad(u))

# -----------------------
# Manufactured solution expressions
# -----------------------
def make_expr():
    u_D = Expression(('pi*cos(pi*t)*(cos(x[1])-3*x[0])',
                      'pi*cos(pi*t)*(x[1]+1.0)'),
                     degree=5, t=0.0)
    p_D = Expression('exp(t)*sin(pi*x[0])*cos(0.5*pi*x[1])+2.0*pi*cos(pi*t)',
                     degree=5, t=0.0)

    sigma_right = Expression(('-8*pi*cos(pi*t) - exp(t)*cos((pi*x[1])/2)*sin(pi*x[0])',
                              '-pi*cos(pi*t)*sin(x[1])'),
                             degree=5, t=0.0)

    f_f = Expression(('pi*cos(pi*t)*cos(x[1]) + pow(pi, 2)*sin(pi*t)*(3.0*x[0] - cos(x[1])) + pi*exp(t)*cos(pi*x[0])*cos((pi*x[1])/2.0)',
                      '-pow(pi,2)*sin(pi*t)*(x[1] + 1) - (pi*exp(t)*sin(pi*x[0])*sin((pi*x[1])/2.0))/2.0'),
                     degree=5, t=0.0)
    f_m = Expression('-2.0*pi*cos(pi*t)', degree=5, t=0.0)

    eta_D = Expression(('sin(pi*t)*(cos(x[1])-3.0*x[0])',
                        'sin(pi*t)*(x[1]+1.0)'),
                       degree=5, t=0.0)
    ksi_D = Expression(('-pi*cos(pi*t)*(3.0*x[0] - cos(x[1]))',
                        'pi*cos(pi*t)*(x[1] + 1.0)'),
                       degree=5, t=0.0)
    phi_D = Expression('exp(t)*sin(pi*x[0])*cos(0.5*pi*x[1])', degree=5, t=0.0)

    grad_phi = Expression(('pi*exp(t)*cos(pi*x[0])*cos((pi*x[1])/2.0)',
                           '-(pi*exp(t)*sin(pi*x[0])*sin((pi*x[1])/2.0))/2.0'),
                          degree=5, t=0.0)

    f_e = Expression(('sin(pi*t)*cos(x[1]) + pow(pi,2)*sin(pi*t)*(3*x[0] - cos(x[1])) + pi*exp(t)*cos(pi*x[0])*cos((pi*x[1])/2.0)',
                      '-pow(pi, 2)*sin(pi*t)*(x[1] + 1.0) - (pi*exp(t)*sin(pi*x[0])*sin((pi*x[1])/2.0))/2.0'),
                     degree=5, t=0.0)

    f_p = Expression('exp(t)*cos((pi*x[1])/2.0)*sin(pi*x[0]) - 2.0*pi*cos(pi*t) + (5.0*pow(pi, 2)*exp(t)*cos((pi*x[1])/2.0)*sin(pi*x[0]))/4.0',
                     degree=5, t=0.0)

    return u_D,p_D,sigma_right,f_f,f_m,eta_D,ksi_D,phi_D,grad_phi,f_e,f_p

# -----------------------
# One run for one mesh
# -----------------------
def run_case(nm: int, T: float = 1.0, dt_scale: float = 0.2):
    mesh_f = RectangleMesh(Point(0.0, 0.0), Point(1.0, 1.0), nm, nm)
    mesh_b = RectangleMesh(Point(0.0, 0.0), Point(1.0, -1.0), nm, nm)

    # Use h from the mesh and choose dt ~ h^2 so temporal error is negligible
    h = mesh_f.hmax()
    dt_target = 1e-4
    steps = max(2, int(np.ceil(T / dt_target)))
    dt_val = T / steps
    dt = Constant(dt_val)

    n_f = FacetNormal(mesh_f)
    tor_f = as_vector((-n_f[1], n_f[0]))
    n_b = FacetNormal(mesh_b)
    tor_b = as_vector((-n_b[1], n_b[0]))

    class Right_F(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0)
    class Left_F(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
    class Top_F(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], 1.0)
    class Interface_F(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)

    bmf_f = MeshFunction("size_t", mesh_f, mesh_f.topology().dim()-1, 0)
    Interface_F().mark(bmf_f, 5)
    Right_F().mark(bmf_f, 2)
    Top_F().mark(bmf_f, 3)
    Left_F().mark(bmf_f, 4)
    ds_f = Measure("ds", domain=mesh_f, subdomain_data=bmf_f)

    class Right_B(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[0], 1.0)
    class Left_B(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[0], 0.0)
    class Bottom_B(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], -1.0)
    class Interface_B(SubDomain):
        def inside(self, x, on_boundary): return on_boundary and near(x[1], 0.0)

    bmf_b = MeshFunction("size_t", mesh_b, mesh_b.topology().dim()-1, 0)
    Interface_B().mark(bmf_b, 5)
    Right_B().mark(bmf_b, 2)
    Bottom_B().mark(bmf_b, 1)
    Left_B().mark(bmf_b, 4)
    ds_b = Measure("ds", domain=mesh_b, subdomain_data=bmf_b)

    u_D,p_D,sigma_right,f_f,f_m,eta_D,ksi_D,phi_D,grad_phi,f_e,f_p = make_expr()

    # Spaces
    V_f = FunctionSpace(mesh_f, MixedElement([
        VectorElement("CG", mesh_f.ufl_cell(), 2),
        FiniteElement("CG", mesh_f.ufl_cell(), 1)
    ]))
    u,p = TrialFunctions(V_f)
    v,q = TestFunctions(V_f)

    V_b = FunctionSpace(mesh_b, MixedElement([
        VectorElement("CG", mesh_b.ufl_cell(), 2),
        VectorElement("CG", mesh_b.ufl_cell(), 2),
        FiniteElement("CG", mesh_b.ufl_cell(), 1)
    ]))
    eta,ksi,phi = TrialFunctions(V_b)
    vp,zeta,psi = TestFunctions(V_b)

    # BCs
    bc_f = [
        DirichletBC(V_f.sub(0), u_D, bmf_f, 4),
        DirichletBC(V_f.sub(0), u_D, bmf_f, 3),
        DirichletBC(V_f.sub(0), u_D, bmf_f, 2),
        DirichletBC(V_f.sub(1), p_D, bmf_f, 4),
        DirichletBC(V_f.sub(1), p_D, bmf_f, 3),
        DirichletBC(V_f.sub(1), p_D, bmf_f, 2)
    ]

    bc_b = [
        DirichletBC(V_b.sub(0), eta_D, bmf_b, 1),
        DirichletBC(V_b.sub(0), eta_D, bmf_b, 2),
        DirichletBC(V_b.sub(0), eta_D, bmf_b, 4),
        DirichletBC(V_b.sub(1), ksi_D, bmf_b, 1),
        DirichletBC(V_b.sub(1), ksi_D, bmf_b, 2),
        DirichletBC(V_b.sub(1), ksi_D, bmf_b, 4),
        DirichletBC(V_b.sub(2), phi_D, bmf_b, 1),
        DirichletBC(V_b.sub(2), phi_D, bmf_b, 2),
        DirichletBC(V_b.sub(2), phi_D, bmf_b, 4)
    ]

    # History functions
    NS_old  = Function(V_f);  u_old,p_old = NS_old.split(True)
    NS_old1 = Function(V_f);  u_old1,p_old1 = NS_old1.split(True)
    NS_new  = Function(V_f)

    Bi_old  = Function(V_b); eta_old,ksi_old,phi_old = Bi_old.split(True)
    Bi_old1 = Function(V_b); eta_old1,ksi_old1,phi_old1 = Bi_old1.split(True)
    Bi_new  = Function(V_b)

    # init at t=0 and t=dt (use exact MMS)
    t = 0.0
    for expr in [u_D,p_D,eta_D,ksi_D,phi_D,sigma_right,f_f,f_m,grad_phi,f_e,f_p]:
        expr.t = t
    u_old1.interpolate(u_D); p_old1.interpolate(p_D)
    eta_old1.interpolate(eta_D); ksi_old1.interpolate(ksi_D); phi_old1.interpolate(phi_D)

    t = dt_val
    for expr in [u_D,p_D,eta_D,ksi_D,phi_D,sigma_right,f_f,f_m,grad_phi,f_e,f_p]:
        expr.t = t
    u_old.interpolate(u_D); p_old.interpolate(p_D)
    eta_old.interpolate(eta_D); ksi_old.interpolate(ksi_D); phi_old.interpolate(phi_D)

    for fn in [u_old, p_old, u_old1, p_old1,
               eta_old, ksi_old, phi_old, eta_old1, ksi_old1, phi_old1]:
        try:
            fn.set_allow_extrapolation(True)
        except Exception:
            pass

    # BDF2 + AB2 coupling
    F_f3 = (rho_f/(2.0*dt))*dot(3.0*u - 4.0*u_old + 1.0*u_old1, v)*dx \
        + 2*mu_f*inner(epsilon(u), epsilon(v))*dx \
        - p*div(v)*dx + div(u)*q*dx \
        + L_1*dot(u, n_f)*dot(v, n_f)*ds_f(5) \
        + C_BJS*dot(u, tor_f)*dot(v, tor_f)*ds_f(5) \
        - dot(f_f, v)*dx - f_m*q*dx \
        - C_BJS*dot(2.0*ksi_old - ksi_old1, tor_f)*dot(v, tor_f)*ds_f(5) \
        - L_1*dot(2.0*u_old - u_old1, n_f)*dot(v, n_f)*ds_f(5) \
        + (2.0*phi_old - phi_old1)*dot(v, n_f)*ds_f(5) \
        - dot(sigma_right, v)*ds_f(2)
    a_f3, L_f3 = lhs(F_f3), rhs(F_f3)

    F_pe3 = (rho_b/(2.0*dt))*dot(3.0*ksi - 4.0*ksi_old + 1.0*ksi_old1, zeta)*dx \
        + 2*mu_b*inner(epsilon(eta), epsilon(zeta))*dx \
        + lambda_b*div(eta)*div(zeta)*dx \
        - alpha*phi*div(zeta)*dx \
        + dot(ksi, vp)*dx \
        - (1.0/(2.0*dt))*dot(3.0*eta - 4.0*eta_old + 1.0*eta_old1, vp)*dx \
        + L_2*dot(ksi, n_b)*dot(zeta, n_b)*ds_b(5) \
        + C_BJS*dot(ksi, tor_b)*dot(zeta, tor_b)*ds_b(5) \
        + phi*dot(zeta, n_b)*ds_b(5) \
        - dot(f_e, zeta)*dx \
        - C_BJS*dot(2.0*u_old - u_old1, tor_b)*dot(zeta, tor_b)*ds_b(5) \
        - L_2*dot(2.0*ksi_old - ksi_old1, n_b)*dot(zeta, n_b)*ds_b(5)

    F_pd3 = (C0/(2.0*dt))*(3.0*phi - 4.0*phi_old + 1.0*phi_old1)*psi*dx \
        + alpha*div(ksi)*psi*dx \
        + K*dot(grad(phi), grad(psi))*dx \
        + L_3*phi*psi*ds_b(5) \
        - dot(ksi, n_b)*psi*ds_b(5) \
        - f_p*psi*dx \
        + dot(2.0*u_old - u_old1, n_b)*psi*ds_b(5) \
        - dot(grad_phi, n_b)*psi*ds_b(1) \
        - L_3*(2.0*phi_old - phi_old1)*psi*ds_b(5)

    a_b3, L_b3 = lhs(F_pe3 + F_pd3), rhs(F_pe3 + F_pd3)

    solver_parameters = {"linear_solver": "umfpack"}

    err_u = err_p = err_eta = err_ksi = err_phi = 0.0

    # main loop: start at step=2 (since we have n-1 and n)
    t = dt_val
    for n in range(2, steps + 1):
        t += dt_val
        for expr in [u_D,p_D,eta_D,ksi_D,phi_D,sigma_right,f_f,f_m,grad_phi,f_e,f_p]:
            expr.t = t

        solve(a_f3 == L_f3, NS_new, bc_f, solver_parameters=solver_parameters)
        solve(a_b3 == L_b3, Bi_new, bc_b, solver_parameters=solver_parameters)

        u_new,p_new = NS_new.split(True)
        eta_new,ksi_new,phi_new = Bi_new.split(True)

        # max-in-time L2 errors
        err_u   = max(err_u,   np.sqrt(assemble(dot(u_D-u_new,   u_D-u_new)   * dx(domain=mesh_f))))
        err_p   = max(err_p,   np.sqrt(assemble((p_D-p_new)      * (p_D-p_new) * dx(domain=mesh_f))))
        err_eta = max(err_eta, np.sqrt(assemble(dot(eta_D-eta_new, eta_D-eta_new) * dx(domain=mesh_b))))
        err_ksi = max(err_ksi, np.sqrt(assemble(dot(ksi_D-ksi_new, ksi_D-ksi_new) * dx(domain=mesh_b))))
        err_phi = max(err_phi, np.sqrt(assemble((phi_D-phi_new)   * (phi_D-phi_new) * dx(domain=mesh_b))))

        # shift history
        u_old1.assign(u_old); p_old1.assign(p_old)
        eta_old1.assign(eta_old); ksi_old1.assign(ksi_old); phi_old1.assign(phi_old)

        u_old.assign(u_new); p_old.assign(p_new)
        eta_old.assign(eta_new); ksi_old.assign(ksi_new); phi_old.assign(phi_new)

        for fn in [u_old, u_old1, eta_old, ksi_old, phi_old, eta_old1, ksi_old1, phi_old1]:
            try:
                fn.set_allow_extrapolation(True)
            except Exception:
                pass

    return h, dt_val, err_u, err_p, err_eta, err_ksi, err_phi

def main():
    # refine the mesh; h will be approximately halved each time
    nm_list = [10, 20, 40, 80]
    T = 0.1
    dt_scale = 0.2   # dt ~ 0.2*h^2

    rows = []
    for nm in nm_list:
        h, dt_val, eu, ep, ee, ek, eph = run_case(nm=nm, T=T, dt_scale=dt_scale)
        rows.append((nm, h, dt_val, eu, ep, ee, ek, eph))

    print("\n" + "="*120)
    print("Spatial convergence — BDF2 + AB2 coupling (with dt ~ h^2)")
    print("="*120)
    print(f"{'nm':>6} {'h':>12} {'dt':>12} {'err_u':>14} {'err_p':>14} {'err_eta':>14} "
          f"{'err_ksi':>14} {'err_phi':>14} {'rate(u)':>10} {'rate(p)':>10} "
          f"{'rate(eta)':>10} {'rate(ksi)':>10} {'rate(phi)':>10}")

    prev_h = None
    prev_u = prev_p = prev_eta = prev_ksi = prev_phi = None

    for nm, h, dt_val, eu, ep, ee, ek, eph in rows:
        rate_u = rate_p = rate_eta = rate_ksi = rate_phi = ""

        if prev_h is not None:
            if eu > 0:
                rate_u   = f"{np.log(prev_u/eu)   / np.log(prev_h/h):10.4f}"
            if ep > 0:
                rate_p   = f"{np.log(prev_p/ep)   / np.log(prev_h/h):10.4f}"
            if ee > 0:
                rate_eta = f"{np.log(prev_eta/ee) / np.log(prev_h/h):10.4f}"
            if ek > 0:
                rate_ksi = f"{np.log(prev_ksi/ek) / np.log(prev_h/h):10.4f}"
            if eph > 0:
                rate_phi = f"{np.log(prev_phi/eph)/ np.log(prev_h/h):10.4f}"

        print(f"{nm:6d} {h:12.5e} {dt_val:12.5e} {eu:14.6e} {ep:14.6e} {ee:14.6e} "
              f"{ek:14.6e} {eph:14.6e} {rate_u} {rate_p} {rate_eta} {rate_ksi} {rate_phi}")

        prev_h = h
        prev_u = eu
        prev_p = ep
        prev_eta = ee
        prev_ksi = ek
        prev_phi = eph

if __name__ == "__main__":
    main()
