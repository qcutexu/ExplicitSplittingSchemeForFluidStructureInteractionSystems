# This code is for the paper: ERROR ANALYSIS OF THE EXPLICIT SPLITTING SCHEME FOR
# FLUID-POROELASTIC STRUCTURE INTERACTION PROBLEMS 
# Yifan Wang, Apr 2026
from fenics import *
import numpy as np
import os

# --- robust settings for macOS MPICH/OFI ---
os.environ.setdefault("FI_PROVIDER", "tcp")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

"""
MPI (2 ranks) version rewritten from the original 'NS_Biot_real_solution 2.py' (Backward Euler splitting),
with temporal + spatial sweep loops.

- NO Python multiprocessing/Pipe
- rank 0 solves fluid; rank 1 solves Biot
- coupling exchanged with mpi4py using BLOCKING Send/Recv (avoids OFI nonblocking progress stalls on macOS)
- robust DOF ordering: transfer fluid velocity using a standalone VectorFunctionSpace on both ranks

Run:
  $CONDA_PREFIX/bin/mpiexec -n 2 python3 NS_Biot_real_solution2_mpi_be_sweep.py
"""

from mpi4py import MPI as pyMPI
comm = pyMPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if size != 2:
    if rank == 0:
        print("ERROR: This script must be run with exactly 2 MPI ranks.", flush=True)
    raise SystemExit(1)

fenics_comm = MPI.comm_self  # keep FEniCS objects serial on each rank

# ----------------------
# Parameters 
# ----------------------
rho_f    = 1.0
mu_f     = 1.0
rho_b    = 1.0
mu_b     = 1.0
lambda_b = 1.0
alpha    = 1.0
C0       = 1.0
K        = 1.0
C_BJS    = 1.0

def epsilon(u):
    return sym(nabla_grad(u))

def make_expr():
    # Fluid exact + sources
    u_D = Expression(('pi*cos(pi*t)*(cos(x[1])-3*x[0])',
                      'pi*cos(pi*t)*(x[1]+1.0)'),
                     degree=5, t=0.0)
    p_D = Expression('exp(t)*sin(pi*x[0])*cos(0.5*pi*x[1])+2.0*pi*cos(pi*t)',
                     degree=5, t=0.0)
    f_f = Expression(('pi*cos(pi*t)*cos(x[1]) + pow(pi, 2)*sin(pi*t)*(3.0*x[0] - cos(x[1])) + pi*exp(t)*cos(pi*x[0])*cos((pi*x[1])/2.0)',
                      '-pow(pi,2)*sin(pi*t)*(x[1] + 1) - (pi*exp(t)*sin(pi*x[0])*sin((pi*x[1])/2.0))/2.0'),
                     degree=5, t=0.0)
    f_m = Expression('-2.0*pi*cos(pi*t)', degree=5, t=0.0)

    # Biot exact + sources
    eta_D = Expression(('sin(pi*t)*(cos(x[1])-3.0*x[0])',
                        'sin(pi*t)*(x[1]+1.0)'),
                       degree=5, t=0.0)
    ksi_D = Expression(('-pi*cos(pi*t)*(3.0*x[0] - cos(x[1]))',
                        'pi*cos(pi*t)*(x[1] + 1.0)'),
                       degree=5, t=0.0)
    phi_D = Expression('exp(t)*sin(pi*x[0])*cos(0.5*pi*x[1])', degree=5, t=0.0)

    f_e = Expression(('sin(pi*t)*cos(x[1]) + pow(pi,2)*sin(pi*t)*(3*x[0] - cos(x[1])) + pi*exp(t)*cos(pi*x[0])*cos((pi*x[1])/2.0)',
                      '-pow(pi, 2)*sin(pi*t)*(x[1] + 1.0) - (pi*exp(t)*sin(pi*x[0])*sin((pi*x[1])/2.0))/2.0'),
                     degree=5, t=0.0)
    f_p = Expression('exp(t)*cos((pi*x[1])/2.0)*sin(pi*x[0]) - 2.0*pi*cos(pi*t) + (5.0*pow(pi, 2)*exp(t)*cos((pi*x[1])/2.0)*sin(pi*x[0]))/4.0',
                     degree=5, t=0.0)
    return u_D, p_D, f_f, f_m, eta_D, ksi_D, phi_D, f_e, f_p

def print_table(title, rows, is_dt=True):
    if rank != 0:
        return
    print("\n" + "="*92)
    print(title)
    print("="*92)
    if is_dt:
        print(f"{'dt':>12} {'nm':>6} {'err_u':>12} {'err_p':>12} {'err_eta':>12} {'err_ksi':>12} {'err_phi':>12} {'rate(u)':>10}")
        prev = None
        for (dt, nm, eu, ep, ee, ek, eph) in rows:
            rate = ""
            if prev is not None:
                rate = np.log(prev/eu)/np.log(2.0)
                rate = f"{rate:10.4f}"
            print(f"{dt:12.4e} {nm:6d} {eu:12.4e} {ep:12.4e} {ee:12.4e} {ek:12.4e} {eph:12.4e} {rate}")
            prev = eu
    else:
        print(f"{'h~1/nm':>12} {'nm':>6} {'err_u':>12} {'err_p':>12} {'err_eta':>12} {'err_ksi':>12} {'err_phi':>12} {'rate(u)':>10}")
        prev = None
        for (nm, eu, ep, ee, ek, eph) in rows:
            h = 1.0/nm
            rate = ""
            if prev is not None:
                rate = np.log(prev/eu)/np.log(2.0)
                rate = f"{rate:10.4f}"
            print(f"{h:12.4e} {nm:6d} {eu:12.4e} {ep:12.4e} {ee:12.4e} {ek:12.4e} {eph:12.4e} {rate}")
            prev = eu

def solve_case(dt_float: float, nm: int, T: float):
    """
    Solve one BE case with given dt and mesh resolution nm, and return max-in-time L2 errors for:
    u, p, eta, ksi, phi.
    """
    nsteps = int(round(T/dt_float))
    dt = Constant(dt_float)

    # broadcast to keep ranks in sync
    dt_float, nm, nsteps, T = comm.bcast((dt_float, nm, nsteps, T), root=0)

    u_D, p_D, f_f, f_m, eta_D, ksi_D, phi_D, f_e, f_p = make_expr()

    # ----------------------
    # Rank 0: fluid
    # ----------------------
    if rank == 0:
        mesh_f = RectangleMesh(fenics_comm, Point(0.0, 0.0), Point(1.0, 1.0), nm, nm)
        mesh_b = RectangleMesh(fenics_comm, Point(0.0, 0.0), Point(1.0, -1.0), nm, nm)

        n_f = FacetNormal(mesh_f)
        tor_f = as_vector((-n_f[1], n_f[0]))

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

        u_elem = VectorElement("CG", mesh_f.ufl_cell(), 2)
        p_elem = FiniteElement("CG", mesh_f.ufl_cell(), 1)
        V_f = FunctionSpace(mesh_f, MixedElement([u_elem, p_elem]))
        u, p = TrialFunctions(V_f)
        v, q = TestFunctions(V_f)

        # Standalone velocity space for robust MPI transfer
        Vuf0 = VectorFunctionSpace(mesh_f, "CG", 2)
        u_send_fn = Function(Vuf0)

        # Biot coupling spaces on mesh_b (to store received eta/ksi/phi)
        eta_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        ksi_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        phi_elem = FiniteElement("CG", mesh_b.ufl_cell(), 2)
        V_b = FunctionSpace(mesh_b, MixedElement([eta_elem, ksi_elem, phi_elem]))
        Biot_old = Function(V_b)
        eta_old, ksi_old, phi_old = Biot_old.split(True)

        NS_old = Function(V_f)
        u_old, p_old = NS_old.split(True)
        NS_present = Function(V_f)

        # BCs
        u_left_bc  = DirichletBC(V_f.sub(0), u_D, bmf_f, 4)
        u_top_bc   = DirichletBC(V_f.sub(0), u_D, bmf_f, 3)
        u_right_bc = DirichletBC(V_f.sub(0), u_D, bmf_f, 2)
        p_left_bc  = DirichletBC(V_f.sub(1), p_D, bmf_f, 4)
        p_top_bc   = DirichletBC(V_f.sub(1), p_D, bmf_f, 3)
        p_right_bc = DirichletBC(V_f.sub(1), p_D, bmf_f, 2)
        bc_f = [u_left_bc, u_top_bc, u_right_bc, p_left_bc, p_top_bc, p_right_bc]

        # init at t=0
        u_D.t = 0.0; p_D.t = 0.0
        u_old.interpolate(u_D)
        p_old.interpolate(p_D)

        # constants
        rhoF = Constant(rho_f)
        muF  = Constant(mu_f)
        Cbjs = Constant(C_BJS)

        # BE fluid forms 
        a_f = rhoF*dot(u, v)*dx \
            + 2*dt*muF*inner(epsilon(u), epsilon(v))*dx \
            - dt*p*div(v)*dx \
            + dt*q*div(u)*dx \
            + dt*dot(u, n_f)*dot(v, n_f)*ds_f(5) \
            + dt*Cbjs*dot(u, tor_f)*dot(v, tor_f)*ds_f(5)

        L_f = dt*dot(f_f, v)*dx \
            + rhoF*dot(u_old, v)*dx \
            + dt*q*f_m*dx \
            + dt*Cbjs*dot(ksi_old, tor_f)*dot(v, tor_f)*ds_f(5) \
            + dt*dot(u_old, n_f)*dot(v, n_f)*ds_f(5) \
            - dt*phi_old*dot(v, n_f)*ds_f(5)

        solver_parameters = {"linear_solver": "umfpack"}

        # MPI buffers
        send_u = np.zeros(u_send_fn.vector().local_size(), dtype=np.float64)
        recv_eta = np.zeros(eta_old.vector().local_size(), dtype=np.float64)
        recv_ksi = np.zeros(ksi_old.vector().local_size(), dtype=np.float64)
        recv_phi = np.zeros(phi_old.vector().local_size(), dtype=np.float64)

        err_u = 0.0
        err_p = 0.0

        t = 0.0
        for nstep in range(nsteps):
            t += dt_float

            # blocking exchange: send u^n, receive (eta^n,ksi^n,phi^n)
            u_send_fn.assign(u_old)
            send_u[:] = u_send_fn.vector().get_local()
            comm.Send(send_u, dest=1, tag=10)
            comm.Recv(recv_eta, source=1, tag=20)
            comm.Recv(recv_ksi, source=1, tag=21)
            comm.Recv(recv_phi, source=1, tag=22)

            eta_old.vector()[:] = recv_eta; eta_old.set_allow_extrapolation(True)
            ksi_old.vector()[:] = recv_ksi; ksi_old.set_allow_extrapolation(True)
            phi_old.vector()[:] = recv_phi; phi_old.set_allow_extrapolation(True)

            # update expressions
            u_D.t = t; p_D.t = t; f_f.t = t; f_m.t = t

            solve(a_f == L_f, NS_present, bc_f, solver_parameters=solver_parameters)
            u_fem, p_fem = NS_present.split(True)

            eu = np.sqrt(assemble(dot(u_D - u_fem, u_D - u_fem)*dx(domain=mesh_f)))
            ep = np.sqrt(assemble((p_D - p_fem)*(p_D - p_fem)*dx(domain=mesh_f)))
            err_u = max(err_u, eu)
            err_p = max(err_p, ep)

            u_old.assign(u_fem)
            p_old.assign(p_fem)

        # ask rank1 to send its error triple
        comm.Send(np.array([1], dtype=np.int32), dest=1, tag=98)
        biot_err = np.empty(3, dtype=np.float64)
        comm.Recv(biot_err, source=1, tag=77)
        err_eta, err_ksi, err_phi = biot_err.tolist()

        return err_u, err_p, err_eta, err_ksi, err_phi

    # ----------------------
    # Rank 1: biot
    # ----------------------
    else:
        mesh_f = RectangleMesh(fenics_comm, Point(0.0, 0.0), Point(1.0, 1.0), nm, nm)
        mesh_b = RectangleMesh(fenics_comm, Point(0.0, 0.0), Point(1.0, -1.0), nm, nm)

        n_b = FacetNormal(mesh_b)
        tor_b = as_vector((-n_b[1], n_b[0]))

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

        eta_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        ksi_elem = VectorElement("CG", mesh_b.ufl_cell(), 2)
        phi_elem = FiniteElement("CG", mesh_b.ufl_cell(), 2)
        V_b = FunctionSpace(mesh_b, MixedElement([eta_elem, ksi_elem, phi_elem]))
        eta, ksi, phi = TrialFunctions(V_b)
        vp, zeta, psi = TestFunctions(V_b)

        Biot_old = Function(V_b)
        eta_old, ksi_old, phi_old = Biot_old.split(True)
        Biot_present = Function(V_b)

        # Standalone fluid velocity receive space must match rank0 send space
        Vuf1 = VectorFunctionSpace(mesh_f, "CG", 2)
        u_old = Function(Vuf1)

        # BCs for Biot
        eta_bottom_bc = DirichletBC(V_b.sub(0), eta_D, bmf_b, 1)
        ksi_bottom_bc = DirichletBC(V_b.sub(1), ksi_D, bmf_b, 1)
        phi_bottom_bc = DirichletBC(V_b.sub(2), phi_D, bmf_b, 1)
        eta_right_bc  = DirichletBC(V_b.sub(0), eta_D, bmf_b, 2)
        ksi_right_bc  = DirichletBC(V_b.sub(1), ksi_D, bmf_b, 2)
        phi_right_bc  = DirichletBC(V_b.sub(2), phi_D, bmf_b, 2)
        eta_left_bc   = DirichletBC(V_b.sub(0), eta_D, bmf_b, 4)
        ksi_left_bc   = DirichletBC(V_b.sub(1), ksi_D, bmf_b, 4)
        phi_left_bc   = DirichletBC(V_b.sub(2), phi_D, bmf_b, 4)
        bc_b = [eta_bottom_bc, eta_right_bc, eta_left_bc,
                phi_right_bc, phi_left_bc, phi_bottom_bc,
                ksi_bottom_bc, ksi_right_bc, ksi_left_bc]

        # init biot at t=0
        eta_D.t = 0.0; ksi_D.t = 0.0; phi_D.t = 0.0
        eta_old.interpolate(eta_D)
        ksi_old.interpolate(ksi_D)
        phi_old.interpolate(phi_D)

        # constants
        rhoB = Constant(rho_b)
        muB  = Constant(mu_b)
        lamB = Constant(lambda_b)
        Alp  = Constant(alpha)
        C0c  = Constant(C0)
        Kc   = Constant(K)
        Cbjs = Constant(C_BJS)

        # BE Biot forms 
        a_be = rhoB*dot(ksi, zeta)*dx \
             + 2*dt*muB*inner(epsilon(eta), epsilon(zeta))*dx \
             + dt*lamB*div(eta)*div(zeta)*dx \
             - dt*Alp*phi*div(zeta)*dx \
             + dt*dot(ksi, vp)*dx \
             - dot(eta, vp)*dx \
             + dt*dot(ksi, n_b)*dot(zeta, n_b)*ds_b(5) \
             + dt*Cbjs*dot(ksi, tor_b)*dot(zeta, tor_b)*ds_b(5)

        L_be = rhoB*dot(ksi_old, zeta)*dx \
             - dot(eta_old, vp)*dx \
             + dt*dot(f_e, zeta)*dx \
             + dt*Cbjs*dot(u_old, tor_b)*dot(zeta, tor_b)*ds_b(5) \
             - dt*phi_old*dot(zeta, n_b)*ds_b(5) \
             + dt*dot(ksi_old, n_b)*dot(zeta, n_b)*ds_b(5)

        a_bp = C0c*phi*psi*dx \
             + dt*Alp*div(ksi)*psi*dx \
             + dt*Kc*dot(grad(phi), grad(psi))*dx \
             + dt*phi*psi*ds_b(5)

        L_bp = dt*f_p*psi*dx \
             + C0c*phi_old*psi*dx \
             - dt*dot(u_old, n_b)*psi*ds_b(5) \
             + dt*dot(ksi_old, n_b)*psi*ds_b(5) \
             + dt*phi_old*psi*ds_b(5)

        a_b = a_be + a_bp
        L_b = L_be + L_bp

        solver_parameters = {"linear_solver": "umfpack"}

        # MPI buffers
        recv_u = np.zeros(u_old.vector().local_size(), dtype=np.float64)
        send_eta = np.zeros(eta_old.vector().local_size(), dtype=np.float64)
        send_ksi = np.zeros(ksi_old.vector().local_size(), dtype=np.float64)
        send_phi = np.zeros(phi_old.vector().local_size(), dtype=np.float64)

        err_eta = 0.0
        err_ksi = 0.0
        err_phi = 0.0

        t = 0.0
        for nstep in range(nsteps):
            t += dt_float

            # blocking exchange: recv u^n, send (eta^n,ksi^n,phi^n)
            comm.Recv(recv_u, source=0, tag=10)
            u_old.vector()[:] = recv_u
            u_old.set_allow_extrapolation(True)

            send_eta[:] = eta_old.vector().get_local()
            send_ksi[:] = ksi_old.vector().get_local()
            send_phi[:] = phi_old.vector().get_local()
            comm.Send(send_eta, dest=0, tag=20)
            comm.Send(send_ksi, dest=0, tag=21)
            comm.Send(send_phi, dest=0, tag=22)

            # update expressions
            eta_D.t = t; ksi_D.t = t; phi_D.t = t; f_e.t = t; f_p.t = t

            solve(a_b == L_b, Biot_present, bc_b, solver_parameters=solver_parameters)
            eta_fem, ksi_fem, phi_fem = Biot_present.split(True)

            ee = np.sqrt(assemble(dot(eta_D - eta_fem, eta_D - eta_fem)*dx(domain=mesh_b)))
            ek = np.sqrt(assemble(dot(ksi_D - ksi_fem, ksi_D - ksi_fem)*dx(domain=mesh_b)))
            eph = np.sqrt(assemble((phi_D - phi_fem)*(phi_D - phi_fem)*dx(domain=mesh_b)))
            err_eta = max(err_eta, ee)
            err_ksi = max(err_ksi, ek)
            err_phi = max(err_phi, eph)

            eta_old.assign(eta_fem)
            ksi_old.assign(ksi_fem)
            phi_old.assign(phi_fem)

        # wait for end-of-case signal and return errors
        flag = np.empty(1, dtype=np.int32)
        comm.Recv(flag, source=0, tag=98)
        comm.Send(np.array([err_eta, err_ksi, err_phi], dtype=np.float64), dest=0, tag=77)
        return None

# ----------------------
# Sweep schedules
# ----------------------
def temporal_sweep():
    # BE should show ~1st-order in time when spatial error small
    dt_list = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    dt_min = min(dt_list)
    nm_fixed = int(round(1.0/(5.0*dt_min)))  # moderate; adjust if needed
    T = 0.2

    rows = []
    for dtv in dt_list:
        if rank == 0:
            eu, ep, ee, ek, eph = solve_case(dtv, nm_fixed, T)
            rows.append((dtv, nm_fixed, eu, ep, ee, ek, eph))
        else:
            solve_case(dtv, nm_fixed, T)
    print_table("Temporal convergence (BE, lagged coupling)", rows, is_dt=True)

def spatial_sweep():
    dt_fixed = 1.25e-3
    nm_list = [20, 40, 80]  # 320 can be heavy on laptop; add if you want
    T = 0.2

    rows = []
    for nm in nm_list:
        if rank == 0:
            eu, ep, ee, ek, eph = solve_case(dt_fixed, nm, T)
            rows.append((nm, eu, ep, ee, ek, eph))
        else:
            solve_case(dt_fixed, nm, T)
    print_table("Spatial convergence (BE, lagged coupling)", rows, is_dt=False)

if __name__ == "__main__":
    if rank == 0:
        print("[rank0] BE sweep starting", flush=True)
    else:
        print("[rank1] BE sweep starting", flush=True)

    temporal_sweep()
    comm.Barrier()
    spatial_sweep()
    comm.Barrier()

    if rank == 0:
        print("[rank0] BE sweep done", flush=True)
    else:
        print("[rank1] BE sweep done", flush=True)
