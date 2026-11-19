"""Fixed-reference 3D Stokes–Biot BDF2/EX2 forms.

Tangential slip uses the rank-two projector I-n (x) n, rather than a single
2D tangent. Surface pullbacks/integration measures must be provided explicitly
by the nonmatching-interface discretization; this module does not choose them.
No ALE, convection, nonlinear elasticity or FSI traction recurrence is added.
"""
from fenics import *

def epsilon(u):return sym(grad(u))
def tangent(u,n):return u-dot(u,n)*n
def slip(u,v,n):return inner(tangent(u,n),tangent(v,n))

def fluid_bulk(V,dt,old,m,source,measure,weights=(3.,4.,1.),denominator=2.):
    u,p=TrialFunctions(V);v,q=TestFunctions(V)
    a0,a1,a2=weights
    return (m['rho_f']/(denominator*dt)*dot(a0*u-a1*old['u']+a2*old['u1'],v)
            +2*m['mu_f']*inner(epsilon(u),epsilon(v))-p*div(v)+div(u)*q-dot(source,v))*measure

def biot_bulk(V,dt,old,m,source_e,source_p,measure,weights=(3.,4.,1.),denominator=2.):
    eta,xi,phi=TrialFunctions(V);vp,zeta,psi=TestFunctions(V)
    a0,a1,a2=weights
    return (m['rho_b']/(denominator*dt)*dot(a0*xi-a1*old['xi']+a2*old['xi1'],zeta)
            +2*m['mu_b']*inner(epsilon(eta),epsilon(zeta))+m['lambda_b']*div(eta)*div(zeta)
            -m['alpha']*phi*div(zeta)+dot(xi,vp)-dot(a0*eta-a1*old['eta']+a2*old['eta1'],vp)/(denominator*dt)
            +m['C0']/(denominator*dt)*(a0*phi-a1*old['phi']+a2*old['phi1'])*psi
            +m['alpha']*div(xi)*psi+m['K']*dot(grad(phi),grad(psi))
            -dot(source_e,zeta)-source_p*psi)*measure

def fluid_interface(u,v,xi_ex,phi_ex,u_ex,n,L,gamma,measure):
    return (L*dot(u-u_ex,n)*dot(v,n)+gamma*slip(u-xi_ex,v,n)+phi_ex*dot(v,n))*measure

def biot_interface(xi,phi,zeta,psi,u_ex,phi_ex,n,L,gamma,measure):
    # beta=0: no normal-velocity Robin term in the mechanics equation.
    return (gamma*slip(xi-u_ex,zeta,n)+phi*dot(zeta,n)
            +(phi-phi_ex)*psi/L-dot(xi,n)*psi+dot(u_ex,n)*psi)*measure

def coupled_interface(u,xi,phi,v,zeta,psi,nf,gamma,measure):
    """Common-surface physical coupling; the Biot normal is -nf.

    The pressure/normal-flow terms are skew under energy testing. L and beta
    do not occur. Arguments may be pulled-back traces of different meshes.
    """
    return (gamma*slip(u-xi,v-zeta,nf)+phi*dot(v-zeta,nf)-dot(u-xi,nf)*psi)*measure
