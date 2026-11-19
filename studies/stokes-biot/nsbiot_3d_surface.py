"""Common-fluid-surface projected quadrature for nonmatching P2 traces.

This is a fixed geometric closest-point pullback, not nearest-node transfer.
All interface bilinear forms use the same points, weights and fluid normal.
Normal pressure/flow coupling is algebraically skew; slip is symmetric PSD.
No multiplier unknowns are added. The native tetrahedral meshes are retained.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

def key(point):return tuple(np.round(point,11))

def closest_on_triangles(points,triangles):
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    e0=b-a;e1=c-a;v=points-a
    d00=np.einsum('ij,ij->i',e0,e0);d01=np.einsum('ij,ij->i',e0,e1);d11=np.einsum('ij,ij->i',e1,e1)
    d20=np.einsum('ij,ij->i',v,e0);d21=np.einsum('ij,ij->i',v,e1);den=d00*d11-d01*d01
    if np.any(den<=0):raise ValueError('Degenerate interface triangle')
    l1=(d11*d20-d01*d21)/den;l2=(d00*d21-d01*d20)/den
    bary=np.column_stack((1-l1-l2,l1,l2));inside=np.min(bary,axis=1)>=0
    projected=np.einsum('ni,nij->nj',bary,triangles)
    best=np.where(inside,np.sum((projected-points)**2,axis=1),np.inf)
    for i,j in [(0,1),(1,2),(2,0)]:
        edge=triangles[:,j]-triangles[:,i]
        t=np.clip(np.einsum('ij,ij->i',points-triangles[:,i],edge)/np.sum(edge*edge,axis=1),0,1)
        q=triangles[:,i]+t[:,None]*edge;dist=np.sum((q-points)**2,axis=1);take=dist<best
        candidate=np.zeros_like(bary);candidate[:,i]=1-t;candidate[:,j]=t
        bary[take]=candidate[take];projected[take]=q[take];best[take]=dist[take]
    return projected,bary,np.sqrt(best)

class SurfaceProjection:
    def __init__(self,triangles):
        self.triangles=np.asarray(triangles);self.centers=self.triangles.mean(axis=1)
        self.radius=float(np.linalg.norm(self.triangles-self.centers[:,None,:],axis=2).max())
        self.tree=cKDTree(self.centers)
    def query(self,points):
        points=np.atleast_2d(points);_,seed=self.tree.query(points)
        _,_,upper=closest_on_triangles(points,self.triangles[seed])
        # Every possible closer triangle lies in this centroid ball: use its
        # global bounding radius, not a fixed number of nearest centroids.
        candidates=self.tree.query_ball_point(points,upper+self.radius+1e-12)
        owners=np.repeat(np.arange(len(points)),[len(a) for a in candidates]);ids=np.concatenate(candidates)
        q,bary,dist=closest_on_triangles(points[owners],self.triangles[ids])
        order=np.lexsort((ids,dist,owners));ordered=owners[order]
        selected=order[np.r_[True,ordered[1:]!=ordered[:-1]]]
        return ids[selected],q[selected],bary[selected],dist[selected]

def triangle_nodes(triangles,degree=2):
    if degree==1:return triangles
    if degree!=2:raise ValueError('Surface degree must be 1 or 2')
    return np.concatenate((triangles,(triangles[:,0:1]+triangles[:,1:2])/2,
                           (triangles[:,1:2]+triangles[:,2:3])/2,(triangles[:,2:3]+triangles[:,0:1])/2),axis=1)

def surface_space(triangles,degree=2):
    nodes=triangle_nodes(triangles,degree)
    coords=np.asarray(sorted({key(p) for p in nodes.reshape(-1,3)}))
    lookup={key(x):i for i,x in enumerate(coords)}
    ids=np.asarray([[lookup[key(p)] for p in row] for row in nodes],dtype=np.int64)
    return coords,ids

def basis(lam,degree=2):
    if degree==1:return lam
    a,b,c=lam.T
    return np.column_stack((a*(2*a-1),b*(2*b-1),c*(2*c-1),4*a*b,4*b*c,4*c*a))

def triangle_rule(order):
    # Tensor Gauss on a Duffy map. order=4 is exact for total degree <=6 on
    # one uncut triangle. Nonmatching pullback crossings require refinement.
    if order<2:raise ValueError('Quadrature order must be >=2')
    x,w=np.polynomial.legendre.leggauss(order);x=(x+1)/2;w=w/2
    lam=[];weights=[]
    for i,u in enumerate(x):
        for j,v in enumerate(x):lam.append((1-u-(1-u)*v,u,(1-u)*v));weights.append(w[i]*w[j]*(1-u))
    return np.asarray(lam),np.asarray(weights)

class InterfaceOperators:
    def __init__(self,fluid_triangles,biot_triangles,normals,material,L,phi_degree=2,order=4,refinement=3,biot_normals=None):
        self.fc,self.fi=surface_space(fluid_triangles,2);self.bc,self.bi=surface_space(biot_triangles,2)
        self.pc,self.pi=surface_space(biot_triangles,phi_degree)
        self.nf=3*len(self.fc);self.nx=3*len(self.bc);self.np=len(self.pc);self.N=self.nf+self.nx+self.np
        self.maps={'u':np.arange(self.nf).reshape(-1,3),'xi':self.nf+np.arange(self.nx).reshape(-1,3),'phi':(self.nf+self.nx+np.arange(self.np))[:,None]}
        self.coordinates={'u':self.fc,'xi':self.bc,'phi':self.pc}
        projection=SurfaceProjection(biot_triangles);rule,rule_weights=triangle_rule(order)
        if refinement<0:raise ValueError("Projection refinement must be nonnegative")
        unresolved_area=0.;patch_count=0
        def patches(tri,vertices,depth=0):
            lam=rule@vertices;ids,q,bl,dist=projection.query(lam@tri)
            feature=np.sum((bl<1e-10)*np.array([1,2,4]),axis=1)
            consistent=np.all(ids==ids[0]) and np.all(feature==feature[0])
            if consistent or depth==refinement:
                yield lam,rule_weights/(4**depth),ids,bl,dist,consistent
            else:
                a,b,c=vertices;ab=(a+b)/2;bc=(b+c)/2;ca=(c+a)/2
                for sub in [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]:yield from patches(tri,np.asarray(sub),depth+1)
        dr=[];dc=[];dv=[];rr=[];rc=[];rv=[];distances=[];alignments=[];area=0.
        gamma=material['gamma']
        def put(target,rows,cols,values):
            rows=np.asarray(rows).reshape(-1);cols=np.asarray(cols).reshape(-1)
            a,b,c=(dr,dc,dv) if target=='D' else (rr,rc,rv)
            a.append(np.repeat(rows,len(cols)));b.append(np.tile(cols,len(rows)));c.append(np.asarray(values).reshape(-1))
        # Chunk sums limit Python triplet overhead; only trace matrices live on
        # the assembly leader. They are scattered by row before volume solves.
        D=sp.csr_matrix((self.N,self.N));R=D.copy()
        for ti,tri in enumerate(fluid_triangles):
            jac=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]));area+=jac/2
            n=normals[ti];P=np.eye(3)-np.outer(n,n);NN=np.outer(n,n)
            fids=self.maps['u'][self.fi[ti]].reshape(-1)
            pieces=list(patches(tri,np.eye(3)));patch_count+=len(pieces)
            for lam,weights,ids,bl,dist,consistent in pieces:
                distances.extend(dist)
                if biot_normals is not None:alignments.extend(-np.asarray(biot_normals)[ids]@n)
                if not consistent:unresolved_area+=jac*sum(weights)
                bf=basis(lam)
                for qi,bt in enumerate(ids):
                    f=bf[qi];b=basis(bl[qi:qi+1])[0];p=basis(bl[qi:qi+1],phi_degree)[0];w=weights[qi]*jac
                    xids=self.maps['xi'][self.bi[bt]].reshape(-1);pids=self.maps['phi'][self.pi[bt]].reshape(-1)
                    ff=np.outer(f,f)*w;bb=np.outer(b,b)*w;fb=np.outer(f,b)*w
                    Nf=np.outer(f,p)*w;Nx=np.outer(b,p)*w;mass=np.outer(p,p)*w
                    F=np.kron(ff,L*NN);Sff=np.kron(ff,gamma*P);Sxx=np.kron(bb,gamma*P);Sfx=np.kron(fb,gamma*P)
                    # Mixed normal entries, ordered [node, component].
                    Fp=(Nf[:,:,None]*n).transpose(0,2,1).reshape(len(fids),len(pids))
                    Xp=(Nx[:,:,None]*n).transpose(0,2,1).reshape(len(xids),len(pids))
                    put('D',fids,fids,F+Sff);put('D',xids,xids,Sxx)
                    put('D',xids,pids,-Xp);put('D',pids,xids,Xp.T);put('D',pids,pids,mass/L)
                    put('R',fids,fids,F);put('R',fids,xids,Sfx);put('R',xids,fids,Sfx.T)
                    put('R',fids,pids,-Fp);put('R',pids,fids,Fp.T);put('R',pids,pids,mass/L)
            if (ti+1)%64==0 or ti+1==len(fluid_triangles):
                for target,a,b,c in [('D',dr,dc,dv),('R',rr,rc,rv)]:
                    block=sp.coo_matrix((np.concatenate(c),(np.concatenate(a),np.concatenate(b))),shape=D.shape).tocsr()
                    if target=='D':D+=block
                    else:R+=block
                    a.clear();b.clear();c.clear()
        self.D=D;self.R=R;self.H=(D-R).tocsr();self.H.eliminate_zeros()
        distances=np.asarray(distances)
        if alignments and min(alignments)<=0:raise ValueError('Closest-point interface has reversed normal alignment; inspect the geometry pairing')
        self.audit={'normal_alignment_min':float(min(alignments)) if alignments else None,'fluid_surface_area':area,'fluid_triangles':len(fluid_triangles),'biot_triangles':len(biot_triangles),
                    'quadrature_order':order,'projection_refinement':refinement,'integration_patches':patch_count,'unresolved_crossing_area_fraction':unresolved_area/area,'quadrature_points':len(distances),'projection_distance_max':float(distances.max()),
                    'projection_distance_mean':float(distances.mean()),'trace_dofs':self.N,'D_nonzeros':D.nnz,'R_nonzeros':R.nnz,
                    'coupling':'common fluid surface; exact closest triangle point; P2/P1 or P2/P2 pulled-back traces; single paired quadrature'}
