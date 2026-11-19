"""Conforming faceted cylinder + finite-thickness annulus, extruded to tetrahedra.

No external mesher required. The polygonal circle converges under angular
refinement. Wall thickness and independent radial/axial resolution are explicit.
Only NumPy is needed to construct arrays; FEniCS writes the distributed inputs.
"""
import numpy as np

def cylinder_arrays(radius,thickness,length,angular,fluid_radial,wall_radial,axial):
    if not all(np.isfinite(x) and x>0 for x in [radius,thickness,length]):raise ValueError('Positive finite dimensions required')
    if angular<6 or min(fluid_radial,wall_radial,axial)<1:raise ValueError('Need >=6 angular divisions and positive radial/axial counts')
    def cross_section(kind):
        coords=[];tri=[]
        if kind=='fluid':
            coords.append([0.,0.]);radii=np.linspace(radius/fluid_radial,radius,fluid_radial);offset=1
        else:radii=np.linspace(radius,radius+thickness,wall_radial+1);offset=0
        for r in radii:
            for j in range(angular):
                theta=2*np.pi*j/angular;coords.append([r*np.cos(theta),r*np.sin(theta)])
        if kind=='fluid':
            for j in range(angular):tri.append([0,1+j,1+(j+1)%angular])
        for layer in range(len(radii)-1):
            for j in range(angular):
                k=(j+1)%angular;a=offset+layer*angular+j;b=offset+layer*angular+k;c=a+angular;d=b+angular
                tri.extend([[a,b,d],[a,d,c]])
        return np.asarray(coords),np.asarray(tri,dtype=np.int64)
    result={}
    for kind in ['fluid','biot']:
        xy,tri=cross_section(kind);n=len(xy)
        xyz=np.concatenate([np.column_stack((xy,np.full(n,z))) for z in np.linspace(0,length,axial+1)])
        cells=[]
        for level in range(axial):
            for row in tri:
                a,b,c=np.sort(row)+level*n;A,B,C=np.array([a,b,c])+n
                cells.extend([[a,b,c,C],[a,b,B,C],[a,A,B,C]])
        cells=np.asarray(cells,dtype=np.int64)
        det=np.linalg.det(xyz[cells[:,1:]]-xyz[cells[:,:1]])
        if np.any(abs(det)<1e-18):raise ValueError('Degenerate tetrahedron')
        negative=det<0;cells[negative,0],cells[negative,1]=cells[negative,1].copy(),cells[negative,0].copy()
        faces=np.sort(np.concatenate([cells[:,k] for k in [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]]),axis=1)
        boundary,count=np.unique(faces,axis=0,return_counts=True);boundary=boundary[count==1]
        points=xyz[boundary];z=points[:,:,2];r=np.linalg.norm(points[:,:,:2],axis=2)
        labels=np.full(len(boundary),2,dtype=np.uint64)
        if kind=='fluid':labels[:]=5
        labels[np.all(abs(z)<1e-12,axis=1)]=1 if kind=='fluid' else 6
        labels[np.all(abs(z-length)<1e-12,axis=1)]=2 if kind=='fluid' else 3
        if kind=='biot':labels[np.all(abs(r-radius)<1e-12,axis=1)]=5
        result[kind]={'coordinates':xyz,'tetrahedra':cells,'boundary_faces':boundary,'boundary_labels':labels}
    # The separately numbered subdomains must share the exact same interface.
    def interface(data):return {tuple(sorted(tuple(np.round(p,12)) for p in tri)) for tri in data['coordinates'][data['boundary_faces'][data['boundary_labels']==5]]}
    if interface(result['fluid'])!=interface(result['biot']):raise ValueError('Nonconforming cylinder interface')
    return result

def write_cylinder_meshes(folder,settings,generator="layered",size_factor=1.0):
    from pathlib import Path
    from fenics import Mesh,MeshEditor,MeshFunction,XDMFFile,facets
    from mpi4py import MPI
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if generator not in ("layered","gmsh"):raise ValueError("Unknown mesh generator")
    data=cylinder_arrays(**settings) if generator=="layered" else gmsh_cylinder_arrays(**settings,size_factor=size_factor)
    audit={}
    for kind,item in data.items():
        mesh=Mesh(MPI.COMM_SELF);editor=MeshEditor();editor.open(mesh,'tetrahedron',3,3)
        editor.init_vertices(len(item['coordinates']));editor.init_cells(len(item['tetrahedra']))
        for i,x in enumerate(item['coordinates']):editor.add_vertex(i,x)
        for i,c in enumerate(item['tetrahedra']):editor.add_cell(i,c.astype(np.uintp))
        editor.close();mesh.init(2,0)
        lookup={tuple(row):int(tag) for row,tag in zip(item['boundary_faces'],item['boundary_labels'])}
        tags=MeshFunction('size_t',mesh,2,0)
        for facet in facets(mesh):tags[facet]=lookup.get(tuple(sorted(int(i) for i in facet.entities(0))),0)
        tags.rename('marker','marker')
        with XDMFFile(MPI.COMM_SELF,str(folder/f'{kind}.xdmf')) as f:f.write(mesh)
        with XDMFFile(MPI.COMM_SELF,str(folder/f'{kind}_boundary.xdmf')) as f:f.write(tags)
        audit[kind]={'vertices':mesh.num_vertices(),'tetrahedra':mesh.num_cells(),'boundary_counts':{str(int(k)):int(np.sum(item['boundary_labels']==k)) for k in np.unique(item['boundary_labels'])},'generator':generator,**item.get('generator_audit',{})}
    return audit

def gmsh_cylinder_arrays(radius,thickness,length,angular,fluid_radial,wall_radial,axial,size_factor=1.0):
    """One CAD fragmentation/3D mesh, then material extraction.

    Unstructured tetrahedra are generated in a sizing coordinate system and
    mapped radially/axially to physical cm. This is not a layered extrusion.
    Shared interface node IDs/faces are retained through material extraction.
    """
    import gmsh
    if min(radius,thickness,length,size_factor)<=0 or not np.isfinite([radius,thickness,length,size_factor]).all():raise ValueError('Positive finite dimensions and size factor required')
    if angular<6 or min(fluid_radial,wall_radial,axial)<1:raise ValueError('Invalid target resolution')
    hr=radius/fluid_radial;hw=thickness/wall_radial;hz=length/axial;ht=2*np.pi*radius/angular
    h0=min(hr,ht,2*hw);computational_thickness=wall_radial*h0;computational_length=axial*h0
    owned=not gmsh.isInitialized()
    if owned:gmsh.initialize()
    model='stokes_biot_conforming_cylinder'
    try:
        gmsh.option.setNumber('General.NumThreads',1)
        gmsh.model.add(model)
        inner=gmsh.model.occ.addCylinder(0,0,0,0,0,computational_length,radius)
        outer=gmsh.model.occ.addCylinder(0,0,0,0,0,computational_length,radius+computational_thickness)
        fragments,mapping=gmsh.model.occ.fragment([(3,outer)],[(3,inner)])
        gmsh.model.occ.synchronize()
        fluid={tag for dim,tag in mapping[1] if dim==3};allvol={tag for dim,tag in mapping[0] if dim==3};wall=allvol-fluid
        if len(fluid)!=1 or len(wall)!=1:raise ValueError('Expected one fluid and one annular CAD volume')
        volumes={'fluid':next(iter(fluid)),'biot':next(iter(wall))}
        surfaces={kind:{abs(tag) for dim,tag in gmsh.model.getBoundary([(3,v)],oriented=False) if dim==2} for kind,v in volumes.items()}
        shared=surfaces['fluid']&surfaces['biot']
        if not shared:raise ValueError('CAD volumes have no shared interface; refusing separate nonconforming meshing')
        labels={}
        for kind,tags in surfaces.items():
            labels[kind]={}
            for tag in tags:
                box=gmsh.model.getBoundingBox(2,tag)
                if tag in shared:label=5
                elif abs(box[2])<1e-6 and abs(box[5])<1e-6:label=1 if kind=='fluid' else 6
                elif abs(box[2]-computational_length)<1e-6 and abs(box[5]-computational_length)<1e-6:label=2 if kind=='fluid' else 3
                elif kind=='biot':label=2
                else:raise ValueError('Unclassified fluid CAD surface')
                labels[kind][tag]=label
        for kind,volume in volumes.items():gmsh.model.addPhysicalGroup(3,[volume],1 if kind=='fluid' else 2)
        grouped={}
        for tagmap in labels.values():
            for tag,label in tagmap.items():grouped.setdefault(label,set()).add(tag)
        for label,tags in grouped.items():gmsh.model.addPhysicalGroup(2,sorted(tags),label)
        gmsh.option.setNumber('Mesh.MeshSizeMin',h0*size_factor)
        gmsh.option.setNumber('Mesh.MeshSizeMax',h0*size_factor)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature',0)
        gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary',0)
        gmsh.option.setNumber('Mesh.ElementOrder',1)
        gmsh.option.setNumber('Mesh.Algorithm3D',1)
        gmsh.option.setNumber('Mesh.RandomSeed',2026)
        gmsh.model.mesh.generate(3)
        gmsh.model.mesh.optimize('')
        node_tags,flat,_=gmsh.model.mesh.getNodes();order=np.argsort(node_tags);node_tags=np.asarray(node_tags)[order];computational=np.asarray(flat).reshape(-1,3)[order]
        physical=computational.copy();rho=np.linalg.norm(computational[:,:2],axis=1);outside=rho>radius
        mapped_radius=radius+(rho[outside]-radius)*thickness/computational_thickness
        physical[outside,:2]*=(mapped_radius/rho[outside])[:,None]
        physical[:,2]*=length/computational_length
        def elements(dim,tag,expected_type):
            types,ids,nodes=gmsh.model.mesh.getElements(dim,tag)
            rows=[]
            for typ,values in zip(types,nodes):
                if typ!=expected_type:raise ValueError('Expected first-order tetrahedra/triangles from Gmsh')
                n=4 if dim==3 else 3;values=np.asarray(values).reshape(-1,n)
                loc=np.searchsorted(node_tags,values)
                if not np.array_equal(node_tags[loc],values):raise ValueError('Missing Gmsh node')
                rows.append(loc)
            return np.concatenate(rows)
        result={}
        for kind,volume in volumes.items():
            global_tets=elements(3,volume,4);boundary=[];tagvalues=[]
            for surface,label in labels[kind].items():
                tri=elements(2,surface,2);boundary.append(tri);tagvalues.extend([label]*len(tri))
            global_faces=np.concatenate(boundary);used=np.unique(global_tets)
            tets=np.searchsorted(used,global_tets);faces=np.searchsorted(used,global_faces)
            xyz=physical[used];before=computational[global_tets];after=physical[global_tets]
            det0=np.linalg.det(before[:,1:]-before[:,:1]);det1=np.linalg.det(after[:,1:]-after[:,:1])
            if np.any(det0*det1<=0):raise ValueError('Sizing-to-physical map inverted or degenerated a tetrahedron; reduce --gmsh-size-factor')
            negative=det1<0;tets[negative,0],tets[negative,1]=tets[negative,1].copy(),tets[negative,0].copy()
            lengths=np.concatenate([np.linalg.norm(xyz[tets[:,i]]-xyz[tets[:,j]],axis=1) for i,j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]])
            result[kind]={'coordinates':xyz,'tetrahedra':tets,'boundary_faces':np.sort(faces,axis=1),'boundary_labels':np.asarray(tagvalues,dtype=np.uint64),
                         'generator_audit':{'generator':'Gmsh 3D Delaunay, one fragmented CAD model','gmsh_version':gmsh.__version__,'conforming_CAD_surface_tags':sorted(shared),'nominal_target_cm':{'fluid_radial':hr,'wall_radial':hw,'circumferential':ht,'axial':hz},'computational_mesh_size':h0*size_factor,'radial_wall_map_scale':thickness/computational_thickness,'axial_map_scale':length/computational_length,'minimum_tetrahedron_volume_cm3':float(abs(det1).min()/6),'total_volume_cm3':float(abs(det1).sum()/6),'minimum_map_determinant_ratio':float((det1/det0).min()),'edge_length_cm_quantiles':np.quantile(lengths,[0,.5,.95,1]).tolist()}}
        def boundary_keys(item):return {tuple(sorted(tuple(np.round(x,12)) for x in points)) for points in item['coordinates'][item['boundary_faces'][item['boundary_labels']==5]]}
        a=boundary_keys(result['fluid']);b=boundary_keys(result['biot'])
        if a!=b:raise ValueError('Extracted fluid/Biot interface triangles differ')
        for item in result.values():item['generator_audit']['interface_triangles_identical']=True;item['generator_audit']['interface_triangle_count']=len(a)
        return result
    finally:
        if owned:gmsh.finalize()
        else:gmsh.model.remove()
