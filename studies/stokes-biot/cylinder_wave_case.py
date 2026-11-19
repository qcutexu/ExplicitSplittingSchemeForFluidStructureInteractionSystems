"""Case-specific pressure pulse, cylinder boundaries and wave probes.

Embedded into stokes_biot_cylinder_wave.py; not a runtime dependency of it.
"""
def cylinder_read_geometry(kind,comm,cfg,mesh_dir=None):
    if mesh_dir is None:raise ValueError('Cylinder mesh directory is required')
    mesh=Mesh(comm)
    with XDMFFile(comm,str(Path(mesh_dir)/f'{kind}.xdmf')) as f:f.read(mesh)
    mvc=MeshValueCollection('size_t',mesh,2)
    with XDMFFile(comm,str(Path(mesh_dir)/f'{kind}_boundary.xdmf')) as f:f.read(mvc,'marker')
    markers=MeshFunction('size_t',mesh,mvc)
    return {'mesh':mesh,'markers':markers,'n':FacetNormal(mesh),'dx':Measure('dx',domain=mesh),'ds':Measure('ds',domain=mesh,subdomain_data=markers)}

def cylinder_validate(cfg):
    for spec in cfg['pore'].values():
        if spec['type']!='flux' or float(spec['value'])!=0:raise ValueError('This cylinder example uses impermeable pore boundaries')
    return cfg

def cylinder_bcs(kind,V,markers,time,cfg):
    cylinder_validate(cfg)
    if kind=='fluid':return [],None
    zero=Constant((0.,0.,0.))
    return [DirichletBC(V.sub(i),zero,markers,tag) for i in [0,1] for tag in [3,6]],None

def pulse_pressure(t,amplitude,duration):
    return float(amplitude*np.sin(np.pi*t/duration)**2) if 0<t<duration else 0.

class CylinderBlock(Block3D):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pressure_load=Constant(0.)
        if self.kind=='fluid':
            v,q=TestFunctions(self.V)
            # sigma_f n = -P_in(t)n on inlet label 1: negative RHS traction.
            self.load+=-self.pressure_load*dot(v,self.g['n'])*self.g['ds'](1)
    def rhs(self,t):
        if self.kind=='fluid':
            pulse=self.cfg['pressure_pulse'];self.pressure_load.assign(pulse_pressure(t,pulse['amplitude'],pulse['duration']))
        return assemble(self.load)

class WaveProbes:
    def __init__(self,cfg,output,world):
        self.cfg=cfg;self.output=output;self.world=world;self.rows=[]
        self.stations=np.asarray(cfg['cylinder']['length'])*np.array([.1,.3,.5,.7,.9])
    def __call__(self,blocks,step,t):
        values=np.zeros((len(self.stations),3));counts=np.zeros_like(values)
        shape=self.cfg['cylinder'];theta=np.pi/shape['angular'];normal=np.array([np.cos(theta),np.sin(theta),0.])
        radius=(shape['radius']+shape['thickness']/2)*np.cos(theta)
        for b in blocks:
            fields=dict(zip(b.names,b.new.split(True)));tree=b.mesh.bounding_box_tree()
            for i,z in enumerate(self.stations):
                point=Point(0.,0.,z) if b.kind=='fluid' else Point(radius*normal[0],radius*normal[1],z)
                if tree.compute_first_entity_collision(point)>=b.mesh.num_cells():continue
                if b.kind=='fluid':
                    values[i,0]=float(fields['p'](point));values[i,1]=float(fields['u'](point)[2]);counts[i,:2]=1
                else:values[i,2]=float(np.dot(fields['eta'](point),normal));counts[i,2]=1
        totals=np.empty_like(values);number=np.empty_like(counts)
        self.world.Allreduce(values,totals,op=mpi.SUM);self.world.Allreduce(counts,number,op=mpi.SUM)
        if np.any(number==0):raise ValueError('A wave probe is outside the distributed cylinder mesh')
        totals/=number
        if self.world.rank==0:
            for i,z in enumerate(self.stations):self.rows.append({'step':step,'t':t,'z_cm':z,'pressure_dyn_cm2':totals[i,0],'axial_velocity_cm_s':totals[i,1],'wall_radial_displacement_cm':totals[i,2],'inlet_pressure_dyn_cm2':pulse_pressure(t,self.cfg['pressure_pulse']['amplitude'],self.cfg['pressure_pulse']['duration'])})
            write_csv(self.output/'wave_probes.csv',self.rows)

def cylinder_configuration(radius,thickness,length,angular,fluid_radial,wall_radial,axial,amplitude,duration):
    cfg=copy.deepcopy(DEFAULT_CONFIGURATION)
    cfg['mesh_coordinate_scale']=1.;cfg['cylinder']={'radius':radius,'thickness':thickness,'length':length,'angular':angular,'fluid_radial':fluid_radial,'wall_radial':wall_radial,'axial':axial}
    cfg['pressure_pulse']={'type':'finite sine-squared normal traction','amplitude':amplitude,'duration':duration,'unit':'dyn/cm^2','time_unit':'s'}
    cfg['fluid']['1']={'type':'pressure_traction','formula':'sigma_f n = -P_in(t) n'}
    cfg['fluid']['2']={'type':'zero_outlet_total_traction'}
    cfg['biot_mechanics']={'2':{'type':'zero_outer_total_traction'},'3':{'type':'clamped_outlet_ring'},'6':{'type':'clamped_inlet_ring'},'5':{'type':'coupled_interface'}}
    cfg['pore']={str(tag):{'type':'flux','value':0.} for tag in [2,3,6]}
    cfg.pop('inlet_profile',None)
    cfg['initial_fields_and_volume_sources']={'initial_fields':'zero at t=0','fluid_volume_force':[0.,0.,0.],'biot_volume_force':[0.,0.,0.],'pore_volume_source':0.}
    return cfg

# All calls by the bundled bulk constructor use these cylinder-specific hooks.
read_configured_geometry=cylinder_read_geometry
exterior_bcs=cylinder_bcs
validate_boundary_config=cylinder_validate
Block3D=CylinderBlock

def cylinder_main():
    p=argparse.ArgumentParser(description='3D pressure pulse in a fluid cylinder with a finite-thickness Biot wall (CGS).')
    p.add_argument('--method',choices=['nr','monolithic'],default='monolithic')
    p.add_argument('--radius',type=float,default=.30);p.add_argument('--thickness',type=float,default=.06);p.add_argument('--length',type=float,default=6.)
    p.add_argument('--angular',type=int,default=24,help='Circumferential target; Gmsh does not impose rings');p.add_argument('--fluid-radial',type=int,default=3,help='Fluid radial target: radius/N');p.add_argument('--wall-radial',type=int,default=3,help='Wall target: thickness/N, not forced unstructured layers');p.add_argument('--axial',type=int,default=24,help='Axial target: length/N; no extruded planes in Gmsh mode')
    p.add_argument('--mesh-generator',choices=['gmsh','layered'],default='gmsh');p.add_argument('--gmsh-size-factor',type=float,default=1.0)
    p.add_argument('--pressure-amplitude',type=float,default=1000.);p.add_argument('--pulse-duration',type=float,default=.002)
    p.add_argument('--dt',type=float,default=None,help='Monolithic default 1e-4 s; NR requires an explicit tested choice');p.add_argument('--T',type=float,default=None,help='Monolithic default 0.05 s; specify explicitly for NR')
    p.add_argument('--xdmf-every',type=int,default=1);p.add_argument('--checkpoint-every',type=int,default=1)
    p.add_argument('--output-dir',type=Path,default=None);p.add_argument('--mesh-only',action='store_true')
    args=p.parse_args();set_log_level(LogLevel.WARNING);world=mpi.COMM_WORLD
    if args.output_dir is None:args.output_dir=Path('results')/('cylinder_wave_'+args.method)
    if args.method=='nr' and (args.dt is None or args.T is None):
        p.error('NR needs explicit --dt and --T: dt=1e-4 was unstable; dt=2.5e-5 was bounded only through T=0.006 in the tested mesh')
    if args.dt is None:args.dt=.0001
    if args.T is None:args.T=.05
    try:
        if args.pressure_amplitude<=0 or args.pulse_duration<=0:raise ValueError('Positive pulse amplitude/duration required')
        cfg=cylinder_configuration(args.radius,args.thickness,args.length,args.angular,args.fluid_radial,args.wall_radial,args.axial,args.pressure_amplitude,args.pulse_duration)
        cfg['mesh_generator']=args.mesh_generator;cfg['gmsh_size_factor']=args.gmsh_size_factor
        meshdir=args.output_dir/'mesh'
        if world.rank==0:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            audit=write_cylinder_meshes(meshdir,cfg['cylinder'],args.mesh_generator,args.gmsh_size_factor)
            (args.output_dir/'mesh_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
            (args.output_dir/'cylinder_configuration.json').write_text(json.dumps(cfg,indent=2)+'\n')
            print('Cylinder mesh:',json.dumps(audit),flush=True)
        world.Barrier()
        if args.mesh_only:return
        observer=WaveProbes(cfg,args.output_dir,world)
        run(cfg,args.method,args.output_dir,args.dt,args.T,mesh_source='cylinder',phi_degree=2,mesh_dir=meshdir,save_fields=False,xdmf_every=args.xdmf_every,checkpoint_every=args.checkpoint_every,step_observer=observer)
    except Exception:
        traceback.print_exc()
        if world.size>1:world.Abort(1)
        raise
