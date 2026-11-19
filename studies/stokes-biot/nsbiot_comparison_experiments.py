#!/usr/bin/env python3
"""Fair BDF2-EX2 Stokes--Biot Neumann--Robin versus Robin--Robin comparison.

Works with legacy FEniCS (dolfin) and the accompanying nsbiot_compare_core.py.

The proposed scheme uses beta=0; the predecessor has beta>0. Only the two
normal mechanical interface terms differ at fixed beta and other settings.

Studies: energy, stability (empirical), temporal, spatial, parameters, cost.

IMPORTANT:
  * Zero-force studies duplicate a nontrivial t=0 pressure projection as the
    t=dt history state. This is a reproducible arbitrary-history energy test,
    NOT a physically accurate start-up method for error convergence.
  * Manufactured forcing uses exact projected states at t=0,dt; stress values
    mu_f=mu_b=lambda_b=alpha=1 are required by this example.
  * Stability labels are finite-horizon numerical criteria, not sharp bounds.
  * Timings are sequential runs, not measured parallel-wall-clock performance.
  * Default phi_degree=2 agrees with the manuscript's P2 pore pressure (user's
    attached scripts default to phi_degree=1). Explicitly set --phi-degree 1
    to reproduce that original code configuration.
"""
import argparse
import csv
import json
import math
import traceback
from pathlib import Path

from nsbiot_compare_core import solve_case, time_grid, _fenics_import_error

FIELDS = ('err_u','err_p','err_eta','err_xi','err_phi')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--study', choices=('all','energy','stability','temporal',
                                       'spatial','parameters','cost'), default='temporal')
    p.add_argument('--mesh', type=int, default=24, help='Fixed mesh intervals per side')
    p.add_argument('--meshes', type=int, nargs='+', default=[8,16,32])
    p.add_argument('--dts', type=float, nargs='+', default=[0.25,0.125,0.0625,0.03125,0.015625])
    p.add_argument('--spatial-dt', type=float, default=0.005)
    p.add_argument('--final-time', type=float, default=1.0)
    p.add_argument('--phi-degree', type=int, choices=(1,2), default=2)
    p.add_argument('--beta', type=float, default=1.0,
                   help='Predecessor mechanical normal Robin coefficient')
    p.add_argument('--L', type=float, default=1.0,
                   help='Normal Robin fluid coefficient, with Darcy coefficient 1/L')
    p.add_argument('--gamma', type=float, default=1.0,
                   help='Beavers--Joseph--Saffman coefficient')
    p.add_argument('--L-values', type=float, nargs='+', default=[0.25,1.0,4.0])
    p.add_argument('--beta-values', type=float, nargs='+', default=[0.25,1.0,4.0])
    p.add_argument('--material-values', type=float, nargs='+', default=[0.1,1.0,10.0])
    p.add_argument('--initial-mode', choices=('smooth','interface','oscillatory'),
                   default='smooth', help='Homogeneous initial pressure excitation')
    p.add_argument('--mode-number', type=int, default=0,
                   help='Oscillatory mode: 0 chooses nm//4 per mesh')
    p.add_argument('--growth-limit', type=float, default=10.0,
                   help='Empirical stability criterion: max(E)/E_at_t1 <= limit')
    p.add_argument('--accuracy-field', choices=FIELDS, default='err_u')
    p.add_argument('--target-error', type=float, default=1e-4)
    p.add_argument('--output-dir', type=Path, default=Path('nsbiot_results'))
    p.add_argument('--max-cases', type=int, default=0,
                   help='Stop after this many cases (0 means no limit)')
    p.add_argument('--dry-run', action='store_true', help='Preview without FEniCS')
    p.add_argument('--keep-going', action='store_true',
                   help='Record solver failures and continue instead of aborting')
    return p.parse_args()


def build_cases(a):
    if not all(math.isfinite(v) and v > 0 for v in (a.beta,a.L,a.gamma,a.final_time)):
        raise ValueError('Require beta,L,gamma,T > 0')
    if any(n <= 0 for n in [a.mesh,*a.meshes]):
        raise ValueError('Mesh intervals must be positive integers')
    if not a.dts or any(not math.isfinite(dt) or dt<=0 for dt in a.dts):
        raise ValueError('All target time steps must be finite and positive')
    if a.growth_limit < 1 or a.target_error <= 0:
        raise ValueError('Require growth-limit >= 1 and target-error > 0')
    if a.spatial_dt <= 0:
        raise ValueError('Require positive spatial-dt')
    if a.max_cases < 0 or a.mode_number < 0:
        raise ValueError('max-cases and mode-number must be nonnegative')
    if any(not math.isfinite(v) or v <= 0 for v in (*a.L_values,*a.beta_values,*a.material_values)):
        raise ValueError('All parameter scan values must be finite and positive')
    all_studies = ('energy','stability','temporal','spatial','parameters','cost')
    studies = all_studies if a.study=='all' else (a.study,)
    cases=[]

    def add(study, nm, target_dt, *, L=None, beta=None, forcing='manufactured',
            material=None, label='baseline'):
        dt, steps = time_grid(target_dt,a.final_time)
        if steps < 2:
            raise ValueError('Need at least two time steps')
        # Deduplicate any target values that map onto the same uniform grid.
        case={ 'study':study,'nm':nm,'dt':dt,'steps':steps,'T':a.final_time,
              'h':1.0/nm,'L':a.L if L is None else L,
              'beta':beta,'gamma':a.gamma,'forcing':forcing,
              'initialization':'projected' if forcing=='manufactured' else 'duplicate',
              'physical':material or {}, 'label':label,
              'phi_degree':a.phi_degree,
              'initial_mode':a.initial_mode if forcing=='zero' else 'smooth',
              'mode_number':a.mode_number if forcing=='zero' else 0,
              'monitor':forcing=='zero' }
        if not any(all(old[k]==case[k] for k in ('study','nm','dt','L','beta','forcing','label','physical'))
                   for old in cases):
            cases.append(case)

    def add_pair(study,nm,dt,**kwargs):
        add(study,nm,dt,beta=0.0,**kwargs)
        add(study,nm,dt,beta=a.beta,**kwargs)

    if 'temporal' in studies:
        for dt in a.dts:
            add_pair('temporal',a.mesh,dt)
    if 'spatial' in studies:
        for nm in a.meshes:
            add_pair('spatial',nm,a.spatial_dt)
    if 'energy' in studies:
        for nm in a.meshes:
            for dt in a.dts:
                add_pair('energy',nm,dt,forcing='zero')
    if 'stability' in studies:
        for nm in a.meshes:
            for dt in a.dts:
                add_pair('stability',nm,dt,forcing='zero')
    if 'parameters' in studies:
        for L in a.L_values:
            add_pair('parameters',a.mesh,a.dts[2 if len(a.dts)>2 else -1],L=L,
                     forcing='zero',label=f'L={L:g}')
        # beta scan is restricted to Robin--Robin; new scheme has beta=0 exactly.
        for beta in a.beta_values:
            add('parameters',a.mesh,a.dts[2 if len(a.dts)>2 else -1],beta=beta,
                forcing='zero',label=f'beta={beta:g}')
        for parameter in ('K','C0','rho_b','mu_b','rho_f','mu_f'):
            for value in a.material_values:
                add_pair('parameters',a.mesh,a.dts[2 if len(a.dts)>2 else -1],
                         forcing='zero',material={parameter:value},
                         label=f'{parameter}={value:g}')
    if 'cost' in studies:
        for nm in a.meshes:
            for dt in a.dts:
                add_pair('cost',nm,dt)
    return cases[:a.max_cases] if a.max_cases>0 else cases


def csv_write(path,rows):
    if not rows:
        return
    keys=[]
    for r in rows:
        keys.extend(k for k in r.keys() if k not in keys)
    with open(path,'w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k:(json.dumps(v) if isinstance(v,(dict,list)) else v)
                        for k,v in row.items()})


def summarize_thresholds(rows,a):
    summaries=[]
    groups=sorted(set((r['nm'],r['scheme']) for r in rows
                      if r['study']=='stability' and r.get('scheme') in ('neumann_robin','robin_robin')))
    for nm,scheme in groups:
        subset=[r for r in rows if r['study']=='stability' and r['nm']==nm and r['scheme']==scheme]
        good=sorted((r['dt'] for r in subset if r.get('criterion_met') is True))
        bad=sorted((r['dt'] for r in subset if r.get('criterion_met') is False))
        summaries.append({
            'nm':nm,'h':1.0/nm,'scheme':scheme,
            'growth_limit':a.growth_limit,'T':a.final_time,
            'largest_passing_dt_tested':max(good) if good else None,
            'smallest_failing_dt_tested':min(bad) if bad else None,
            'n_passing':len(good),'n_failing':len(bad),
            'note':('Finite-horizon, tested-grid criterion; not a sharp theoretical stability threshold')
        })
    return summaries


def summarize_cost(rows,a):
    out=[]
    for scheme in ('neumann_robin','robin_robin'):
        opts=[r for r in rows if r['study']=='cost' and r.get('scheme')==scheme
              and r.get('status')=='ok'
              and r.get(a.accuracy_field) is not None
              and r[a.accuracy_field] <= a.target_error]
        if not opts:
            out.append({'scheme':scheme,'target_error':a.target_error,
                        'accuracy_field':a.accuracy_field,'best_wall_s':None,
                        'note':'No tested setting met this tolerance'})
        else:
            best=min(opts,key=lambda x:x['wall_s'])
            out.append({'scheme':scheme,'target_error':a.target_error,
                        'accuracy_field':a.accuracy_field,'best_wall_s':best['wall_s'],
                        'dt':best['dt'],'nm':best['nm'],
                        'achieved_error':best[a.accuracy_field],
                        'note':'Sequential wall time; no parallel execution'})
    return out


def convergence_table(rows):
    """Observed rates at fixed mesh; blank when no positive error ratio exists."""
    out=[]
    groups=sorted({(r['scheme'],r['nm']) for r in rows
                   if r['study']=='temporal' and r.get('status')=='ok'})
    for scheme,nm in groups:
        subset=sorted((r for r in rows if r['study']=='temporal'
                       and r.get('status')=='ok' and r['scheme']==scheme
                       and r['nm']==nm),key=lambda r:-r['dt'])
        previous=None
        for r in subset:
            entry={k:r[k] for k in ('scheme','nm','dt',*FIELDS)}
            for field in FIELDS:
                rate=None
                if previous and previous[field]>0 and r[field]>0 and previous['dt']>r['dt']:
                    rate=math.log(previous[field]/r[field])/math.log(previous['dt']/r['dt'])
                entry[field+'_rate']=rate
            out.append(entry)
            previous=r
    return out


def main():
    a=parse_args()
    cases=build_cases(a)
    print(f'{len(cases)} comparison cases; study={a.study}, T={a.final_time}, '
          f'phi_degree={a.phi_degree}, L={a.L}, beta_RR={a.beta}')
    print('beta=0: proposed NR | beta>0: predecessor RR | L_1=L, L_3=1/L')
    for i,case in enumerate(cases,1):
        print(f' {i:3d} {case["study"]:10s} '
              f'nm={case["nm"]:3d} dt={case["dt"]:.6g} '
              f'beta={case["beta"]:g} L={case["L"]:g} {case["label"]}')
    if a.dry_run:
        print('DRY RUN: no numerical solutions or experimental results were produced.')
        return
    if _fenics_import_error is not None:
        raise SystemExit('Legacy FEniCS/dolfin is not installed. Use --dry-run here; '
                         'execute numerical solutions in an environment with FEniCS.')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; energy_rows=[]
    for i,case in enumerate(cases,1):
        print(f'RUN {i}/{len(cases)} {case["study"]} nm={case["nm"]} '
              f'dt={case["dt"]:.6g} beta={case["beta"]:g}',flush=True)
        metadata=dict(case)
        metadata.pop('monitor')
        try:
            r=solve_case(case['dt'],case['nm'],case['T'],case['phi_degree'],
                  beta=case['beta'],L=case['L'],gamma=case['gamma'],
                  physical=case['physical'],forcing=case['forcing'],
                  initialization=case['initialization'],monitor=case['monitor'],
                  initial_mode=case['initial_mode'],mode_number=case['mode_number'])
            energies=r.pop('energy_history')
            row={'status':'ok',**metadata,**r}
            if case['forcing']=='zero':
                # This is a *chosen finite horizon* criterion, not a theorem or
                # assertion of spectral stability.
                row['criterion_met']=bool(r['peak_energy_ratio'] is not None and
                            math.isfinite(r['peak_energy_ratio']) and
                            r['peak_energy_ratio']<=a.growth_limit)
            else:
                row['criterion_met']=None
            rows.append(row)
            for en in energies:
                energy_rows.append({'study':case['study'],'nm':case['nm'],
                   'h':case['h'],'dt':case['dt'],'scheme':r['scheme'],
                   'beta':case['beta'],'L':case['L'],'label':case['label'],**en})
            print(f'  {r["scheme"]}: elapsed={r["wall_s"]:.3f}s '
                  f'energy_ratio={r["peak_energy_ratio"]} '
                  f'err_u={r["err_u"]:.3e}',flush=True)
        except Exception as exc:
            row={'status':'failed',**metadata,
                 'scheme':'neumann_robin' if case['beta']==0 else 'robin_robin',
                 'criterion_met':False if case['forcing']=='zero' else None,
                 'failure_type':type(exc).__name__,'failure':str(exc)}
            rows.append(row)
            print(f'  FAILED: {type(exc).__name__}: {exc}',flush=True)
            if not a.keep_going:
                csv_write(a.output_dir/'runs.csv',rows)
                csv_write(a.output_dir/'energy.csv',energy_rows)
                raise
        finally:
            # Keep partial results usable if a long sweep is interrupted.
            csv_write(a.output_dir/'runs.csv',rows)
            csv_write(a.output_dir/'energy.csv',energy_rows)
    csv_write(a.output_dir/'temporal_convergence.csv',convergence_table(rows))
    summary1=summarize_thresholds(rows,a)
    summary2=summarize_cost(rows,a)
    csv_write(a.output_dir/'empirical_thresholds.csv',summary1)
    if any(r['study']=='cost' for r in rows):
        csv_write(a.output_dir/'cost_at_tolerance.csv',summary2)
    with open(a.output_dir/'protocol.json','w') as f:
        json.dump({'notes':[
            'All runs have identical grids and time integrators; beta is sole normal mechanics variation.',
            'Two zero-force history levels are duplicated from t=0, with nonzero interior pore pressure.',
            'Stability is a finite-time energy growth test, not an identified sharp CFL coefficient.',
            'Timing sequential solves does not measure true two-subdomain parallel wall clock time.',
            'For pressure_degree=2 the FE selection matches the stated manuscript example.',
            'Material sweeps use homogeneous forcing because manufactured traction is not coefficient robust.'
          ],'args':vars(a)|{'output_dir':str(a.output_dir)},'n_cases':len(cases)},f,indent=2)
    print('Saved:',a.output_dir.resolve())


if __name__=='__main__':
    main()
