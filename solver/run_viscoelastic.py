"""Bounded screening run with accepted-state gas-filament measurements."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from viscoelastic_jam import Viscoelastic


def neck_measurement(z, radius):
    neck = int(np.argmin(radius))
    minimum = float(radius[neck])
    cutoff = 1.2*minimum
    # Follow the connected component of the neck in the periodic axial grid.
    edges = []
    counts = []
    length = z[-1]-z[0]
    for side in [-1,1]:
        inner = neck
        for count in range(1,len(z)):
            outer = (neck+side*count)%len(z)
            if radius[outer]>cutoff:
                if abs(z[outer]-z[inner])>length/2:
                    return minimum,0.,0,False
                fraction = (cutoff-radius[inner])/(radius[outer]-radius[inner])
                edges.append(z[inner]+fraction*(z[outer]-z[inner]))
                counts.append(count)
                break
            inner = outer
    if len(edges)<2:
        return minimum,0.,0,False
    width = float(edges[1]-edges[0])
    intervals = sum(counts)-2
    return minimum,width,intervals,minimum<=.05 and width/(2*minimum)>=5 and intervals>=12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ns',type=int,default=97)
    parser.add_argument('--ny',type=int,default=13)
    parser.add_argument('--max-steps',type=int,default=0)
    parser.add_argument('--wall',type=float,default=28000.)
    parser.add_argument('--deadline',type=float,default=float('inf'))
    parser.add_argument('--horizon-s',type=float,default=12.)
    args = parser.parse_args()
    started = time.time()
    config = json.loads(Path(__file__).with_name('cases.json').read_text())
    params = dict(config['common'],**next(row for row in config['cases'] if row['id']==args.case))
    out = args.output
    out.mkdir(exist_ok=False,parents=True)
    solver = Viscoelastic(args.ns,args.ny,params)
    check = solver.check_coupling()
    params.update(ns=args.ns,ny=args.ny,time_unit_s=solver.time_unit,De=1/solver.inv_de,
        Ec=solver.modulus,Oh_solvent=solver.Oh,mu_s=params['mu']*(1-params['polymer_fraction']),
        mu_p=params['mu']*params['polymer_fraction'],coupling_check=check)
    (out/'config.json').write_text(json.dumps(params,indent=2))
    print('CHECK',json.dumps(check),flush=True)
    state = older = solver.initial_state()
    clock = 0.
    previous_dt = dt = .002
    count = rejects = 0
    initial_volume = np.dot(solver.zweights,solver.fields(state)[3].reshape(args.ny,args.ns)[-1]**2)
    next_save = 0.
    candidate_start = None
    candidate_count = 0
    maximum_candidate_duration = 0.
    detected = False
    last = None
    status = 'RUNNING'
    reason = ''
    end = args.horizon_s/solver.time_unit

    def save(name,q,t):
        fields = solver.fields(q).reshape(12,args.ny,args.ns)
        np.savez_compressed(out/name,hydro=fields[:8],conformation=fields[8:],
            component_order=['zz','rr','theta_theta','zr'],z=solver.z,y=solver.y,t=t,t_s=t*solver.time_unit)

    save('initial.npz',state,clock)
    with (out/'diagnostics.csv').open('w',newline='') as stream:
        writer = None
        try:
            while clock<end:
                if time.time()>min(started+args.wall,args.deadline-120):
                    status = 'WALL_LIMIT'
                    break
                dt = min(dt,end-clock)
                for attempt in range(16):
                    if dt<1e-6:
                        raise RuntimeError('adaptive timestep floor')
                    try:
                        full,_,_ = solver.advance(state,older,dt,previous_dt,count==0)
                        middle,_,_ = solver.advance(state,older,dt/2,previous_dt,count==0)
                        fine,_,_ = solver.advance(middle,state,dt/2,dt/2)
                        a,b = solver.fields(full),solver.fields(fine)
                        errors = [np.max(abs(a[3]/b[3]-1))/.0005]
                        for indices,tol,floor in [([0,1,4,5],.005,1e-6),([2,6],.005,1e-6),([8,9,10,11],.001,1.)]:
                            errors.append(np.linalg.norm(a[indices]-b[indices])/
                                (tol*max(np.linalg.norm(b[indices]),floor*np.sqrt(b[indices].size))))
                        error = float(max(errors))
                        if error<=1:
                            break
                        reason = 'step-doubling error '+str(error)
                    except RuntimeError as exc:
                        reason = str(exc)
                    rejects += 1
                    dt *= .5
                    print('REJECT',clock,dt,reason,flush=True)
                else:
                    raise RuntimeError('adaptive rejection limit')

                hydro = np.r_[b[:8].ravel(),0.,0.]
                diagnostic = solver.diagnostics(hydro)
                radius = b[3].reshape(args.ny,args.ns)[-1]
                volume_error = float(np.dot(solver.zweights,radius**2)/initial_volume-1)
                if abs(volume_error)>=.001 or diagnostic['max_actual_divergence']>=.001 or abs(fine[-1])>=1e-7:
                    raise RuntimeError('conservation or divergence gate '+str((volume_error,diagnostic,fine[-1])))
                if diagnostic['max_wall_speed']>=1e-8:
                    raise RuntimeError('wall velocity gate')
                old_time = clock
                clock += dt
                state,older,previous_dt = fine,middle,dt/2
                count += 1
                minimum,width,intervals,shape_pass = neck_measurement(solver.z,radius)
                physical_time = clock*solver.time_unit
                if shape_pass:
                    candidate_count += 1
                    if candidate_start is None:
                        candidate_start = physical_time
                    duration = physical_time-candidate_start
                    maximum_candidate_duration = max(maximum_candidate_duration,duration)
                    detected = detected or (candidate_count>=3 and duration>=20e-6)
                else:
                    candidate_start = None
                    candidate_count = 0
                    duration = 0.
                zz,rr,tt,zr = b[8:]
                last = dict(step=count,t_star=clock,t_s=physical_time,a_min_R=minimum,
                    a_min_um=1e6*params['R_m']*minimum,dt_s=previous_dt*solver.time_unit,
                    flat_length_um=1e6*params['R_m']*width,length_over_diameter=width/(2*minimum),
                    flat_intervals=intervals,shape_pass=int(shape_pass),candidate_duration_s=duration,
                    filament_candidate=int(detected),volume_error=volume_error,
                    divergence=diagnostic['max_actual_divergence'],wall_speed=diagnostic['max_wall_speed'],
                    max_trace_A=float(np.max(zz+rr+tt)),min_det_A=float(np.min((zz*rr-zr*zr)*tt)),
                    wall_s=time.time()-started)
                if writer is None:
                    writer = csv.DictWriter(stream,fieldnames=list(last))
                    writer.writeheader()
                writer.writerow(last)
                stream.flush()
                if physical_time>=next_save or minimum<=.2:
                    if minimum<=.2:
                        save('frame_%06d_mid.npz'%count,middle,old_time+dt/2)
                    save('frame_%06d_end.npz'%count,state,clock)
                    next_save = physical_time+.02
                temp = out/'checkpoint.tmp'
                with temp.open('wb') as file:
                    np.savez_compressed(file,packed=state,packed_older=older,t=clock,dt_previous=previous_dt,z=solver.z,y=solver.y)
                temp.replace(out/'checkpoint.npz')
                print('PROGRESS',json.dumps(last),flush=True)
                dt = min(4.,dt*min(1.5,max(.5,.9*max(error,1e-12)**(-1/3))))
                if args.max_steps and count>=args.max_steps:
                    status = 'PROBE_COMPLETE'
                    break
            else:
                status = 'TIME_HORIZON'
        except RuntimeError as exc:
            status = 'NUMERICAL_LIMIT'
            reason = str(exc)
    save('final.npz',state,clock)
    report = dict(status=status,reason=reason if status=='NUMERICAL_LIMIT' else '',
        accepted_steps=count,rejections=rejects,filament_candidate=detected,
        max_candidate_duration_s=maximum_candidate_duration,last=last,wall_s=time.time()-started)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print('RESULT',json.dumps(report),flush=True)
    if status=='NUMERICAL_LIMIT':
        raise SystemExit(2)


if __name__=='__main__':
    main()
