"""Own tube, exact BDF continuation; no radius-based completion condition."""
import argparse,csv,hashlib,json,time
from pathlib import Path
import numpy as np
from mixed_pressure import Mixed

p=argparse.ArgumentParser()
p.add_argument('--restart',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--deadline',type=float,required=True)
p.add_argument('--wall',type=float,default=85000)
p.add_argument('--tend',type=float,default=600.)
p.add_argument('--max-steps',type=int,default=0)
p.add_argument('--frame-interval',type=float,default=.1)
a=p.parse_args();out=a.output;out.mkdir(exist_ok=False);start=time.time()
s=Mixed(193,21,Oh=1/np.sqrt(2),L=7.108612701053386,cluster=.95)
check=s.check_mixed();assert check['static_residual']<1e-8 and check['jacobian_direction_error']<1e-5
with np.load(a.restart) as n:
    assert all(k in n.files for k in ['packed','packed_older','previous_dt','next_step','accepted_macrosteps'])
    assert np.allclose(n['z'],s.z,atol=1e-12,rtol=0) and np.allclose(n['y'],s.y,atol=1e-12,rtol=0)
    q=n['packed'].copy();older=n['packed_older'].copy();t=float(n['t'])
    previous_dt=float(n['previous_dt']);step=float(n['next_step']);count=int(n['accepted_macrosteps'])
    assert np.max(abs(s.unpack(q)-n['x']))<1e-10
initial_time=t;local_steps=0;rejections=0;records=[];status='RUNNING'
next_frame=(np.floor(t/a.frame_interval)+1)*a.frame_interval
v0=np.pi*s.L*.645
config=dict(ns=193,ny=21,length=s.L,L=s.L,oh=s.Oh,Oh=s.Oh,mug=s.mug,cluster=.95,gas_regular=True,
    restart_mode='exact_saved_BDF_history',t_start=t,target_radius=None,preclosure_stop=False,
    source_native=str(a.restart),source_native_sha256=hashlib.sha256(a.restart.read_bytes()).hexdigest(),
    initial_history_prefix='Archived Newtonian circular-tube history',
    save_policy='both accepted half-step states; checkpoint includes complete BDF history',
    static_check=check,physical_time_unit_s=.0008838834764831844,
    original_initial_condition='a/R=.8+.1*cos(2pi*z/L); original zero-velocity history retained')
(out/'config.json').write_text(json.dumps(config,indent=2))
np.savez_compressed(out/'initial.npz',x=s.unpack(q),z=s.z,y=s.y,t=t)

def inspect(value,clock):
    native=s.unpack(value);x=native[:-2].reshape(8,21,193);d=s.diagnostics(native)
    mass=s.metrics(native)[1]/v0-1
    assert np.isfinite(native).all() and d['max_actual_divergence']<.001,(clock,d)
    assert abs(mass)<.001 and abs(value[-1])<1e-7,(clock,mass,value[-1])
    assert d['max_wall_speed']<1e-8,(clock,d)
    return native,x,d,mass

def save_history():
    tmp=out/'history.tmp'
    with tmp.open('wb') as f:
        np.savez_compressed(f,x=s.unpack(q),packed=q,packed_older=older,z=s.z,y=s.y,t=t,
            previous_dt=previous_dt,next_step=step,next_frame=next_frame,accepted_macrosteps=count)
    tmp.replace(out/'history_final.npz')

try:
    inspect(q,t)
    while t<a.tend:
        if time.time()>min(start+a.wall,a.deadline-120):status='WALL_LIMIT';break
        gap=min(next_frame,a.tend)-t
        step=gap/2 if step<gap<2*step else min(step,gap)
        for attempt in range(16):
            assert step>=1e-6,'adaptive timestep floor (numerical limit, not closure)'
            try:
                full,_,_=s.advance(q,older,step,previous_dt,False)
                mid,_,_=s.advance(q,older,step/2,previous_dt,False)
                fine,_,_=s.advance(mid,q,step/2,step/2)
            except RuntimeError as error:
                rejections+=1;step*=.5;print('REJECT_NEWTON',t,step,str(error),flush=True);continue
            native,x,d,mass=inspect(fine,t+step);mid_native,_,_,_=inspect(mid,t+step/2)
            xf=s.unpack(full)[:-2].reshape(8,21,193)
            herr=float(np.max(abs(xf[3,-1]/x[3,-1]-1)))
            perr=float(np.linalg.norm(xf[[2,6]]-x[[2,6]])/max(np.linalg.norm(x[[2,6]]),1e-12))
            vel=x[[0,1,4,5]]
            verr=float(np.linalg.norm(xf[[0,1,4,5]]-vel)/max(np.linalg.norm(vel),1e-6*np.sqrt(vel.size)))
            scaled=max(herr/.0005,perr/.005,verr/.005)
            if scaled<=1:break
            rejections+=1;step*=max(.2,.8*scaled**(-1/3));print('REJECT_ERROR',t,step,scaled,flush=True)
        else:raise RuntimeError('adaptive rejection limit')
        old_time=t;t+=step;count+=1;local_steps+=1;q=fine;older=mid;previous_dt=step/2
        for suffix,values,clock in [('mid',mid_native,old_time+step/2),('end',native,t)]:
            np.savez_compressed(out/('frame_%06d_%s.npz'%(count,suffix)),x=values,z=s.z,y=s.y,t=clock)
        row=dict(step=count,t=t,a_min=float(x[3,-1].min()),base_dt=step,accepted_dt=step/2,
            local_profile_error=herr,local_pressure_error=perr,local_velocity_error=verr,
            actual_divergence=d['max_actual_divergence'],volume_error=mass,
            pressure_max=float(abs(x[[2,6]]).max()),wall_speed=d['max_wall_speed'],wall_seconds=time.time()-start)
        records.append(row)
        with (out/'diagnostics.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(records)
        if t>=next_frame-1e-10:next_frame+=a.frame_interval
        step=min(4.,step*min(1.5,max(.5,.9*max(scaled,1e-12)**(-1/3))))
        save_history();print('PROGRESS',json.dumps(row),flush=True)
        if a.max_steps and local_steps>=a.max_steps:status='PROBE_COMPLETE';break
    else:status='TIME_HORIZON'
except Exception as error:
    status='NUMERICAL_LIMIT';print('STOP',repr(error),flush=True)
np.savez_compressed(out/'final.npz',x=s.unpack(q),z=s.z,y=s.y,t=t)
save_history()
summary=dict(status=status,t_start=initial_time,t=t,closed=False,preclosure_stop=False,
    accepted_macrosteps=local_steps,rejections=rejections,last=records[-1] if records else None,
    wall_seconds=time.time()-start,note='No radius target. Numerical termination never proves topology closure.')
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if status=='NUMERICAL_LIMIT':raise SystemExit(2)
