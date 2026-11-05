"""Closed periodic circular tube: gas core and Newtonian wall film.

Research extension of the frozen JAM gas-neck solver.  Rwall=1, rho_l=1,
sigma=1; time unit sqrt(rho_l Rwall^3/sigma).  Both phases retain inertia.
The wall has w=u=0 and continuity, NOT an open pressure boundary.
Two bordered constraints fix a collocated pressure-tail null mode and the
common pressure constant.  Their compatibility multipliers and the actual
divergence are reported separately; they are never physical source terms.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import sympy as sy
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, MatrixRankWarning
import gas_neck_jam_v3 as base


def wall_equations():
    fields = [[sy.symbols('%s_%s' % (name,d))
               for d in ('v','z','zz','y','yy','zy','t')]
              for name in ('wi','ui','pi','hi','wo','uo','po','ho')]
    yi,yo,R,mu,rg,mug=sy.symbols('yi yo R mu rg mug',positive=True)
    w,u,p,h=fields[4:8]
    r=h[0]*sy.exp(yo*sy.log(R/h[0]))
    b=sy.diff(r,yo)
    az=-sy.diff(r,h[0])*h[1]/b
    div=w[1]+az*w[3]+u[3]/b+u[0]/r
    eqs=[w[0],u[0],div,h[3]]
    args=sum(fields,[])+[yi,yo,R,mu,rg,mug]
    entries=[]
    for e,q in enumerate(eqs):
        for j,v in enumerate(sum(fields,[])):
            d=sy.diff(q,v)
            if d!=0:entries.append((e,j//7,j%7,d))
    return (sy.lambdify(args,eqs+[v[3] for v in entries],'numpy',cse=True),
            [(v[0],v[1],v[2]) for v in entries])


class TubeSolver(base.Solver):
    def __init__(self,ns=97,ny=15,Oh=1.,L=7.108612701053386,
                 mug=0.0002545584412271571,cluster=.85):
        super().__init__(ns,ny,Oh,R=1.,L=L,cluster=cluster)
        self.mug=mug
        self.compiled['far']=wall_equations()
        # The base's phase coefficients are retained except that mu_g is fixed
        # by the user's physical gas viscosity, independently of liquid Oh.
        for kind,(fn,entries) in list(self.compiled.items()):
            def fixed_gas(*args,_fn=fn):
                return _fn(*args[:-1],self.mug)
            self.compiled[kind]=(fixed_gas,entries)
        nphys=8*self.n
        it=(ny-1)*ns+np.arange(ns)
        self.gas_tau=sp.csc_matrix((np.ones(ns),
                     (2*self.n+it,np.zeros(ns,dtype=int))),shape=(nphys+1,1))
        # Absolute pressure gauge: average gas pressure along the axis is zero.
        self.mean_gauge=sp.csc_matrix((np.ones(ns)/ns,
                     (np.zeros(ns,dtype=int),2*self.n+np.arange(ns))),
                     shape=(1,nphys+1))

    def initial(self,amp=.1):
        h=.8+amp*np.cos(2*np.pi*self.z/self.L)
        hz=self.ops[1][:self.ns,:self.ns]@h
        hzz=self.ops[2][:self.ns,:self.ns]@h
        kap=1/(h*np.sqrt(1+hz*hz))-hzz/(1+hz*hz)**1.5
        X=np.zeros((8,self.ny,self.ns))
        X[3]=h;X[7]=h
        X[6]=-kap
        return np.r_[X.ravel(),0.,0.]

    def evaluate(self,x,hist,bp,jac=True):
        result=super().evaluate(x[:-1],hist,bp,jac)
        r,J=result if jac else (result,None)
        r=r+self.gas_tau.toarray().ravel()*x[-1]
        r=np.r_[r,float((self.mean_gauge@x[:-1])[0])]
        if not jac:return r
        return r,sp.bmat([[J,self.gas_tau],[self.mean_gauge,None]],format='csc')

    def metrics(self,x):
        return super().metrics(x[:-1])

    def diagnostics(self,x):
        X=x[:-2].reshape(8,self.ny,self.ns)
        divs=[]
        for off in (0,4):
            h=X[off+3].ravel()
            hz=self.ops[1]@h
            y=np.repeat(self.y,self.ns)
            wy=self.ops[3]@X[off].ravel()
            wz=self.ops[1]@X[off].ravel()
            uy=self.ops[3]@X[off+1].ravel()
            if off==0:
                r=y*h;b=h;a=-y*hz/h
            else:
                r=h*np.exp(y*np.log(1/h));b=r*np.log(1/h)
                a=-(1-y)*hz/(h*np.log(1/h))
            with np.errstate(divide='ignore',invalid='ignore'):
                d=wz+a*wy+uy/b+X[off+1].ravel()/r
            if off==0:d[:self.ns]=wz[:self.ns]+2*uy[:self.ns]/h[:self.ns]
            divs.append(d)
        h=X[3,-1]
        radial=float(1/np.expm1((self.y[1]-self.y[0])*np.log(1/float(h.min()))))
        return {'max_actual_divergence':float(np.max(np.abs(divs))),
                'max_wall_speed':float(np.max(np.abs(X[4:6,-1]))),
                'outer_tau':float(x[-2]),'gas_tau':float(x[-1]),
                'radial_neck_cells':radial}

    def newton(self,xold,xolder,dt,dtold,first=False):
        if first:bp,bm,bmm=1/dt,-1/dt,0
        else:
            bp=(2*dt+dtold)/(dt*(dt+dtold))
            bm=-(dt+dtold)/(dt*dtold)
            bmm=dt/(dtold*(dt+dtold))
        hist=bm*xold[:-2].reshape(8,self.n)+bmm*xolder[:-2].reshape(8,self.n)
        x=xold.copy()
        for it in range(16):
            r,J=self.evaluate(x,hist,bp)
            err=float(np.max(np.abs(r)))
            if err<1e-8:return x,it,err
            if not np.isfinite(r).all():raise RuntimeError('nonfinite residual')
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning)
                dx=spsolve(J,-r)
            if not np.isfinite(dx).all():raise RuntimeError('nonfinite Newton update')
            alpha=1.
            for ls in range(14):
                trial=x+alpha*dx
                h=trial[:-2].reshape(8,self.n)[[3,7]]
                if h.min()>1e-7 and h.max()<.999:
                    rt=self.evaluate(trial,hist,bp,False)
                    if np.isfinite(rt).all() and abs(rt).max()<err*(1-1e-4*alpha):
                        x=trial;break
                alpha*=.5
            else:raise RuntimeError('Newton line search failed')
        raise RuntimeError('Newton iteration limit')

    def check(self):
        x=self.initial(0)
        hist=-x[:-2].reshape(8,self.n)/.01
        r,J=self.evaluate(x,hist,100.)
        v=np.random.RandomState(713).randn(x.size);v/=np.linalg.norm(v)
        eps=1e-6
        fd=(self.evaluate(x+eps*v,hist,100,False)-self.evaluate(x-eps*v,hist,100,False))/(2*eps)
        rel=float(np.linalg.norm(fd-J@v)/np.linalg.norm(fd))
        out={'static_residual':float(abs(r).max()),'jacobian_direction_relative_error':rel,
             'static_diagnostics':self.diagnostics(x)}
        if x.size<2000:
            sv=np.linalg.svd(J.toarray(),compute_uv=False)
            out['static_matrix_min_singular_value']=float(sv[-1])
            out['static_matrix_relative_min_singular_value']=float(sv[-1]/sv[0])
        return out


def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'frames').mkdir()
    start=time.time();s=TubeSolver(a.ns,a.ny,a.oh,a.length,a.mug,a.cluster)
    checks=s.check()
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    print('CHECKS',json.dumps(checks),flush=True)
    if checks['static_residual']>1e-8 or checks['jacobian_direction_relative_error']>1e-5:
        raise RuntimeError('startup check failed')
    config=dict(vars(a),**{'Rwall':1.,'mean_gas_radius':.8,'gas_density':.001,
                   'initial_amplitude':a.amp,'wall':'no slip, impermeable',
                   'axial_boundary':'periodic','time_unit_s':0.0008838834764831844,
                   'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    x=s.initial(a.amp);older=x.copy();t=0.;dt=min(a.dtmax,.05*a.oh);dtold=dt
    step=0;rejects=0;consecutive=0;v0=s.metrics(x)[1];status='WALL_LIMIT'
    analytic=np.pi*a.length*(.8**2+a.amp**2/2)
    if abs(v0/analytic-1)>1e-10:raise RuntimeError('initial analytical volume mismatch')
    def save(name):
        np.savez_compressed(out/name,x=x,older=older,t=t,dtold=dtold,step=step,z=s.z,y=s.y)
    with (out/'diagnostics.csv').open('w',newline='') as fp:
        fields=['step','t','t_over_Oh','dt','hmin','hmax','gas_volume','relative_volume_error',
                'axial_neck_cells','shape_mismatch','newton_it','algebraic_residual',
                'wall_seconds','rejected','max_actual_divergence','max_wall_speed',
                'outer_tau','gas_tau','radial_neck_cells']
        writer=csv.DictWriter(fp,fieldnames=fields);writer.writeheader()
        def record(it,err):
            h,v,c,mis=s.metrics(x);d=s.diagnostics(x)
            row=dict(zip(fields[:14],[step,t,t/a.oh,dt,h,
                     float(x[:-2].reshape(8,a.ny,a.ns)[3,-1].max()),v,(v-v0)/v0,
                     c,mis,it,err,time.time()-start,rejects]))
            row.update(d)
            writer.writerow(row);fp.flush();return row
        row=record(0,0);save('frames/step-000000.npz')
        time_tol=1e-12*max(1.,a.tmax)
        while t<a.tmax-time_tol and time.time()-start<a.wall:
            h=row['hmin']
            # Slow initial film drainage is allowed a larger implicit step.
            # Below h=.5, use the conservative local capillary-time cap.
            cap=a.dtmax if h>.5 else .04*h**1.5+.02*a.oh*h
            dt=min(dt,a.dtmax,a.tmax-t,cap)
            if dt<1e-9:status='DT_FLOOR';break
            try:
                xn,it,err=s.newton(x,older,dt,dtold,step==0)
                hn,vn,_,_=s.metrics(xn)
                if abs(vn/v0-1)>.003:raise RuntimeError('volume drift >0.3 percent')
                if abs(hn-h)>.04*h:raise RuntimeError('neck changes >4 percent')
            except (RuntimeError,ValueError) as e:
                rejects+=1;consecutive+=1;dt*=.5
                print('REJECT',rejects,str(e),'dt',dt,flush=True)
                if consecutive>=30:status='NEWTON_FAIL';break
                continue
            older=x;x=xn;t+=dt;dtold=dt;step+=1;consecutive=0
            row=record(it,err)
            if step%20==0:print('PROGRESS',json.dumps(row),flush=True)
            if step%a.frame_every==0 or row['hmin']<=a.hstop:save('frames/step-%06d.npz'%step)
            if step%100==0:save('checkpoint.npz')
            if abs(row['relative_volume_error'])>.001 or row['max_actual_divergence']>.001:
                status='NUMERICAL_GATE';break
            if row['hmin']<=a.hstop:status='HSTOP';break
            if row['hmax']>=.995:status='THIN_WALL_FILM_GATE';break
            if step>=a.steps:status='STEP_LIMIT';break
            if it<=5:dt=min(a.dtmax,dt*1.2)
        else:status='TMAX' if t>=a.tmax-time_tol else 'WALL_LIMIT'
    save('final.npz');save('frames/step-%06d.npz'%step)
    summary=dict(row,status=status,observed_breakup=False,checks=checks)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY',json.dumps(summary),flush=True)
    return 2 if status in ('DT_FLOOR','NEWTON_FAIL','NUMERICAL_GATE') else 0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--oh',type=float,required=True)
    p.add_argument('--length',type=float,required=True)
    p.add_argument('--mug',type=float,default=1.8e-5/np.sqrt(1000*.02*250e-6))
    p.add_argument('--ns',type=int,default=97);p.add_argument('--ny',type=int,default=15)
    p.add_argument('--cluster',type=float,default=.85);p.add_argument('--amp',type=float,default=.1)
    p.add_argument('--dtmax',type=float,default=.1);p.add_argument('--tmax',type=float,default=10.)
    p.add_argument('--hstop',type=float,default=.03);p.add_argument('--wall',type=float,default=3600)
    p.add_argument('--steps',type=int,default=1000000);p.add_argument('--frame-every',type=int,default=5)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if not (0<a.oh and a.ns>=9 and a.ny>=5 and 0<a.dtmax<=a.tmax and a.length>0):
        p.error('invalid material, mesh or time parameters')
    sys.exit(run(a))
