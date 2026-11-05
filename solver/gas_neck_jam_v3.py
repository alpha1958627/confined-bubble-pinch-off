"""Two-fluid axisymmetric Newtonian gas-neck extension of the JAM formulation.
Analytical local Jacobians (SymPy), sparse collocation/FD assembly, implicit BDF.
Reference: Herrada & Montanero, JCP 306 (2016); miguelherrada/JAM.
Local research extension: NOT an upstream validated gas-neck example.
"""
import argparse
import csv
import json
import os
import time
import warnings
from pathlib import Path
import numpy as np
import sympy as sy
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, MatrixRankWarning
from scipy.optimize import least_squares
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "source"))
from collocation import Chevigood

def symbolic():
    fields = [[sy.symbols("%s_%s" % (name, d)) for d in ("v","z","zz","y","yy","zy","t")]
              for name in ("wi","ui","pi","hi","wo","uo","po","ho")]
    yi, yo, rr, mu, rg, mug = sy.symbols("yi yo R mu rg mug", positive=True)
    allvars = sum(fields, [])
    def phase(offset, y, outer, density, viscosity):
        w,u,p,h = fields[offset:offset+4]
        f,hz,hzz,_,_,_,ht = h
        if outer:
            r = f*sy.exp(y*sy.log(rr/f))
        else:
            r = y*f
        b = sy.diff(r,y)
        a = -sy.diff(r,f)*hz/b
        ct = -sy.diff(r,f)*ht/b
        az = sy.diff(a,f)*hz + sy.diff(a,hz)*hzz + sy.diff(a,y)*a
        def deriv(q):
            q0,qz,qzz,qy,qyy,qzy,qt=q
            dr=qy/b
            dz=qz+a*qy
            drr=qyy/b**2-qy*sy.diff(b,y)/b**3
            dzz=qzz+2*a*qzy+a*a*qyy+az*qy
            dt=qt+ct*qy
            return dr,dz,drr+dzz,dt
        wr,wz,wl,wt=deriv(w)
        ur,uz,ul,ut=deriv(u)
        pr,pz,_,_=deriv(p)
        bulk=[density*(wt+u[0]*wr+w[0]*wz)+pz-viscosity*(wl+wr/r),
              density*(ut+u[0]*ur+w[0]*uz)+pr-viscosity*(ul+ur/r-u[0]/r**2),
              wz+ur+u[0]/r,h[4]]
        return bulk,(wr,wz,ur,uz,pr),r
    bi,di,_=phase(0,yi,False,rg,mug)
    bo,do,_=phase(4,yo,True,1,mu)
    hi=fields[3]; ho=fields[7]
    slope=hi[1]
    norm=sy.sqrt(1+slope**2)
    nz,nr=-slope/norm,1/norm
    tz,tr=1/norm,slope/norm
    def traction(offset,d,visc):
        wr,wz,ur,uz,pr=d
        p=fields[offset+2][0]
        zz=-p+2*visc*wz
        rrr=-p+2*visc*ur
        zr=visc*(wr+uz)
        normal=nz*nz*zz+nr*nr*rrr+2*nz*nr*zr
        tangent=tz*nz*zz+tr*nr*rrr+(tz*nr+tr*nz)*zr
        return normal,tangent
    ni,ti=traction(0,di,mug)
    no,to=traction(4,do,mu)
    kap=1/(hi[0]*norm)-hi[2]/norm**3
    # Interface rows keep each phase's continuity equation.
    interface_i=[no-ni-kap,to-ti,bi[2],hi[6]+fields[0][0]*slope-fields[1][0]]
    interface_o=[fields[4][0]-fields[0][0],fields[5][0]-fields[1][0],
                 bo[2],ho[0]-hi[0]]
    axis=[di[0],fields[1][0],di[4],hi[3]]
    far=[do[0],do[2],fields[6][0],ho[3]]
    kinds={"ib":bi,"ob":bo,"axis":axis,"far":far,"ii":interface_i,"oi":interface_o}
    args=allvars+[yi,yo,rr,mu,rg,mug]
    compiled={}
    for name,eqs in kinds.items():
        entries=[]
        for row,e in enumerate(eqs):
            for j,v in enumerate(allvars):
                d=sy.diff(e,v)
                if d!=0: entries.append((row,j//7,j%7,d))
        fn=sy.lambdify(args,eqs+[v[3] for v in entries],"numpy",cse=True)
        compiled[name]=(fn,[(v[0],v[1],v[2]) for v in entries])
    return compiled

class Solver:
    def __init__(self, ns=96, ny=13, Oh=.1, R=4*np.pi, L=4*np.pi, cluster=.9):
        self.ns,self.ny,self.n=ns,ny,ns*ny
        self.Oh,self.R,self.L=Oh,R,L
        q=np.arange(ns)*L/ns
        k=2*np.pi/L
        self.z=q+cluster/k*np.sin(k*q)
        zs=1+cluster*np.cos(k*q)
        self.zweights=L/ns*zs
        zss=-cluster*k*np.sin(k*q)
        rows=np.arange(ns)
        d1=sp.lil_matrix((ns,ns));d2=sp.lil_matrix((ns,ns))
        for off,c1,c2 in [(-2,1,-1),(-1,-8,16),(0,0,-30),(1,8,16),(2,-1,-1)]:
            d1[rows,(rows+off)%ns]=c1/(12*(L/ns))
            d2[rows,(rows+off)%ns]=c2/(12*(L/ns)**2)
        dz=sp.diags(1/zs)@d1.tocsr()
        dzz=sp.diags(1/zs**2)@d2.tocsr()-sp.diags(zss/zs**3)@d1.tocsr()
        self.y,dy,dyy=Chevigood(ny-1,1,0)
        # A spectral pressure-tail constraint removes an unobservable mode.
        # A tau unknown distributes the compatibility residual over the entire
        # outer interface; no individual continuity point is silently omitted.
        wy=(-1.)**np.arange(ny);wy[[0,-1]]*=.5
        self.pressure_gauge=np.repeat(wy,ns)/ns
        self.tau_column=sp.csc_matrix((np.ones(ns),(6*self.n+np.arange(ns),np.zeros(ns,dtype=int))),shape=(8*self.n,1))
        self.gauge_vector=sp.csc_matrix((self.pressure_gauge,(np.zeros(self.n,dtype=int),6*self.n+np.arange(self.n))),shape=(1,8*self.n))
        I=sp.eye(self.n,format="csr")
        Dz=sp.kron(sp.eye(ny),dz,format="csr")
        Dzz=sp.kron(sp.eye(ny),dzz,format="csr")
        Dy=sp.kron(dy,sp.eye(ns),format="csr")
        Dyy=sp.kron(dyy,sp.eye(ns),format="csr")
        self.ops=[I,Dz,Dzz,Dy,Dyy,Dy@Dz,I]
        ax=np.arange(ns); it=(ny-1)*ns+ax
        bulk=np.arange(ns,(ny-1)*ns)
        self.groups=[("ib",0,bulk,bulk,bulk),("ob",1,bulk,bulk,bulk),
                     ("axis",0,ax,ax,ax),("far",1,it,it,it),
                     ("ii",0,it,it,ax),("oi",1,ax,it,ax)]
        self.cache={}
        for name,phase,rows_,ii,oo in self.groups:
            self.cache[name]=[[[op[idx,:].tocoo() for op in self.ops] for idx in (ii,oo)]][0]
        self.compiled=symbolic()
    def initial(self,amp=.05):
        h=1+amp*np.cos(2*np.pi*self.z/self.L)
        hz=self.ops[1][:self.ns,:self.ns]@h
        hzz=self.ops[2][:self.ns,:self.ns]@h
        kap=1/(h*np.sqrt(1+hz**2))-hzz/(1+hz**2)**1.5
        pg=float(np.mean(kap))
        X=np.zeros((8,self.ny,self.ns))
        X[2,:,:]=pg; X[3,:,:]=h
        X[6,:,:]=(pg-kap)[None,:]*(1-self.y[:,None])
        X[7,:,:]=h
        return np.r_[X.reshape(-1),0.]
    def evaluate(self,x,hist,bp,jac=True):
        X=x[:-1].reshape(8,self.n)
        vals=[[op@X[j] for op in self.ops[:6]]+[bp*X[j]+hist[j]] for j in range(8)]
        res=np.zeros(8*self.n+1)
        data=[]; rr=[]; cc=[]
        for name,phase,rows_,ii,oo in self.groups:
            args=[]
            for j in range(8):
                idx=ii if j<4 else oo
                args.extend([v[idx] for v in vals[j]])
            args.extend([np.repeat(self.y,self.ns)[ii],np.repeat(self.y,self.ns)[oo],
                         self.R,self.Oh,.001,self.Oh*.01])
            fn,entries=self.compiled[name]
            with np.errstate(divide="ignore",invalid="ignore",over="ignore"):
                out=fn(*args)
            for e in range(4):
                res[(phase*4+e)*self.n+rows_]=out[e]
            if not jac: continue
            for val,(e,j,d) in zip(out[4:],entries):
                coo=self.cache[name][0 if j<4 else 1][d]
                coef=np.broadcast_to(val,(len(rows_),))
                vv=coef[coo.row]*coo.data*(bp if d==6 else 1)
                data.append(vv);rr.append((phase*4+e)*self.n+rows_[coo.row]);cc.append(j*self.n+coo.col)
        res[6*self.n:6*self.n+self.ns]+=x[-1]
        res[-1]=self.pressure_gauge@X[6]
        if not jac:return res
        J=sp.coo_matrix((np.concatenate(data),(np.concatenate(rr),np.concatenate(cc))),
                       shape=(8*self.n,8*self.n)).tocsc()
        return res,sp.bmat([[J,self.tau_column],[self.gauge_vector,None]],format="csc")
    def newton(self,xold,xolder,dt,dtold,first=False):
        if first: bp,bm,bmm=1/dt,-1/dt,0
        else:
            bp=(2*dt+dtold)/(dt*(dt+dtold))
            bm=-(dt+dtold)/(dt*dtold)
            bmm=dt/(dtold*(dt+dtold))
        hist=bm*xold[:-1].reshape(8,self.n)+bmm*xolder[:-1].reshape(8,self.n)
        x=xold.copy()
        for it in range(16):
            r,J=self.evaluate(x,hist,bp)
            err=float(np.max(np.abs(r)))
            if err<1e-8:return x,it,err
            if not np.isfinite(r).all():raise RuntimeError("nonfinite residual")
            with warnings.catch_warnings():
                warnings.simplefilter("error",MatrixRankWarning)
                dx=spsolve(J,-r)
            if not np.isfinite(dx).all():raise RuntimeError("nonfinite Newton update")
            alpha=1.
            for ls in range(12):
                trial=x+alpha*dx
                hh=trial[:-1].reshape(8,self.n)[[3,7]]
                if np.min(hh)>1e-7 and np.max(hh)<self.R*.95:
                    rt=self.evaluate(trial,hist,bp,False)
                    if np.isfinite(rt).all() and np.max(np.abs(rt))<err*(1-1e-4*alpha):
                        x=trial;break
                alpha*=.5
            else:raise RuntimeError("Newton line search failed")
        raise RuntimeError("Newton iteration limit")
    def metrics(self,x):
        X=x[:-1].reshape(8,self.ny,self.ns)
        h=X[3,-1,:]
        # Periodic trapezoidal quadrature in computational coordinates,
        # including dz/dq. Physical-z trapezoids are only second order.
        vol=float(np.pi*np.dot(self.zweights,h*h))
        j=int(np.argmin(h))
        hz=self.ops[1]@X[3].ravel(); hzz=self.ops[2]@X[3].ravel()
        curvature_z=abs(hzz.reshape(self.ny,self.ns)[-1,j])
        ell=np.sqrt(float(h[j])/(curvature_z+1e-30))
        dzlocal=min((self.z[(j+1)%self.ns]-self.z[j])%self.L,
                    (self.z[j]-self.z[(j-1)%self.ns])%self.L)
        return float(h.min()),vol,float(ell/dzlocal),float(np.max(np.abs(X[3]-X[7])))
    def auxiliary(self,x):
        X=x[:-1].reshape(8,self.n)
        hz=self.ops[1]@X[7]; wy=self.ops[3]@X[4]
        uy=self.ops[3]@X[5]; wz=self.ops[1]@X[4]
        h=X[7][0]; ll=np.log(self.R/h)
        gauge_div=float(wz[0]-hz[0]/(h*ll)*wy[0]+uy[0]/(h*ll)+X[5][0]/h)
        hm=float(np.min(X[3]));dy=self.y[1]-self.y[0]
        radial_cells=float(1/np.expm1(dy*np.log(self.R/hm)))
        return gauge_div,radial_cells
    def check(self):
        x=self.initial(0)
        hist=-x[:-1].reshape(8,self.n)/.01
        r,J=self.evaluate(x,hist,100.)
        rng=np.random.RandomState(713)
        v=rng.randn(x.size);v/=np.linalg.norm(v)
        eps=1e-6
        fd=(self.evaluate(x+eps*v,hist,100,False)-self.evaluate(x-eps*v,hist,100,False))/(2*eps)
        rel=float(np.linalg.norm(fd-J@v)/np.linalg.norm(fd))
        return {"static_residual":float(abs(r).max()),"jacobian_direction_relative_error":rel}

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    start=time.time()
    s=Solver(a.ns,a.ny,a.oh,R=a.outer_radius)
    check=s.check()
    (out/"checks.json").write_text(json.dumps(check,indent=2))
    if check["static_residual"]>1e-8 or check["jacobian_direction_relative_error"]>1e-5:
        raise RuntimeError("startup numerical gate failed: "+str(check))
    x=s.initial();older=x.copy();t=0.;dtold=a.dtmax;dt=a.dtmax;step=0;rejects=0
    v0=s.metrics(x)[1];status="TIME_LIMIT"
    (out/"config.json").write_text(json.dumps(vars(a),indent=2))
    with (out/"diagnostics.csv").open("w",newline="") as fp:
        writer=csv.writer(fp)
        writer.writerow(["step","t","dt","hmin","gas_volume","relative_volume_error","axial_neck_cells","shape_mismatch","newton_it","residual","wall_seconds","rejected","gauge_continuity_residual","radial_neck_cells"])
        def record(it,err):
            h,v,c,mis=s.metrics(x)
            gd,rc=s.auxiliary(x)
            writer.writerow([step,t,dt,h,v,(v-v0)/v0,c,mis,it,err,time.time()-start,rejects,gd,rc]);fp.flush()
            return h,v,c
        record(0,0)
        while t<a.tmax and time.time()-start<a.wall:
            h,_,_,_=s.metrics(x)
            # Implicit BDF accuracy cap includes both inertial and viscous
            # local capillary times. Actual neck change is separately limited.
            dt=min(dt,a.dtmax,a.tmax-t,.04*h**1.5+.02*a.oh*h)
            if dt<1e-9:status="DT_FLOOR";break
            try:
                xn,it,err=s.newton(x,older,dt,dtold,step==0)
                hn,vn,cn,mis=s.metrics(xn)
                if abs((vn-v0)/v0)>.003:raise RuntimeError("volume drift above 0.3 percent")
                if abs(hn-h)>.04*h:raise RuntimeError("neck change exceeds 4 percent")
            except (RuntimeError,ValueError) as e:
                rejects+=1;dt*=.5
                if rejects%10==0:print("REJECT",rejects,str(e),"dt",dt,flush=True)
                continue
            older=x;x=xn;t+=dt;dtold=dt;step+=1
            h,v,c=record(it,err)
            gd,rc=s.auxiliary(x)
            if abs(gd)>1e-3 or abs((v-v0)/v0)>1e-3:
                status="NUMERICAL_GATE";break
            if step%20==0:print("PROGRESS",step,t,h,c,time.time()-start,flush=True)
            if step%100==0:np.savez_compressed(str(out/"checkpoint.npz"),x=x,older=older,t=t,dtold=dtold,z=s.z,y=s.y)
            if h<=a.hstop:status="HSTOP";break
            if step>=a.steps:status="STEP_LIMIT";break
            if it<=5:dt=min(a.dtmax,dt*1.2)
        else:
            status="TMAX" if t>=a.tmax else "TIME_LIMIT"
    np.savez_compressed(str(out/"final.npz"),x=x,older=older,t=t,dtold=dtold,z=s.z,y=s.y)
    h,v,c,mis=s.metrics(x)
    summary={"status":status,"step":step,"t":t,"hmin":h,"gas_volume_relative_error":(v-v0)/v0,
             "axial_neck_cells":c,"wall_seconds":time.time()-start,"rejected":rejects,"checks":check,
             "gauge_continuity_residual":s.auxiliary(x)[0],"radial_neck_cells":s.auxiliary(x)[1]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if status in ("DT_FLOOR",):return 2
    return 0

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--oh",type=float,default=.1);p.add_argument("--ns",type=int,default=96)
    p.add_argument("--ny",type=int,default=13);p.add_argument("--dtmax",type=float,default=.01)
    p.add_argument("--tmax",type=float,default=80);p.add_argument("--hstop",type=float,default=.05)
    p.add_argument("--steps",type=int,default=1000000);p.add_argument("--wall",type=float,default=72000)
    p.add_argument("--output",default="result")
    p.add_argument("--outer-radius",type=float,default=4*np.pi)
    sys.exit(run(p.parse_args()))
