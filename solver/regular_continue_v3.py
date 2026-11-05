"""Regular gas-axis formulation for the frozen circular-tube problem.

Gas coordinate s=(r/a)^2; stored gas variable ui=v=u_r/r.  All gas
unknowns are smooth even functions of physical r.  Axis rows are the
finite axial momentum, radial momentum/r, and continuity limits, rather
than extrapolating a continuity equation omitted at the axis.
Outer-liquid equations, stress conditions and physical parameters stay fixed.
"""
import numpy as np
import sympy as sy
from scipy.interpolate import BarycentricInterpolator, CubicSpline
from scipy.signal import resample
from tube_film_jam_run import TubeSolver as OriginalTubeSolver
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve,MatrixRankWarning
import warnings


class ConstantPreserving:
    """Evaluate derivatives as neighbor differences, including dense Chebyshev rows."""
    def __init__(self,matrix):
        m=matrix.tocoo();keep=m.row!=m.col
        self.row=m.row[keep];self.col=m.col[keep];self.data=m.data[keep];self.shape=m.shape
        diagonal=-np.bincount(self.row,weights=self.data,minlength=m.shape[0])
        self.matrix=(sp.coo_matrix((self.data,(self.row,self.col)),shape=m.shape)+sp.diags(diagonal)).tocsr()
    def __matmul__(self,x):
        return np.bincount(self.row,weights=self.data*(x[self.col]-x[self.row]),minlength=self.shape[0])
    def __getitem__(self,key):return self.matrix[key]


def regular_equations():
    fields=[[sy.symbols('%s_%s'%(name,d)) for d in ('v','z','zz','y','yy','zy','t')]
            for name in ('wi','ui','pi','hi','wo','uo','po','ho')]
    yi,yo,R,mu,rg,mug=sy.symbols('yi yo R mu rg mug',positive=True)
    w,v,p,h=fields[:4];a,az,azz,_,_,_,at=h
    c=-2*yi*az/a
    cz=-2*yi*azz/a+6*yi*az**2/a**2
    def gas_deriv(q):
        return q[1]+c*q[3],q[2]+2*c*q[5]+c*c*q[4]+cz*q[3],q[6]-2*yi*at/a*q[3]
    wz,wzz,wt=gas_deriv(w);vz,vzz,vt=gas_deriv(v);pz,_,_=gas_deriv(p)
    gas=[rg*(wt+2*yi*v[0]*w[3]+w[0]*wz)+pz-mug*(4/a**2*(w[3]+yi*w[4])+wzz),
         rg*(vt+v[0]*(v[0]+2*yi*v[3])+w[0]*vz)+2*p[3]/a**2-mug*(4/a**2*(2*v[3]+yi*v[4])+vzz),
         wz+2*v[0]+2*yi*v[3],h[4]]
    ri=a*sy.sqrt(yi)
    di=(2*ri/a**2*w[3],wz,v[0]+2*yi*v[3],ri*vz)
    wo,uo,po,ho=fields[4:8]
    ro=ho[0]*sy.exp(yo*sy.log(R/ho[0]));b=sy.diff(ro,yo)
    c=-sy.diff(ro,ho[0])*ho[1]/b
    do=(wo[3]/b,wo[1]+c*wo[3],uo[3]/b,uo[1]+c*uo[3])
    outer_div=do[1]+do[2]+uo[0]/ro
    norm=sy.sqrt(1+h[1]**2);nz=-h[1]/norm;nr=1/norm;tz=nr;tr=-nz
    def traction(pressure,d,visc):
        wr,wz,ur,uz=d
        zz=-pressure+2*visc*wz;rr=-pressure+2*visc*ur;zr=visc*(wr+uz)
        return (nz*nz*zz+nr*nr*rr+2*nz*nr*zr,
                tz*nz*zz+tr*nr*rr+(tz*nr+tr*nz)*zr)
    ni,ti=traction(p[0],di,mug);no,to=traction(po[0],do,mu)
    kap=1/(a*norm)-h[2]/norm**3
    kinds={'ib':gas,'axis':[gas[0],gas[1],gas[2],h[3]],
           'ii':[no-ni-kap,to-ti,gas[2],h[6]+w[0]*h[1]-a*v[0]],
           'oi':[wo[0]-w[0],uo[0]-a*v[0],outer_div,ho[0]-a]}
    av=sum(fields,[]);args=av+[yi,yo,R,mu,rg,mug];compiled={}
    for kind,eqs in kinds.items():
        entries=[]
        for row,q in enumerate(eqs):
            for j,u in enumerate(av):
                dq=sy.diff(q,u)
                if dq!=0:entries.append((row,j//7,j%7,dq))
        compiled[kind]=(sy.lambdify(args,eqs+[x[3] for x in entries],'numpy',cse=True),
                        [(x[0],x[1],x[2]) for x in entries])
    return compiled


class Solver(OriginalTubeSolver):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for kind,(fn,entries) in regular_equations().items():
            def fixed_gas(*args,_fn=fn):return _fn(*args[:-1],self.mug)
            self.compiled[kind]=(fixed_gas,entries)
        self.ops=[op if j in [0,6] else ConstantPreserving(op) for j,op in enumerate(self.ops)]
        for name,phase,rows,ii,oo in self.groups:
            self.cache[name]=[[op[idx,:].tocoo() for op in self.ops] for idx in (ii,oo)]

    def initial(self,amp=.1):
        h=.8+amp*np.cos(2*np.pi*self.z/self.L);full=np.tile(h,self.ny)
        hz=(self.ops[1]@full)[:self.ns];hzz=(self.ops[2]@full)[:self.ns]
        kap=1/(h*np.sqrt(1+hz*hz))-hzz/(1+hz*hz)**1.5
        X=np.zeros((8,self.ny,self.ns));X[3]=h;X[7]=h;X[6]=-kap
        return np.r_[X.ravel(),0.,0.]

    def newton(self,xold,xolder,dt,dtold,first=False):
        if first:bp,bm,bmm=1/dt,-1/dt,0
        else:
            bp=(2*dt+dtold)/(dt*(dt+dtold));bm=-(dt+dtold)/(dt*dtold);bmm=dt/(dtold*(dt+dtold))
        hist=bm*xold[:-2].reshape(8,self.n)+bmm*xolder[:-2].reshape(8,self.n);x=xold.copy()
        for it in range(16):
            residual,J=self.evaluate(x,hist,bp);err=float(abs(residual).max())
            if err<1e-8:return x,it,err
            if not np.isfinite(residual).all():raise RuntimeError('nonfinite residual')
            # Explicit row/column equilibration affects linear algebra only.
            # The Newton line search and acceptance use original residuals.
            row=1/np.maximum(abs(J).max(axis=1).toarray().ravel(),1e-30)
            A=sp.diags(row)@J;col=1/np.maximum(abs(A).max(axis=0).toarray().ravel(),1e-30)
            A=(A@sp.diags(col)).tocsc()
            with warnings.catch_warnings():
                warnings.simplefilter('error',MatrixRankWarning)
                dx=col*spsolve(A,-row*residual,permc_spec='COLAMD')
            if not np.isfinite(dx).all():raise RuntimeError('nonfinite Newton update')
            linear=float(abs(J@dx+residual).max())
            if linear>max(1e-7,1e-5*err):raise RuntimeError('unscaled linear residual')
            alpha=1.
            for trial_number in range(14):
                trial=x+alpha*dx;hh=trial[:-2].reshape(8,self.n)[[3,7]]
                if hh.min()>1e-7 and hh.max()<.999:
                    rr=self.evaluate(trial,hist,bp,False)
                    if np.isfinite(rr).all() and abs(rr).max()<err*(1-1e-4*alpha):x=trial;break
                alpha*=.5
            else:raise RuntimeError('Newton line search failed')
        raise RuntimeError('Newton iteration limit')

    def diagnostics(self,x):
        X=x[:-2].reshape(8,self.ny,self.ns);s=self.y[:,None]
        h=X[3];hz=(self.ops[1]@h.ravel()).reshape(h.shape)
        wy=(self.ops[3]@X[0].ravel()).reshape(h.shape)
        vy=(self.ops[3]@X[1].ravel()).reshape(h.shape)
        wz=(self.ops[1]@X[0].ravel()).reshape(h.shape)
        gas=wz-2*s*hz/h*wy+2*X[1]+2*s*vy
        ho=X[7];hz=(self.ops[1]@ho.ravel()).reshape(ho.shape)
        r=ho*np.exp(s*np.log(1/ho));b=r*np.log(1/ho)
        c=-(1-s)*hz/(ho*np.log(1/ho))
        wz=(self.ops[1]@X[4].ravel()).reshape(ho.shape)
        wy=(self.ops[3]@X[4].ravel()).reshape(ho.shape)
        uy=(self.ops[3]@X[5].ravel()).reshape(ho.shape)
        liquid=wz+c*wy+uy/b+X[5]/r
        hm=float(h[-1].min())
        return {'max_actual_divergence':float(max(abs(gas).max(),abs(liquid).max())),
                'gas_divergence_max':float(abs(gas).max()),'gas_axis_divergence_max':float(abs(gas[0]).max()),
                'liquid_divergence_max':float(abs(liquid).max()),
                'max_wall_speed':float(abs(X[4:6,-1]).max()),
                'outer_tau':float(x[-2]),'gas_tau':float(x[-1]),
                'radial_neck_cells':float(1/np.expm1((self.y[1]-self.y[0])*np.log(1/hm)))}

    def convert_linear_state(self,x,old_y):
        """Same z mesh: convert primitive u_r(y) to regular v(s), no projection."""
        old=x[:-2].reshape(8,len(old_y),self.ns);new=np.empty((8,self.ny,self.ns))
        from gas_neck_jam_v3 import Chevigood
        _,Dy,_=Chevigood(len(old_y)-1,1,0)
        rr=np.sqrt(self.y)
        for j in [0,2,3]:new[j]=BarycentricInterpolator(old_y,old[j],axis=0)(rr)
        ur=BarycentricInterpolator(old_y,old[1],axis=0)(rr)
        h=new[3]
        new[1,1:]=ur[1:]/(rr[1:,None]*h[1:])
        new[1,0]=(Dy@old[1])[0]/h[0]
        for j in range(4,8):new[j]=BarycentricInterpolator(old_y,old[j],axis=0)(self.y)
        return np.r_[new.ravel(),x[-2:]]

    def remap_linear_snapshot(self,snapshot,key='x'):
        old=snapshot[key]
        X=old[:-2].reshape(8,len(snapshot['y']),len(snapshot['z']))
        if X.shape[-1]!=self.ns or np.max(abs(snapshot['z']-self.z))>1e-12:
            z=np.r_[snapshot['z'],self.L]
            X=CubicSpline(z,np.concatenate([X,X[...,:1]],axis=2),axis=2,bc_type='periodic')(self.z)
        return self.convert_linear_state(np.r_[X.ravel(),old[-2:]],snapshot['y'])

    def remap_regular_state(self,snapshot,key='x'):
        old=snapshot[key];oldny=len(snapshot['y']);oldns=len(snapshot['z'])
        X=old[:-2].reshape(8,oldny,oldns)
        if oldns!=self.ns or np.max(abs(snapshot['z']-self.z))>1e-12:
            X=CubicSpline(np.r_[snapshot['z'],self.L],np.concatenate([X,X[...,:1]],axis=2),axis=2,bc_type='periodic')(self.z)
        if oldny!=self.ny:X=BarycentricInterpolator(snapshot['y'],X,axis=1)(self.y)
        return np.r_[X.ravel(),old[-2:]]

    def primitive_gas(self,x):
        X=x[:-2].reshape(8,self.ny,self.ns)
        return X[0],X[3]*np.sqrt(self.y[:,None])*X[1]
