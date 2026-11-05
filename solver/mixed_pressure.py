"""Candidate P_N/P_(N-2) radial pressure space for the tube extension.

Pressure is represented on two fewer radial points in each phase; continuity
is collocated at interior radial points. One common pressure constant is fixed
with a bordered compatibility multiplier. All true divergence, including both
boundaries, must still be audited. This is a candidate, not a validated solver.
"""
import numpy as np
from scipy import sparse as sp
from scipy.interpolate import BarycentricInterpolator
from scipy.sparse.linalg import spsolve
from regular_continue_v3 import Solver as Regular


class Mixed(Regular):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from gas_neck_jam_v3 import Chevigood
        self.yp=Chevigood(self.ny-3,1,0)[0]
        radial=BarycentricInterpolator(self.yp,np.eye(self.ny-2),axis=0)(self.y)
        pi=sp.kron(sp.csr_matrix(radial),sp.eye(self.ns),format='csr')
        self.P=sp.block_diag([pi if j in [2,6] else sp.eye(self.n) for j in range(8)],format='csr')
        removed=[]
        for off in [2,6]:
            removed.extend((off*self.n+np.arange(self.ns)).tolist())
            removed.extend((off*self.n+(self.ny-1)*self.ns+np.arange(self.ns)).tolist())
        self.keep=np.setdiff1d(np.arange(8*self.n),removed)
        assert len(self.keep)==self.P.shape[1]
        self.compat=np.isin(self.keep//self.n,[2,6]).astype(float)
        gauge=np.zeros(8*self.n);gauge[2*self.n:2*self.n+self.ns]=1/self.ns
        self.gauge=np.asarray(gauge@self.P).ravel()

    def pack(self,x):
        X=x[:-2].reshape(8,self.ny,self.ns);out=[]
        for j in range(8):
            out.append((BarycentricInterpolator(self.y,X[j],axis=0)(self.yp) if j in [2,6] else X[j]).ravel())
        return np.r_[np.concatenate(out),0.]

    def unpack(self,x):
        return np.r_[self.P@x[:-1],0.,0.]

    def residual(self,x,reference,slope,bp,jac=True):
        X=(self.P@x[:-1]).reshape(8,self.n)
        vals=[[op@X[j] for op in self.ops[:6]]+[bp*(X[j]-reference[j])+slope[j]] for j in range(8)]
        res=np.zeros(8*self.n);data=[];rr=[];cc=[]
        for name,phase,rows,ii,oo in self.groups:
            args=[]
            for j in range(8):
                idx=ii if j<4 else oo;args.extend([v[idx] for v in vals[j]])
            args.extend([np.repeat(self.y,self.ns)[ii],np.repeat(self.y,self.ns)[oo],self.R,self.Oh,.001,self.Oh*.01])
            fn,entries=self.compiled[name];out=fn(*args)
            for e in range(4):res[(phase*4+e)*self.n+rows]=out[e]
            if not jac:continue
            for val,(e,j,d) in zip(out[4:],entries):
                coo=self.cache[name][0 if j<4 else 1][d]
                coef=np.broadcast_to(val,(len(rows),))
                data.append(coef[coo.row]*coo.data*(bp if d==6 else 1))
                rr.append((phase*4+e)*self.n+rows[coo.row]);cc.append(j*self.n+coo.col)
        r=np.r_[res[self.keep]+self.compat*x[-1],self.gauge@x[:-1]]
        if not jac:return r
        J=sp.coo_matrix((np.concatenate(data),(np.concatenate(rr),np.concatenate(cc))),shape=(8*self.n,8*self.n)).tocsr()
        A=(J[self.keep]@self.P).tocsc()
        return r,sp.bmat([[A,sp.csc_matrix(self.compat[:,None])],[sp.csc_matrix(self.gauge[None,:]),None]],format='csc')

    def check_mixed(self):
        x=self.pack(self.initial(0));ref=(self.P@x[:-1]).reshape(8,self.n);slope=np.zeros_like(ref)
        r,J=self.residual(x,ref,slope,100.)
        v=np.random.RandomState(713).randn(len(x));v/=np.linalg.norm(v);eps=1e-6
        fd=(self.residual(x+eps*v,ref,slope,100,False)-self.residual(x-eps*v,ref,slope,100,False))/(2*eps)
        return {'static_residual':float(abs(r).max()),'jacobian_direction_error':float(np.linalg.norm(fd-J@v)/np.linalg.norm(fd)),
                'unknowns':len(x),'rows':J.shape[0]}

    def advance(self,old,older,dt,dtold,first=False):
        reference=(self.P@old[:-1]).reshape(8,self.n)
        bp=1/dt if first else (2*dt+dtold)/(dt*(dt+dtold))
        bmm=0. if first else dt/(dtold*(dt+dtold))
        slope=bmm*((self.P@older[:-1]).reshape(8,self.n)-reference)
        x=old.copy()
        for it in range(20):
            r,J=self.residual(x,reference,slope,bp);err=float(abs(r).max())
            if err<1e-8:return x,it,err
            row=1/np.maximum(abs(J).max(axis=1).toarray().ravel(),1e-30)
            A=sp.diags(row)@J;col=1/np.maximum(abs(A).max(axis=0).toarray().ravel(),1e-30)
            dx=col*spsolve((A@sp.diags(col)).tocsc(),-row*r)
            lin=float(abs(J@dx+r).max())
            if not np.isfinite(dx).all() or lin>max(1e-7,1e-5*err):raise RuntimeError('mixed linear residual '+str(lin))
            for k in range(14):
                trial=x+2.**(-k)*dx;X=(self.P@trial[:-1]).reshape(8,self.n)
                if X[[3,7]].min()>1e-7 and X[[3,7]].max()<.999:
                    rt=self.residual(trial,reference,slope,bp,False)
                    if np.isfinite(rt).all() and abs(rt).max()<err*(1-1e-4*2.**(-k)):x=trial;break
            else:raise RuntimeError('mixed line search')
        raise RuntimeError('mixed iteration limit')
