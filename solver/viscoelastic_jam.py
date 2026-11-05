"""Fully coupled liquid conformation on the existing two-phase JAM map."""
import numpy as np
import sympy as sy
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve

from mixed_pressure import Mixed


def polymer_kernels():
    labels = ['w','u','h','azz','arr','att','azr']
    fields = [sy.symbols(f'{name} {name}_z {name}_y {name}_t') for name in labels]
    w,u,h,azz,arr,att,azr = fields
    slope, y, modulus, inv_de, inv_l2 = sy.symbols('slope y G inv_de inv_l2')
    radius = h[0]**(1-y)
    dr_dy = -radius*sy.log(h[0])
    z_map = -(1-y)*h[1]/(-h[0]*sy.log(h[0]))
    t_map = -(1-y)*h[3]/(-h[0]*sy.log(h[0]))
    def dz(q):
        return q[1]+z_map*q[2]
    def dr(q):
        return q[2]/dr_dy

    conformation = sy.Matrix([[azz[0],azr[0],0],[azr[0],arr[0],0],[0,0,att[0]]])
    grad = sy.Matrix([[dz(w),dr(w),0],[dz(u),dr(u),0],[0,0,u[0]/radius]])
    peterlin = (1-3*inv_l2)/(1-inv_l2*sy.trace(conformation))
    stress = modulus*(peterlin*conformation-sy.eye(3))
    advected = []
    for q in [azz,arr,att,azr]:
        advected.append(q[3]+t_map*q[2]+w[0]*dz(q)+u[0]*dr(q))
    convected = grad*conformation+conformation*grad.T
    material = [advected[k]-convected[i,j]+inv_de*(peterlin*conformation[i,j]-int(i==j))
                for k,(i,j) in enumerate([(0,0),(1,1),(2,2),(0,1)])]

    def stress_derivative(value, direction):
        return sum(sy.diff(value,q[0])*direction(q) for q in [azz,arr,att,azr])

    axial = -(stress_derivative(stress[0,0],dz)+stress_derivative(stress[0,1],dr)+stress[0,1]/radius)
    radial = -(stress_derivative(stress[0,1],dz)+stress_derivative(stress[1,1],dr)
               +(stress[1,1]-stress[2,2])/radius)
    normal = sy.Matrix([-slope,1,0])
    tangent = sy.Matrix([1,slope,0])
    traction = [(v.T*stress*normal)[0]/(1+slope**2) for v in [normal,tangent]]
    arguments = [symbol for field in fields for symbol in field]+[slope,y,modulus,inv_de,inv_l2]
    # Original full-field variable order followed by the four conformation fields.
    variables = [(value, field, derivative) for field,q in zip([4,5,7,8,9,10,11],fields)
                 for derivative,value in zip([0,1,3,6],q)]
    variables.append((slope,3,1))
    kernels = {}
    for name, equations in [('bulk',[axial,radial]),('material',material),('interface',traction)]:
        entries = []
        for row,equation in enumerate(equations):
            for value,field,derivative in variables:
                differential = sy.diff(equation,value)
                if differential != 0:
                    entries.append((row,field,derivative,differential))
        kernels[name] = (sy.lambdify(arguments,equations+[item[3] for item in entries],'numpy',cse=True),
                         len(equations),[item[:3] for item in entries])
    return kernels


class Viscoelastic(Mixed):
    def __init__(self,ns,ny,parameters):
        self.params = parameters
        if parameters['rho_g']/parameters['rho_l'] != .001:
            raise ValueError('The inherited gas equations fix the density ratio at 0.001')
        viscosity_unit = np.sqrt(parameters['rho_l']*parameters['sigma']*parameters['R_m'])
        self.time_unit = np.sqrt(parameters['rho_l']*parameters['R_m']**3/parameters['sigma'])
        super().__init__(ns,ny,Oh=(parameters['mu']*(1-parameters['polymer_fraction']))/viscosity_unit,
                         mug=parameters['mu_g']/viscosity_unit,L=parameters['L_over_R'],cluster=.95)
        self.inv_de = self.time_unit/parameters['lambda_s']
        self.modulus = parameters['mu']*parameters['polymer_fraction']*parameters['R_m']/(
            parameters['lambda_s']*parameters['sigma'])
        self.inv_l2 = 0. if parameters['L2'] is None else 1./parameters['L2']
        self.nh = self.P.shape[1]
        self.full_projection = sp.block_diag([self.P,sp.eye(4*self.n)],format='csr')
        self.full_keep = np.r_[self.keep,np.arange(8*self.n,12*self.n)]
        self.kernels = polymer_kernels()
        all_nodes = np.arange(self.n)
        bulk = all_nodes[ns:-ns]
        interface = all_nodes[:ns]
        self.y_nodes = np.repeat(self.y,self.ns)
        self.poly_groups = [
            ('bulk',bulk,bulk,[4,5]),
            ('material',all_nodes,all_nodes,[8,9,10,11]),
            ('interface',interface,all_nodes[-ns:],[0,1]),
        ]
        self.poly_operators = {}
        for name,liquid,gas,_ in self.poly_groups:
            for field in [3,4,5,7,8,9,10,11]:
                for derivative in [0,1,3,6]:
                    indices = gas if field==3 else liquid
                    self.poly_operators[name,field,derivative] = self.ops[derivative][indices,:].tocoo()

    def initial_state(self,amplitude=.1):
        hydro = super().pack(super().initial(amplitude))
        equilibrium = np.tile([1.,1.,1.,0.],(self.n,1)).T.ravel()
        return np.r_[hydro[:-1],equilibrium,hydro[-1]]

    def fields(self,state):
        return (self.full_projection@state[:-1]).reshape(12,self.n)

    def admissible(self,fields):
        zz,rr,tt,zr = fields[8:]
        return (np.isfinite(fields).all() and fields[[3,7]].min()>1e-7 and fields[[3,7]].max()<.999
                and min(zz.min(),rr.min(),tt.min())>0 and (zz*rr-zr*zr).min()>0
                and (self.inv_l2==0 or (self.inv_l2*(zz+rr+tt)).max()<1))

    def coupled_residual(self,state,reference,slope,bp,jac=True):
        fields = self.fields(state)
        base = super().residual(np.r_[state[:self.nh],state[-1]],reference[:8],slope[:8],bp,jac)
        base_r,base_j = base if jac else (base,None)
        time_derivative = bp*(fields-reference)+slope
        values = {field:[fields[field], self.ops[1]@fields[field], self.ops[3]@fields[field],
                         time_derivative[field]] for field in [4,5,7,8,9,10,11]}
        gas_slope = self.ops[1]@fields[3]
        extra = np.zeros(12*self.n)
        rows,columns,data = [],[],[]
        for name,liquid,gas,row_fields in self.poly_groups:
            fn,num_equations,entries = self.kernels[name]
            arguments = [v[liquid] for vv in values.values() for v in vv]
            arguments += [gas_slope[gas],self.y_nodes[liquid],self.modulus,self.inv_de,self.inv_l2]
            output = fn(*arguments)
            for i,field in enumerate(row_fields):
                extra[field*self.n+gas] = output[i]
            if jac:
                for value,(i,field,derivative) in zip(output[num_equations:],entries):
                    operator = self.poly_operators[name,field,derivative]
                    rows.append(row_fields[i]*self.n+gas[operator.row])
                    columns.append(field*self.n+operator.col)
                    data.append(operator.data*np.broadcast_to(value,(len(gas),))[operator.row]
                                *(bp if derivative==6 else 1.))
        residual = np.r_[base_r[:-1],np.zeros(4*self.n),base_r[-1]]
        residual[:-1] += extra[self.full_keep]
        if not jac:
            return residual
        extra_j = sp.coo_matrix((np.concatenate(data),(np.concatenate(rows),np.concatenate(columns))),
                                 shape=(12*self.n,12*self.n)).tocsr()
        extra_j = extra_j[self.full_keep]@self.full_projection
        base_j = sp.bmat([[base_j[:-1,:-1],None,base_j[:-1,-1:]],
                          [None,sp.csc_matrix((4*self.n,4*self.n)),None],
                          [base_j[-1:,:-1],None,base_j[-1:,-1:]]],format='csc')
        return residual,base_j+sp.block_diag([extra_j,sp.csc_matrix((1,1))],format='csc')

    def advance(self,old,older,dt,dtold,first=False):
        reference = self.fields(old)
        bp = 1/dt if first else (2*dt+dtold)/(dt*(dt+dtold))
        bmm = 0. if first else dt/(dtold*(dt+dtold))
        slope = bmm*(self.fields(older)-reference)
        state = old.copy()
        for iteration in range(20):
            residual,jacobian = self.coupled_residual(state,reference,slope,bp)
            error = float(abs(residual).max())
            if error<1e-8:
                return state,iteration,error
            if not np.isfinite(error):
                raise RuntimeError('nonfinite residual')
            row_scale = 1/np.maximum(abs(jacobian).max(axis=1).toarray().ravel(),1e-30)
            scaled_jac = sp.diags(row_scale)@jacobian
            col_scale = 1/np.maximum(abs(scaled_jac).max(axis=0).toarray().ravel(),1e-30)
            step = col_scale*spsolve((scaled_jac@sp.diags(col_scale)).tocsc(),-row_scale*residual)
            if not np.isfinite(step).all() or abs(jacobian@step+residual).max()>max(1e-7,1e-5*error):
                raise RuntimeError('linear residual')
            for k in range(14):
                fraction = 2.**-k
                trial = state+fraction*step
                if self.admissible(self.fields(trial)):
                    trial_residual = self.coupled_residual(trial,reference,slope,bp,False)
                    if abs(trial_residual).max()<error*(1-1e-4*fraction):
                        state = trial
                        break
            else:
                raise RuntimeError('coupled line search')
        raise RuntimeError('coupled Newton iteration limit')

    def check_coupling(self):
        # A locally homogeneous steady shear has an exact Oldroyd-B state.
        h, y, shear, inv_de = .8, .4, .3, .4
        dr_dy = -h**(1-y)*np.log(h)
        shear_args = [0.,0.,shear*dr_dy,0., 0.,0.,0.,0., h,0.,0.,0.,
                      1+2*(shear/inv_de)**2,0.,0.,0., 1.,0.,0.,0.,
                      1.,0.,0.,0., shear/inv_de,0.,0.,0.,
                      0.,y,.1,inv_de,0.]
        shear_residual = abs(np.asarray(self.kernels['material'][0](*shear_args)[:4])).max()
        assert shear_residual<1e-12
        equilibrium = self.initial_state(0.)
        reference = self.fields(equilibrium)
        zero = np.zeros_like(reference)
        stationary = abs(self.coupled_residual(equilibrium,reference,zero,100.,False)).max()
        state = self.initial_state()
        rng = np.random.RandomState(743)
        state[self.nh:-1] += .015*rng.standard_normal(4*self.n)
        direction = rng.standard_normal(len(state))
        direction /= np.linalg.norm(direction)
        ref = self.fields(state)
        residual,jacobian = self.coupled_residual(state,ref,zero,100.)
        delta = 1e-6
        derivative = (self.coupled_residual(state+delta*direction,ref,zero,100.,False)
                      -self.coupled_residual(state-delta*direction,ref,zero,100.,False))/(2*delta)
        relative = np.linalg.norm(derivative-jacobian@direction)/np.linalg.norm(derivative)
        assert stationary<1e-8 and relative<1e-5 and np.isfinite(residual).all()
        return dict(stationary_residual=float(stationary),jacobian_direction_error=float(relative),
                    analytic_steady_shear_residual=float(shear_residual))
