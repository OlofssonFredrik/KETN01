import numpy as np
from scipy.optimize import fsolve, root

class Sol():
    pass

def solve_ivp_mass( f, tspan, y0, n=200, mass=None, debug=False, method='assimulo'):
    '''
    Solves a differential algebraic equation system using backward euler method (1st order) with support for mass matrix
        mass*dydt = f(t,y)
    This implementation uses fsolve to solve the implicit equation system. It uses a fixed step length defined by n. 
    A too low value leads to instability.
        
    f - the function to be solved
    tspan - A two element list defining the start and end time of the integration
    y0 - initial values
    n - number of time steps to solve. Too low value will lead to instability. 
    mass - mass matrix (squared with the same number of rows and columns as the number of states in y)
    method - choose if you want to use the IDA solver from 'assimulo' or an euler method. Assimulo can be installed with 
        "conda install -c conda-forge assimulo"
    '''
    if ( np.ndim ( y0 ) == 0 ):
        m = 1
    else:
        m = len ( y0 )

    if mass is None:
        mass = np.eye(len(y0))
    
    y0 = np.array(y0)

    # test 
    f0 = f(0,y0)
    if not isinstance(f0, np.ndarray):
        raise Exception('Derivative must be a np.array')
    
    dt = ( tspan[1] - tspan[0] ) / float ( n )
    tfact = 1.0

    
    # solve initial values
    aeIdx = np.diag(mass)<0.5

    if np.sum(aeIdx)<m:
        qinit = y0[aeIdx]
        to = 0.0
        yo = y0
        tfact0 = 1e-10
        tp = tfact0
        yp = y0
        sol = root ( lambda q: dae_init_residual(q, aeIdx, yp, f, to, yo, tp, mass), qinit, method='hybr')
        
        if debug:
            print(f'nfev: {sol["nfev"]}')
        if not sol['success']:
            print('Warning: Failed to initialize dae')
        qopt = sol['x']
        y0[aeIdx] = qopt
    
    if method=='euler':
        t = np.zeros ( n + 1 )
        y = np.zeros ( [1, m ] )

        to = tspan[0]
        yo = y0
        t = np.array(to)
        y[0,:] = y0
        nsuccess = 0
        while to<tspan[1]:    

            sol = {'success':False}
            while not sol['success']:
                tp = to + dt*tfact
                yp = yo + dt*tfact * f ( to, yo )
                
                sol = root ( backward_euler_residual, yo, args = ( f, to, yo, tp, mass ), method='hybr')
                niter = sol["nfev"]/m
                if debug:
                    print(f'nfev: {sol["nfev"]}, niter: {niter:.1f}')
                yp = sol['x']
                if not sol['success'] or niter>8:
                    nsuccess = 0
                    tfact = tfact/1.5
                    if debug:
                        print(f'tfact reduced to {tfact}')
            nsuccess += 1
            if nsuccess>20:
                if tfact<0.99:
                    tfact = tfact*1.5
                    if debug:
                        print(f'tfact increased to {tfact}')

            if tfact<1e-10:
                raise Exception(f'This dae could not be solved at time {to} (current time step: {dt*tfact})')
            t   = np.hstack((t,tp))
            y   = np.vstack((y,yp))
            if debug:
                print(to)
            to = tp
            yo = yp
        sol = Sol()
        sol.t = t
        sol.y = y.T
    elif method=='assimulo':
        from assimulo.solvers import IDA
        from assimulo.problem import Implicit_Problem

        def residual(t,y,yd, f, mass):
            res = mass@yd-f(t,y)
            return res

        t0 = tspan[0]
        yd0 = f(t0,y0)
        model = Implicit_Problem(lambda t,y,yd: residual(t,y,yd,f,mass), y0, yd0, t0)
        
        sim = IDA(model)
        sim.simulate(tspan[1],n)

        sol = Sol()
        sol.t = np.array(sim.t_sol)
        sol.y = np.array(sim.y_sol).T
        
    return sol



def dae_init_residual(q, aeIdx, yp, f, to, yo, tp, mass):
    yp[aeIdx] = q
    value = backward_euler_residual ( yp, f, to, yo, tp, mass )
    
    resvalue = value[aeIdx]
    return resvalue

def backward_euler_residual ( yp, f, to, yo, tp, mass ):

  value = mass.dot(yp-yo) - ( tp - to ) * f ( tp, yp )

  return value