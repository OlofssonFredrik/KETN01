# -*- coding: utf-8 -*-
"""
Instructions: How to use the method discretize from another file


Example:
from FVMdisc import FVMdisc1st, FVMdisc2nd, FVMdiscBV

N = 4
L = 1.
h = L/N
Dax = 1e-5
v = 0.1

Atype1 = '2pb'
Atype2 = '3pc'

BVtype = [0, 1]             # [Dirichlet, general von Neumann] There are other possible combinations.
bv = [[1, -1], [0, 0]]

A1,A1f = FVMdisc1st(N, h, Atype1)
A2,A2f = FVMdisc2nd(N, h, Atype2)
B1,B0 = FVMdiscBV(N, h, BVtype, bv)


              
"""


import numpy as np
import matplotlib.pylab as plt
#from scipy import sparse
import scipy.sparse as sps



def FVMdisc1st(N, h, Atype1):
    # [A,Af]=FVMdisc1st(N,h,Atype)
    # [A,Af]=FVMdisc1st(N,h,Atype,spflag)
    # FVMdisc1st is a Finite Volume Method domain discritisation
    #   of 1st order derivative
    # input:
    #   N: number of grid points
    #   h: grid size
    #   Atype: type of 1st order derivative discritisation
    #    Atype='2pb'  => 2-point backward approximation
    #    Atype='2pf'  => 2-point forward approximation
    #    Atype='3pc'  => 3-point central approximation
    #    Atype='3pb  => 3-point backward approximation
    #    Atype='4p2b  => 4-point 2-step backward approximation
    #    Atype='4p2Q  => 4-point 2-step backward approximation, QUICK
    #    Atype='5pc  => 5-point central approximation
    #    Atype='5p3b  => 5-point 3-step backward approximation 
    A = []
    Af = []
    
    if Atype1 == '2pb': # 2-point backward approximation
       A = 1/h * (np.eye(N) - np.eye(N, k=-1))
       
       z = np.zeros((N, 2))
       z[0, 0] = -1
       Af = 1/h * z
    
    elif Atype1 == '2pf': # 2-point forward approximation
        A = 1/h * (np.eye(N, k=1)-np.eye(N))
        
        z = np.zeros((N, 2))
        z[N-1, 1] = 1
        Af = 1/h * z
    
    elif Atype1 == '3pc': # 3-point central approximation
        A = 1/(2*h) * (np.eye(N, k=1) - np.eye(N, k=-1))
        
        z = np.zeros((N, 2))
        z[0, 0] = -1
        z[N-1, 1] = 1
        Af = 1/(2*h) * z
    
    elif Atype1 == '3pb': # 3-point backward approximation
        A = 1/(2*h) * (3*np.eye(N) - 4*np.eye(N, k=-1) + np.eye(N, k=-2))
        z = np.zeros((1, N))
        z[0, 0] = 1./h 
        A[0] = z
        
        z = np.zeros((N, 2))
        z[0, 0] = -2
        z[1, 0] = 1
        Af = 1/(2*h) * z 
    
    elif Atype1 == '4p2b': # 4-point 2-step backward approximation
        A = 1/(24*h) * (8*np.eye(N, k=1) + 12*np.eye(N) - 24*np.eye(N, k=-1) + 4*np.eye(N, k=-2))
        z = np.zeros((1, N))
        z[0, 0] = 1./h 
        A[0] = z
        
        z = np.zeros((N, 2))
        z[0, 0] = -24
        z[1, 0] = 4
        z[N-1, 1] = 8
        Af = 1/(24*h) * z
          
    elif Atype1 == '4p2Q': # 4-point 2-step backward approximation, QUICK
        A = 1/(8*h) * (3*np.eye(N, k=1) + 3*np.eye(N) - 7*np.eye(N, k=-1) + np.eye(N, k=-2))
        z = np.zeros((1, N))
        z[0, 0] = 1/h 
        A[0] = z
    
        z = np.zeros((N, 2))
        z[0, 0] = -8
        z[1, 0] = 1
        z[N-1, 1] = 3
        Af = 1/(8*h) * z
    
    elif Atype1 == '5pc': # 5-point central approximation
        A = 1/(24*h) * (-2*np.eye(N, k=2) + 16*np.eye(N, k=1) - 16*np.eye(N, k=-1) + 2*np.eye(N, k=-2))
        z = np.zeros((1, N))
        z[0, 1] = 1./(2*h)
        A[0] = z
        z = np.zeros((1, N))
        z[0, N-2] = -1./(2*h)
        A[N-1] = z
        
        z = np.zeros((N, 2))
        z[0, 0] = -1./2
        z[1, 0] = 2./24
        z[N-2, 1] = -2./24
        z[N-1, 1] = 1./2
        Af = 1/h * z
    
    elif Atype1 == '5p3b': # 5-point 3-step backward approximation
        A = 1/(24*h) * (6*np.eye(N, k=1) + 20*np.eye(N) - 36*np.eye(N, k=-1) + 12*np.eye(N, k=-2) - 2*np.eye(N, k=-3))
        A[0,0] = 1/(24*h)*24
        A[0,1] = 1/(24*h)*0
        A[1,0] = 1/(24*h)*(-24)
        A[1,1] = 1/(24*h)*12
        A[1,2] = 1/(24*h)*8
        
        
        z = np.zeros((N, 2))        
        z[0, 0] = -24
        z[1, 0] = 4
        z[2, 0] = -2
        z[-1, 1] = 6
        Af = 1./(24*h) * z
    
    else: 
        print("Unknown approximation method! 2-point backward selected")
        A = 1/h * (np.eye(N) - np.eye(N, k=-1))
       
        z = np.zeros((N, 2))
        z[0, 0] = -1
        Af = 1/h * z

    return A, Af


def FVMdisc2nd(N, h, Atype2):
    # [A,Af]=FVMdisc2nd(N,h,Atype)
    # [A,Af]=FVMdisc2nd(N,h,Atype,spflag)
    # FVMdisc2nd is a Finite Volume Method domain discritisation
    #   of 2nd order derivative
    # input:
    #   N: number of grid points (in domain)
    #   h: grid size
    #   Atype: type of 2nd order derivative discritisation
    #     Atype='3pc'  => 3-point central approximation
    #     Atype='5pc'  => 5-point central approximation
    A2 = []
    A2f = []    
    
    if Atype2 == '3pc':
        A2 = 1/(h**2) * (np.eye(N, k=1) - 2*np.eye(N) + np.eye(N, k=-1))
        
        z = np.zeros((N, 2))
        z[0, 0] = 1.
        z[N-1, 1] = 1.
        A2f = 1./(h**2) * z
        
    elif Atype2 == '5pc':
        A2 = 1/(24*h**2) * (-2*np.eye(N, k=2) + 32*np.eye(N, k=1) - 60*np.eye(N) + 32*np.eye(N, k=-1) - 2*np.eye(N, k=-2))
        z = np.zeros((1, N))
        z[0, 0] = -2.
        z[0, 1] = 1.
        A2[0] = 1./(h**2) * z
        z = np.zeros((1, N))
        z[0, N-2] = 1.
        z[0, N-1] = -2.
        A2[N-1] = 1./(h**2) * z
        
        z = np.zeros((N, 2))
        z[0, 0] = 1.
        z[1, 0] = -2./24
        z[N-2, 1] = -2./24
        z[N-1, 1] = 1.
        A2f = 1./(h**2) * z
        
    else:
        print("Unknown approximation method! 3-point selected")
        A2 = 1/(h**2) * (np.eye(N, k=1) - 2*np.eye(N) + np.eye(N, k=-1))
        
        z = np.zeros((N, 2))
        z[0, 0] = 1
        z[N-1, 1] = 1
        A2f = 1/(h**2) * z
    
    return A2, A2f
    

def FVMdiscBV(N, h, BVtype, bv):
    ''' [B1,B0]=FVMdiscBV(N,h,BVtype,bv)
     [B1,B0]=FVMdiscBV(N,h,BVtype,bv)
     FVMdiscBV is Finite Volume Method boundary value discretisation
     input:
       N: number of grid points
       h: grid size
       BVtype: [1x2]-vector indicating type of Boundary Value Model
           [0 0]: left=Dirichlet,    right=Dirichlet
           [0 1]: left=Dirichlet,    right=Flux/Neumann
           [1 0]: left=Flux/Neumann, right=Dirichlet
           [1 1]: left=Flux/Neumann, right=Flux/Neumann
       bv: [2x2]-matrix with values in boundary expressions
           Boundary expressions:
               left:  BVtype[0]*dc/dz = bv[0,0]*c + bv[0,1]
               right: BVtype[1]*dc/dz = bv[1,0]*c + bv[1,1]
           Examples:
           ex. Left: Flux model, bv[0,0]=k/D; bv[0,1]=-k/D*cL;
           ex. Left: Dirichlet, (c=cL) => bv[0,0]=1; bv[0,1]=-cL;
           ex. Right: no Flux: bv[1,0]=0; bv[1,1]=0;
           ex. Right: Dirichlet, (c=cR) => bv[0,0]=1; bv[0,1]=-cR;
    '''

    B1 = np.zeros((2, N))
    B0 = np.zeros(2)
    
    bv = np.array(bv)
    
    #left bv
    den1 = 2*BVtype[0]+h*bv[0,0]
    B1[0,0] = (2*BVtype[0]-h*bv[0,0])/den1
    B0[0] = -2*h*bv[0,1]/den1
    
    # right bv
    den2 = 2*BVtype[1]-h*bv[1,0]
    B1[1,-1] = (2*BVtype[1]+h*bv[1,0])/den2
    B0[1] = 2*h*bv[1,1]/den2
    
    return B1, B0


    
def FVMjpattern(system, N_states):
    '''
    Creates a jacobian pattern to be used by matlabs odesolvers.
    Syntax: S = FVMjpattern(lambda t,y: modelfunction(t,y, params...), numberofstates)
    '''

    # Create a place for S
    S = np.empty((N_states,N_states))

    # Creates a random vector for the sta tes sent into the ode function.
    Y0 = np.random.rand(N_states)

    # Simulates the derivative at the base states.
    dY0 = system(0,Y0)

    # Checking all derivatives to see which state affects it.
    for i in range (N_states):

        # Reset Y
        Y = Y0.copy()

        # Changes the values of Y
        Y[i] += 1.0

        # Runs the ode function with one state changed.
        dY = system(0,Y)

        # Checks which derivatives was affected and saves them in S as ones.
        diff2 = dY!=dY0
        S[:,i] = diff2

    return S
    


if __name__ == "__main__":
    pass