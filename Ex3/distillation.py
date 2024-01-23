import matplotlib.pyplot as plt
import numpy as np

def antoineEthanolWater(T):
    # AntoineEthanolWater calculates the 
    #   patial pressure for ethanol and water respectivly.
    #   ZacchiBioFuel: Per Sassner
    Cwater   = [73.649, -7258.2, 0, 0, -7.3037, 4.1653E-06, 2]
    Cethanol = [73.304, -7122.3, 0, 0, -7.1424, 2.8853E-06, 2]

    C=np.vstack((Cethanol, Cwater)).T
    lnp0 = np.zeros((len(T),2))
    for i in range(2):
        lnp0[:,i]=C[0,i] + C[1,i]/(T+C[2,i]) + C[3,i]*T + C[4,i]*np.log(T) + C[5,i]*T**C[6,i]
    p01=np.exp(lnp0[:,0])
    p02=np.exp(lnp0[:,1])
    return p01, p02

def wilsonActFact(x,T):
    # WilssonActFact calculates the 
    #   activity factors for ethanol/water system.
    #   ZacchiBioFuel: Per Sassner
    a12,a21,b12,b21 = -2.5035,-0.0503,346.1512,-69.6372

    # Ethanol
    x1=x

    # Water
    x2 = 1-x1

    A12 = np.exp(a12+b12/T)
    A21 = np.exp(a21+b21/T)

    gamma1 = np.exp(1 - np.log(x1+A12*x2) - (x1/(x1+A12*x2) + A21*x2/(A21*x1+x2)))
    gamma2 = np.exp(1 - np.log(x2+A21*x1) - (A12*x1/(x1+A12*x2) + x2/(A21*x1+x2)))
    return gamma1,gamma2


def enthalpy(state,z,T):
    # Calculates the enthalpy [kJ/mol]
    # for a mixture of Ethanol/Water.
    # Inputs:
    #  state: string 'liq'/'vap'
    #  z: ethanol composition in liquid/vapour, x or y
    #  T: temperature [K]
    # Examples:
    # h in liqiud phase
    #    h = enthalpy('liq',x,T)
    # H in vapour phase
    #    H = enthalpy('vap',y,T)
    if state=='vap':
        Het = enthalpyEthanolWater('ethanol','vap',T-273.15)
        Hw = enthalpyEthanolWater('water','vap',T-273.15)
        h = z*Het+(1-z)*Hw  
    elif state=='liq':
        het = enthalpyEthanolWater('ethanol','liq',T-273.15)
        hw = enthalpyEthanolWater('water','liq',T-273.15)
        h = z*het+(1-z)*hw
    else:
        raise Exception(f"state must be 'liq' or 'vap', not '{state}'")
    
    return h

def enthalpyEthanolWater(comp,state,T):
    # Calculates the enthalpy [kJ/kmol]
    # for Ethanol or Water in liquid or vapour phase.
    # Examples:
    # h for water in liqiud phase
    #    h = enthalpyEthanolWater('water','liq',T)
    # H for ethanol in vapour phase
    #    H = enthalpyEthanolWater('ethanol','vap',T)
    # Reference:
    #   Elliot, J., Lira, C. (2001), 
    #   Introductory Chemical Engineering
    #   Thermodynamics, pp. 631-634
    R = 8.3145
    if comp=='water':
        Tb = 100
        dHvap = 2257*18.02      #kJ/mol
        # polynomial for water in liquid phase
        al = 8.712
        bl = 1.25e-3
        cl = 1.8e-7
        # polynomial for water in vapour phase
        a = 32.24
        b = 1.924e-3
        c = 1.055e-5
        d = -3.596e-9
    elif comp=='ethanol':
        Tb = 78.3
        dHvap = 43e3      #kJ/mol
        # polynomial for ethanol in liquid phase
        al = 33.866
        bl=-1.7260e-1
        cl = 3.4917e-4
        # polynomial for ethanol in vapour phase
        a = 9.014
        b = 2.141e-1
        c = -8.39e-5
        d = 1.373e-9

    Hover = np.zeros(T.shape)
    Hover[T>Tb] = 1
    H = (al*T+bl*T**2/2+cl*T**3/3)*R +\
            ((a*T+b*T**2/2+c*T**3/3+d*T**4/4)-\
            (a*Tb+b*Tb**2/2+c*Tb**3/3+d*Tb**4/4))*Hover  #kJ/mol
    if state=='vap':
        H = H + dHvap
    else:
        pass # do nothing
    return H


def francisWC(M, M0, k):
    # Francis weir correlation
    # Liquid mole flow rate as a function of
    #   mole holdup on tray.
    # Parameters:
    #   M0=basic mole content on tray
    #   k=flow/content-coefficient
    # Example:
    #   L = francisWC(M, 20, 50)
    L = k*(M-M0)**1.5
    return L




