import numpy as np

def func_min_3(X_CDF,CDF_inv,V):
    # Funcion a minimizar
    A = V[0]
    #Sigmag2 = V[1]
    r = V[2]
    
    #L0 = 1/Sigmag2
    L0 = V[1]
    L1 = L0/(1 + 1/(A*r))
    L2 = L0/(1 + 2/(A*r))
    #L3 = L0/(1 + 3/(A*r))
    #L4 = L0/(1 + 4/(A*r))
    #L5 = L0/(1 + 5/(A*r))
    #L6 = L0/(1 + 6/(A*r))
    #L7 = L0/(1 + 7/(A*r))
    #L8 = L0/(1 + 8/(A*r))
    #L9 = L0/(1 + 9/(A*r))
    #L10 = L0/(1 + 10/(A*r))

    x = X_CDF/10
    C = np.exp(-A)
    y = np.sum((CDF_inv - C*np.exp(-L0*(10**x)) - A*C*np.exp(-L1*(10**x)) - (A**2/2)*C*np.exp(-L2*(10**x)))**2)

    return y
    
