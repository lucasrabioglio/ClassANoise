import numpy as np
from math import *
from scipy.special import factorial
import numpy.matlib as np_matlib
from RFI_EMParamK_AFixed import RFI_EMParamK_AFixed
from RFI_EMCalculateObjFunc import RFI_EMCalculateObjFunc
from scipy.io import loadmat

def RFI_EMTwoParamEst(N, M, z, A_init, K_init):
    #-------------------------------------------
    #---- Funcion para estimar los valores de A y K utilizando
    #---- el algoritmo EM ------------------------------------

    # Dummy Initialization
    K_prev = 10
    A_prev = 10
    # Change of variables
    A = A_init
    Kp = K_init
    # Initialization
    NumIter = 0
    # Conversion vector de entrada de 1D a 2D
    #z = z.reshape(1,N)

    while (abs((K_prev - Kp) / K_prev)  + abs((A_prev - A)/A_prev) > 10e-7 and NumIter < 100):   # convergence criterion, incremental error less than 10^-7
        # Niter < 100 used to avoid infinite loop which can occur due to
        # inconsistent input envelope data or in-accuracies in calculations.
        K_prev = Kp
        A_prev = A
        K_possible = []
        A_possible = []

        # Calculate a_ij's as defined in equation (8) in [1]
        a_ij = np.zeros ((N, M))
        j    = np.array(list(range(0,M)))
        pi_j = exp(-A)*(A**j)/factorial(j)
        for i in range(N):
            for j in range(M):
                h_j       = 2*z[i]*(A + Kp)/(j + Kp)*exp(-1*(z[i]**2)*(A + Kp)/(j + Kp))
                a_ij[i,j] = pi_j[j]*h_j
            a_ij[i,:]     = a_ij[i,:]/np.sum(a_ij[i,:])

        beta1 = np.sum((1 + z**2)*a_ij[:,0].T)
        beta2 = np.sum((1 + z**2)*(a_ij[:,1:].dot((1/np.array(list(range(1,M)),dtype=float)).reshape(1,M-1).T)).T)
        beta3 = np.sum((z**2)*a_ij[:,0].T)
        beta4 = np.sum((z**2)*(a_ij[:,1:].dot((((1/np.array(list(range(1,M)),dtype = float))**2).reshape(1,M-1)).T)).T)
        beta5 = np.sum((z**2)*(a_ij[:,1:].dot((1/np.array(list(range(1,M)))).T)).T)

        alpha = np.sum(a_ij*(np_matlib.repmat(np.array(list(range(0,M))),N,1)))

        c1    = (beta4**2)*(N**2) - beta2*beta4*N*(beta5 + N) + beta4*N*((beta5 + N)**2)
        c2    = (-beta2*beta4*(N**2) - beta2*beta3*beta4*N - beta2*beta4*alpha*N) + (2*beta3*beta4*N - beta1*beta4*N + (beta2**2)*N + 2*beta4*alpha*N)*(beta5 + N) - (beta2*N)*((beta5 + N)**2)
        c3    = (beta4*N**3 + 2*beta3*beta4*N**2 - beta1*beta4*N**2 - beta1*beta4*N**2 + 2*beta4*alpha*N**2) + (beta3**2*beta4*N + beta2**2*beta3*N - beta1*beta3*beta4*N - beta1*beta4*alpha*N + 2*beta3*beta4*alpha*N + beta4*alpha**2*N) + (-beta2*N**2 - 3*beta2*beta3*N + 2*beta1*beta2*N - beta2*alpha*N)*(beta5 + N) + (N**2 * beta3*N - beta1*N)*(beta5+N)**2
        c4    = (-2*beta2*beta3*N**2 - 2*beta2*beta3**2*N + 2*beta1*beta2*beta3*N - 2*beta2*beta3*alpha*N) + (2*beta3*N**2 - beta1*N**2 - 3*beta1*beta3*N + 2*beta3**2*N + beta1**2*N - beta1*alpha*N + 2*beta3*alpha*N)*(beta5 + N)
        c5    = (beta3*N**3 - 2*beta1*beta3*N**2 + 2*beta3**2*N**2 + 2*beta3*alpha*N**2) + (beta1**2*beta3*N - 2*beta1*beta3**2*N + beta3**2*N - 2*beta1*beta3*alpha*N + 2*beta3**2*alpha*N + beta3*alpha**2*N)

        r = np.real(np.roots(np.array([c1,c2,c3,c4,c5])))
        r_ind = np.where((r > 9.09e-7) & (r < 1.1e-2) & (np.imag(r) == 0) & (np.real(r) > 0))
        k_r = r[r_ind[0]] 

        j_vec = np_matlib.repmat(np.array(list(range(0,M))),N,1)
        z_vec = np_matlib.repmat((z.reshape(1,N)).T,1,M)

        k_r = np.append(k_r,[9.09e-7,1.1e-2])
        zeta = []
        A_corr = []
        if (len(k_r) > 0):
            for i in range(len(k_r)):
                zeta.append(np.sum(np.divide(z_vec**2*a_ij,j_vec + k_r[i])))
            
            A_corr = np.divide(N*k_r + zeta*k_r - alpha - N - np.sqrt((N*k_r + zeta*k_r - alpha - N)**2 + 4*(N + np.array(zeta))*alpha*k_r),-2*(N + np.array(zeta)))
            indA = np.where((A_corr > 9.09e-3) & (A_corr < 1.1))

            k_r = k_r[indA[0]]
            A_corr = A_corr[indA[0]]
        
        K_possible = np.append(K_possible,k_r)
        A_possible = np.append(A_possible,A_corr)

        # Estimate A at boundary values of K
        K1 = RFI_EMParamK_AFixed(N,M,z,9.09e-3,Kp)   # Estimate A at K = 9.09*10^-3
        K2 = RFI_EMParamK_AFixed(N,M,z,1.1,Kp)      # Estimate A at K = 1.1

        if (len(K1) > 0):
            K_possible = np.append(K_possible,K1)
            A_possible = np.append(A_possible,9.09e-3*np.ones(K1.size))

        if (len(K2) > 0):
            K_possible = np.append(K_possible,K2)
            A_possible = np.append(A_possible,1.1*np.ones(K2.size))
        
        j_vec = np_matlib.repmat(np.array(list(range(0,M))),N,1)
        z_vec = np_matlib.repmat((z.reshape(1,N)).T,1,M)

        alpha = np.sum(a_ij*j_vec)

        Obj_func = []
        for i in range(len(K_possible)):
            A_i = A_possible[i]
            K_i = K_possible[i]
            Obj_func.append(RFI_EMCalculateObjFunc(N,M,z,A_i,K_i))
        
        obj_max = np.amax(Obj_func)
        ind     = np.where (Obj_func == obj_max)[0]

        Kp = K_possible[ind]
        A  = A_possible[ind]
        NumIter = NumIter + 1

        A_est = A[0]
        K_est = Kp[0]

        A  = A_est
        Kp = K_est
        #print ('r')

    return A_est,K_est,NumIter

##------------------ Script para correr la funcion de arriba ---------------
"""
N = 5
M = 2
z = np.array([0.1,0.001,0.5,0.2,0.00001])
A_inicial = 0.1
K_inicial = 0.5

matlab = loadmat("/home/lucas/Escritorio/matlab.mat")
z_matlab = matlab["z"]
z_matlab = z_matlab[0]

Ty = RFI_EMTwoParamEst(100,3,z_matlab,0.1,1)

print (Ty)
##--------------------------------------------------------------------------
"""