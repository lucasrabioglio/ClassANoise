import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA
from Estimadores import *

mpl.rc('font',family = 'Times New Roman')

#----------------------
#N = 1000
#----------------------
r = 0.01
A = 0.05
Sigma_G_sq = 0.001
#----------------------

veces = 50
N = np.array([500,1000,2000,5000])
n = np.size(N)

# Inicializacion del vector A_ini
vec_A_ini       = np.zeros((veces,n))
vec_r_ini       = np.zeros((veces,n))
vec_Sigmag2_ini = np.zeros((veces,n))

# Inicializacion del vector A_ini
A_est_Luc       = np.zeros((veces,n))
Sigmag2_est_Luc = np.zeros((veces,n))
r_est_Luc       = np.zeros((veces,n))

# Inicializacion de vectores de estimaciones estimador Kanemoto
A_est_Kane      = np.zeros((veces,n))
K_est_Kane      = np.zeros((veces,n))
r_est_Kane      = np.zeros((veces,n))

# Inicializacion de vectores de estimaciones estimador Zabin
A_est_Zabin = np.zeros((veces,n))
K_est_Zabin = np.zeros((veces,n))

# Inicializacion de vectores de estimaciones estimador EM
A_est_EM = np.zeros((veces,n))
K_est_EM = np.zeros((veces,n))

for j in range(len(N)):
    print (j)
    for i in range(veces):
        print (i)
        env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N[j],Sigma_G_sq)

        # Estimador inicial
        A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)
        vec_A_ini[i,j]          = A_ini
        vec_Sigmag2_ini[i,j]    = Sigmag2_ini
        vec_r_ini[i,j]          = r_ini

        # Estimador mio
        A_est_Luc[i,j],Sigmag2_est_Luc[i,j],r_est_Luc[i,j],Numiter_Luc = Est_Param_ClassA_CDF(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],100,0.0001)

        # Estimador Kanemoto
        A_est_Kane[i,j],r_est_Kane[i,j],K_est_Kane[i,j] = Est_Momentos_Kanemoto(env_data_Norm)

        # Estimador Zabin
        A_est_Zabin[i,j],K_est_Zabin[i,j] = Est_Momentos_Zabin_Poor(env_data_Norm)

        # Estimador EM
        A_est_EM[i,j],NumIter = RFI_EMParamA(N[j], 10, env_data_Norm,A_ini,A_ini*r_ini,9)
        A_errado,K_est_EM[i,j],NumIter = RFI_EMTwoParamEst(N[j], 10, env_data_Norm, A_ini, A_ini*r_ini)

# ----------- Cálculo del error cuadrático medio en la est. de A -------------------
mse_est_simple  = np.sum((vec_A_ini - A)**2,axis=0)/veces
mse_est_Luc     = np.sum((A_est_Luc - A)**2,axis=0)/veces
mse_est_Kane    = np.sum((A_est_Kane - A)**2,axis=0)/veces
mse_est_Zabin   = np.sum((A_est_Zabin - A)**2,axis=0)/veces
mse_est_EM      = np.sum((A_est_EM - A)**2,axis=0)/veces

# ----------- Cálculo del error cuadrático medio en la est. de r -------------------
mse_est_simple_r = np.sum((vec_r_ini - r)**2,axis=0)/veces
mse_est_Luc_r    = np.sum((r_est_Luc - r)**2,axis=0)/veces
mse_est_Kane_r   = np.sum((r_est_Kane - r)**2,axis=0)/veces
mse_est_Zabin_r  = np.sum((K_est_Zabin/A_est_Zabin - r)**2,axis=0)/veces
mse_est_EM_r     = np.sum((K_est_EM/A_est_EM - r)**2,axis=0)/veces

# ----------- Cálculo del error cuadrático medio en la est. de r -------------------
mse_est_simple_sigma = np.sum((vec_Sigmag2_ini - Sigma_G_sq)**2,axis=0)/veces
mse_est_Luc_sigma    = np.sum((Sigmag2_est_Luc - Sigma_G_sq)**2,axis=0)/veces


print ('MSE Est. Simple A:',mse_est_simple)
print ('MSE Est. Lucas A:',mse_est_Luc)
print ('MSE Est. Kanemoto A:',mse_est_Kane)
print ('MSE Est. Zabin-Poor A:',mse_est_Zabin)
print ('MSE Est. EM A:',mse_est_EM)

print ('MSE Est. Simple r:',mse_est_simple_r)
print ('MSE Est. Lucas r:',mse_est_Luc_r)
print ('MSE Est. Kanemoto r:',mse_est_Kane_r)
print ('MSE Est. Zabin-Poor r:',mse_est_Zabin_r)
print ('MSE Est. EM r:',mse_est_EM_r)

print ('MSE Est. Simple Sigma:',mse_est_simple_sigma)
print ('MSE Est. Lucas Sigma:',mse_est_Luc_sigma)

print ('FIN')