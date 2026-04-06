import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA
from Estimadores import *
import time as tm

mpl.rc('font',family = 'Times New Roman')

#----------------------
#N = 1000
#----------------------
r = 0.05
A = 0.05
Sigma_G_sq = 0.001
#----------------------

veces = 100
N       = np.array([500,1000,2000,3000])
N_Kane  = np.array([500,1000,2000,3000])
N_Zabin = np.array([10000,15000])
n       = np.size(N)


# Inicializacion del vector A_ini
vec_A_ini       = np.zeros((veces,n))
vec_r_ini       = np.zeros((veces,n))
vec_Sigmag2_ini = np.zeros((veces,n))
time_ini        = np.zeros((veces,n))

# Inicializacion del vector A_ini
A_est_Luc       = np.zeros((veces,n))
Sigmag2_est_Luc = np.zeros((veces,n))
r_est_Luc       = np.zeros((veces,n))
time_Luc        = np.zeros((veces,n))

# Inicializacion de vectores de estimaciones estimador Kanemoto
A_est_Kane      = np.zeros((veces,n))
K_est_Kane      = np.zeros((veces,n))
r_est_Kane      = np.zeros((veces,n))
time_Kane       = np.zeros((veces,n))

# Inicializacion de vectores de estimaciones estimador Zabin
A_est_Zabin = np.zeros((veces,n))
K_est_Zabin = np.zeros((veces,n))
time_Zabin  = np.zeros((veces,n))

# Inicializacion de vectores de estimaciones estimador EM
A_est_EM = np.zeros((veces,n))
K_est_EM = np.zeros((veces,n))
time_EM  = np.zeros((veces,n))

for j in range(len(N)):
    print (j)
    for i in range(veces):
        """
        env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N[j],Sigma_G_sq)
        
        # Estimador inicial
        t0                      = tm.time()
        A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)
        time_ini[i,j]           = tm.time() - t0
        vec_A_ini[i,j]          = A_ini
        vec_Sigmag2_ini[i,j]    = Sigmag2_ini
        vec_r_ini[i,j]          = r_ini
        
        # Estimador mio
        t0                      = tm.time()
        A_est_Luc[i,j],Sigmag2_est_Luc[i,j],r_est_Luc[i,j],Numiter_Luc = Est_Param_ClassA_CDF(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],100,0.0001)
        time_Luc[i,j]           = tm.time() - t0
        """
        """
        # Estimador EM
        t0                  = tm.time()
        A_est_EM[i,j],NumIter = RFI_EMParamA(N[j], 10, env_data_Norm,A_ini,A_ini*r_ini,9)
        A_errado,K_est_EM[i,j],NumIter = RFI_EMTwoParamEst(N[j], 10, env_data_Norm, A_ini, A_ini*r_ini)
        time_EM[i,j]                 = tm.time() - t0
        
        # Estimador Zabin
        env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N_Zabin[j],Sigma_G_sq)
        t0                            = tm.time()
        A_est_Zabin[i,j],K_est_Zabin[i,j] = Est_Momentos_Zabin_Poor(env_data_Norm)
        time_Zabin[i,j]               = tm.time() - t0
        """
        # Estimador Kanemoto
        env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N_Kane[j],Sigma_G_sq)
        t0                      = tm.time()
        A_est_Kane[i,j],r_est_Kane[i,j],K_est_Kane[i,j] = Est_Momentos_Kanemoto(env_data_Norm)
        time_Kane[i,j]          = tm.time() - t0
        
        

# ----------- Cálculo del tiempo promedio -------------------
#time_prom_ini   = np.sum(time_ini,axis=0)/veces
#time_prom_Luc   = np.sum(time_Luc,axis=0)/veces
time_prom_Kane  = np.sum(time_Kane,axis=0)/veces
#time_prom_Zabin = np.sum(time_Zabin,axis=0)/veces
#time_prom_EM    = np.sum(time_EM,axis=0)/veces

#mse_est_simple  = np.sum((vec_A_ini - A)**2,axis=0)/veces
#mse_est_Luc     = np.sum((A_est_Luc - A)**2,axis=0)/veces
mse_est_Kane    = np.sum((A_est_Kane - A)**2,axis=0)/veces
#mse_est_Zabin   = np.sum((A_est_Zabin - A)**2,axis=0)/veces
#mse_est_EM      = np.sum((A_est_EM - A)**2,axis=0)/veces

#mse_est_simple_r = np.sum((vec_r_ini - r)**2,axis=0)/veces
#mse_est_Luc_r    = np.sum((r_est_Luc - r)**2,axis=0)/veces
mse_est_Kane_r   = np.sum((r_est_Kane - r)**2,axis=0)/veces
#mse_est_Zabin_r  = np.sum((K_est_Zabin/A_est_Zabin - r)**2,axis=0)/veces
#mse_est_EM_r     = np.sum((K_est_EM/A_est_EM - r)**2,axis=0)/veces

# ----------- Cálculo del error cuadrático medio en la est. de r -------------------
#mse_est_simple_sigma = np.sum((vec_Sigmag2_ini - Sigma_G_sq)**2,axis=0)/veces
#mse_est_Luc_sigma    = np.sum((Sigmag2_est_Luc - Sigma_G_sq)**2,axis=0)/veces

#print ('Tiempo promedio Est. Simple:',time_prom_ini)
#print ('Tiempo promedio Est. Lucas:',time_prom_Luc)
print ('Tiempo promedio Est. Kanemoto:',time_prom_Kane)
#print ('Tiempo promedio Est. Zabin-Poor:',time_prom_Zabin)
#print ('Tiempo promedio Est. EM:',time_prom_EM)

print ('--------------------------------------')
#print ('MSE Est. Simple A:',mse_est_simple)
#print ('MSE Est. Lucas A:',mse_est_Luc)
print ('MSE Est. Kanemoto A:',mse_est_Kane)
#print ('MSE Est. Zabin-Poor A:',mse_est_Zabin)
#print ('MSE Est. EM:',mse_est_EM)

print ('--------------------------------------')
#print ('MSE Est. Simple r:',mse_est_simple_r)
#print ('MSE Est. Lucas r:',mse_est_Luc_r)
print ('MSE Est. Kanemoto r:',mse_est_Kane_r)
#print ('MSE Est. Zabin-Poor r:',mse_est_Zabin_r)
#print ('MSE Est. EM r:',mse_est_EM_r)
print ('--------------------------------------')
#print ('MSE Est. Simple sigma:',mse_est_simple_sigma)
#print ('MSE Est. Lucas sigma:',mse_est_Luc_sigma)