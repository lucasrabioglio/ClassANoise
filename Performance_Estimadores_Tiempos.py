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
r = 0.01
A = 0.05
Sigma_G_sq = 0.001
#----------------------

veces = 51
N = np.array([500,1000,2000,5000])
n = np.size(N)


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
        env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N[j],Sigma_G_sq)

        # Estimador inicial
        t0                      = tm.time()
        A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)
        time_ini[i,j]           = tm.time() - t0
        vec_A_ini[i,j]          = A_ini
        vec_Sigmag2_ini[i,j]    = Sigmag2_ini
        vec_r_ini[i,j]          = r_ini

        # Estimador mio
        """
        t0                      = tm.time()
        A_est_Luc[i,j],Sigmag2_est_Luc[i,j],r_est_Luc[i,j],Numiter_Luc = Est_Param_ClassA_CDF(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],100,0.0001)
        time_Luc[i,j]           = tm.time() - t0
        """
        
        # Estimador Kanemoto
        t0                      = tm.time()
        A_est_Kane[i,j],r_est_Kane[i,j],K_est_Kane[i,j] = Est_Momentos_Kanemoto(env_data_Norm)
        time_Kane[i,j]          = tm.time() - t0

        # Estimador Zabin
        t0                            = tm.time()
        A_est_Zabin[i,j],K_est_Zabin[i,j] = Est_Momentos_Zabin_Poor(env_data_Norm)
        time_Zabin[i,j]               = tm.time() - t0

        # Estimador EM
        t0                  = tm.time()
        A_est_EM[i,j],NumIter = RFI_EMParamA(N[j], 10, env_data_Norm,A_ini,A_ini*r_ini,9)
        A_errado,K_est_EM[i,j],NumIter = RFI_EMTwoParamEst(N[j], 10, env_data_Norm, A_ini, A_ini*r_ini)
        time_EM[i,j]                 = tm.time() - t0


# ----------- Cálculo del tiempo promedio -------------------
time_prom_ini   = np.sum(time_ini,axis=0)/veces
time_prom_Luc   = np.sum(time_Luc,axis=0)/veces
time_prom_Kane  = np.sum(time_Kane,axis=0)/veces
time_prom_Zabin = np.sum(time_Zabin,axis=0)/veces
time_prom_EM    = np.sum(time_EM,axis=0)/veces


print ('Tiempo promedio Est. Simple:',time_prom_ini)
print ('Tiempo promedio Est. Lucas:',time_prom_Luc)
print ('Tiempo promedio Est. Kanemoto:',time_prom_Kane)
print ('Tiempo promedio Est. Zabin-Poor:',time_prom_Zabin)
print ('Tiempo promedio Est. EM:',time_prom_EM)
print ('FIN')