import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA
from Estimadores import *
from matplotlib.gridspec import GridSpec

mpl.rc('font',family = 'Times New Roman')

#----------------------
#N = 1000
#----------------------
r = 1
A = 0.005
Sigma_G_sq = 0.001
#----------------------

#---------------------------------------------------------------------
#---------------------------------------------------------------------
veces = 50
N = np.array([500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
n = np.size(N)

X = np.array([500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Inicializacion del vector A_ini
vec_A_ini       = np.zeros((veces,n))
vec_r_ini       = np.zeros((veces,n))
vec_Sigmag2_ini = np.zeros((veces,n))

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

        # Estimador Kanemoto
        A_est_Kane[i,j],r_est_Kane[i,j],K_est_Kane[i,j] = Est_Momentos_Kanemoto(env_data_Norm)

        # Estimador Zabin
        A_est_Zabin[i,j],K_est_Zabin[i,j] = Est_Momentos_Zabin_Poor(env_data_Norm)

        # Estimador EM
        A_est_EM[i,j],NumIter = RFI_EMParamA(N[j], 10, env_data_Norm,A_ini,A_ini*r_ini,9)
        A_errado,K_est_EM[i,j],NumIter = RFI_EMTwoParamEst(N[j], 10, env_data_Norm, A_ini, A_ini*r_ini)

print ('FIN')

# ----------- Cálculo del error cuadrático medio en la est. de A -------------------
mse_est_simple  = np.sum((vec_A_ini - A)**2,axis=0)/veces
mse_est_Kane    = np.sum((A_est_Kane - A)**2,axis=0)/veces
mse_est_Zabin   = np.sum((A_est_Zabin - A)**2,axis=0)/veces
mse_est_EM      = np.sum((A_est_EM - A)**2,axis=0)/veces

# ----------- Cálculo del error cuadrático medio en la est. de r -------------------
mse_est_simple_r = np.sum((vec_r_ini - r)**2,axis=0)/veces
mse_est_Kane_r   = np.sum((r_est_Kane - r)**2,axis=0)/veces
mse_est_Zabin_r  = np.sum((K_est_Zabin/A_est_Zabin - r)**2,axis=0)/veces
mse_est_EM_r     = np.sum((K_est_EM/A_est_EM - r)**2,axis=0)/veces

# ----------- Cálculo del error cuadrático medio en la est. de r -------------------
mse_est_simple_sigma = np.sum((vec_Sigmag2_ini - Sigma_G_sq)**2,axis=0)/veces

# -------------- FIGURAS -------------------------------------------------------------
# ------------------------------------------------------------------------------------
fig1 = plt.figure(figsize=(10, 5))
gs = GridSpec(nrows=2, ncols=2)

ax0 = fig1.add_subplot(gs[0, :])
ax0.set_yscale('log')
ax0.set_ylabel(r'MSE A',fontsize = 18)
ax0.set_xlabel('N [Quantity Samples]',fontsize = 18)
ax0.plot(X,mse_est_simple, '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'Proposed')
ax0.plot(X,mse_est_Kane, '--o', fillstyle = 'none', markersize = 5, color = 'tab:blue', label = 'Kanemoto')
ax0.plot(X,mse_est_Zabin, '--o', fillstyle = 'none', markersize = 5, color = 'tab:orange', label = 'Zabin-Poor')
ax0.plot(X,mse_est_EM, '--o', fillstyle = 'none', markersize = 5, color = 'tab:red', label = 'EM')
ax0.grid(True)

ax1 = fig1.add_subplot(gs[1, 0])
ax1.set_yscale('log')
ax1.set_ylabel(r'MSE $\Gamma$',fontsize = 18)
ax1.set_xlabel('N [Quantity Samples]',fontsize = 18)
ax1.plot(X,mse_est_simple_r, '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'Proposed')
ax1.plot(X,mse_est_Kane_r, '--o', fillstyle = 'none', markersize = 5, color = 'tab:blue', label = 'Kanemoto')
ax1.plot(X,mse_est_Zabin_r, '--o', fillstyle = 'none', markersize = 5, color = 'tab:orange', label = 'Zabin-Poor')
ax1.plot(X,mse_est_EM_r, '--o', fillstyle = 'none', markersize = 5, color = 'tab:red', label = 'EM')
ax1.grid(True)

ax2 = fig1.add_subplot(gs[1, 1])
ax2.set_yscale('log')
ax2.set_ylabel(r'MSE $\sigma_{G}^{2}$',fontsize = 18)
ax2.set_xlabel('N [Quantity Samples]',fontsize = 18)
ax2.plot(X,mse_est_simple_sigma, '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'Proposed')
ax2.grid(True)


ax0.legend(loc = 1, fontsize = 10)
ax1.legend(loc = 1, fontsize = 10)
ax2.legend(loc = 1, fontsize = 10)

plt.show()