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
"""
n = 101
veces = 50

N = np.arange(n)
N = N[1:]*100

X = np.arange(n+1)
X = X[1:]*100
"""

veces = 2
N = np.array([500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])
n = np.size(N)

X = np.array([500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000])

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

print ('FIN')

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


fig1,ax1 = plt.subplots(1)

#X = np.reshape(X,(1,len(X)))

ax1.set_ylabel('Est. A',fontsize = 16)
ax1.set_xlabel('N [Cant. de Muestras]',fontsize = 16)
for k in range(veces):
    #ax1.plot(X[:-1],vec_A_ini[k,:-1], 'o', fillstyle = 'none', markersize = 5, color = 'black', label = 'Est. Propuesto')
    ax1.plot(X,A_est_Luc[k,:], 'o', fillstyle = 'full', markersize = 5, color = 'green')
ax1.axhline(xmin = 0,xmax = n,y = A, linestyle = '--',color = 'red',linewidth = 2, label = 'Valor Real')

#------------------------ A --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha A -----------------------------------------
#ax1.annotate(r'$A $', xy=(100, A), xytext=(1000, A + A),
#                   arrowprops=dict(arrowstyle="->",
#                                   connectionstyle="angle3,angleA=0,angleB=-90"),
#                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'))

ax1.grid(True)

"""
fig2,(ax2A,ax2B) = plt.subplots(2)
ax2A.set_yscale('log')
ax2A.set_ylabel('MSE',fontsize = 18)
ax2A.plot(X[:-1],mse_est_simple[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'MSE est. simple')
ax2A.plot(X[:-1],mse_est_Luc[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'green', label = 'MSE est. propuesto')
ax2A.plot(X[:-1],mse_est_Kane[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'tab:blue', label = 'MSE est. Kanemoto')
ax2A.plot(X[:-1],mse_est_Zabin[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'tab:orange', label = 'MSE est. Zabin-Poor')
ax2A.plot(X[:-1],mse_est_EM[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'tab:red', label = 'MSE est. EM')
ax2A.grid(True)

ax2B.set_yscale('log')
ax2B.set_ylabel('MSE',fontsize = 18)
ax2B.set_xlabel('N [Cant. de Muestras]',fontsize = 18)
ax2B.plot(X[:-1],mse_est_simple[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'MSE est. simple')
ax2B.plot(X[:-1],mse_est_Luc[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'green', label = 'MSE est. propuesto')
ax2B.grid(True)
"""

fig2,(ax2A) = plt.subplots(1)
ax2A.set_yscale('log')
ax2A.set_ylabel(r'MSE A',fontsize = 18)
ax2A.set_xlabel('N [Cant. de Muestras]',fontsize = 18)
ax2A.plot(X,mse_est_simple, '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'MSE est. simple')
ax2A.plot(X,mse_est_Luc, '--o', fillstyle = 'none', markersize = 5, color = 'green', label = 'MSE est. propuesto')
ax2A.plot(X,mse_est_Kane, '--o', fillstyle = 'none', markersize = 5, color = 'tab:blue', label = 'MSE est. Kanemoto')
ax2A.plot(X,mse_est_Zabin, '--o', fillstyle = 'none', markersize = 5, color = 'tab:orange', label = 'MSE est. Zabin-Poor')
ax2A.plot(X,mse_est_EM, '--o', fillstyle = 'none', markersize = 5, color = 'tab:red', label = 'MSE est. EM')
ax2A.grid(True)

fig3,(ax3A,ax3B) = plt.subplots(2)
ax3A.set_yscale('log')
ax3A.set_ylabel(r'MSE $\Gamma$',fontsize = 18)
ax3A.plot(X,mse_est_simple_r, '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'MSE est. simple')
ax3A.plot(X,mse_est_Luc_r, '--o', fillstyle = 'none', markersize = 5, color = 'green', label = 'MSE est. propuesto')
ax3A.plot(X,mse_est_Kane_r, '--o', fillstyle = 'none', markersize = 5, color = 'tab:blue', label = 'MSE est. Kanemoto')
ax3A.plot(X,mse_est_Zabin_r, '--o', fillstyle = 'none', markersize = 5, color = 'tab:orange', label = 'MSE est. Zabin-Poor')
ax3A.plot(X,mse_est_EM_r, '--o', fillstyle = 'none', markersize = 5, color = 'tab:red', label = 'MSE est. EM')
ax3A.grid(True)

ax3B.set_yscale('log')
ax3B.set_ylabel(r'MSE $\sigma_{G}^{2}$',fontsize = 18)
ax3B.set_xlabel('N [Cant. de Muestras]',fontsize = 18)
ax3B.plot(X,mse_est_simple_sigma, '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'MSE est. simple')
ax3B.plot(X,mse_est_Luc_sigma, '--o', fillstyle = 'none', markersize = 5, color = 'green', label = 'MSE est. propuesto')
ax3B.grid(True)


ax1.legend(loc = 1, fontsize = 10)
ax2A.legend(loc = 1, fontsize = 10)
ax3A.legend(loc = 1, fontsize = 10)
ax3B.legend(loc = 1, fontsize = 10)

plt.show()