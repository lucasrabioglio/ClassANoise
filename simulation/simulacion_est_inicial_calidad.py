import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA
from Estimadores import *
from RFI_EMTwoParamEst import *
from RFI_EMParamA import *

mpl.rc('font',family = 'Times New Roman')

#----------------------
N = 1000
#----------------------
r = 0.001
A = 0.05
Sigma_G_sq = 0.001
#----------------------
n = 5
env_data_Norm    = np.zeros((n,N))
env_data_DesNorm = np.zeros((n,N))

# ------------- Generacion de vectores muestras -------------------------------
for i in range(n):
    env_data_Norm[i],env_data_DesNorm[i] = RFI_MakeEnvelopeDataClassA(A,r,10,N,Sigma_G_sq)


# Inicializacion de vectores de estimaciones estimador Lucas
A_est_Luc = np.zeros(n)
r_est_Luc = np.zeros(n)
Sigmag2_est_Luc = np.zeros(n)

# Inicializacion de vectores de estimaciones estimador Kanemoto
A_est_Kane = np.zeros(n)
r_est_Kane = np.zeros(n)
K_est_Kane = np.zeros(n)

# Inicializacion de vectores de estimaciones estimador Zabin
A_est_Zabin = np.zeros(n)
K_est_Zabin = np.zeros(n)

# Inicializacion de vectores de estimaciones estimador EM
A_est_EM = np.zeros(n)
K_est_EM = np.zeros(n)

# Inicializacion de vectores del Estimador Inicial
vec_A_ini = np.zeros(n)
vec_r_ini = np.zeros(n)
vec_Sigmag2_ini = np.zeros(n)

for i in range(n):
    #env_data = RFI_MakeEnvelopeDataClassA(A,r,10,long,Sigmag2)

    # Estimador inicial
    A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm[i])

    # Estimador mio
    #A_est_Luc[i],Sigmag2_est_Luc[i],r_est_Luc[i],Numiter_Luc = Est_Param_ClassA_CDF(env_data_DesNorm[i,:],[A_ini,Sigmag2_ini,r_ini],1000,0.0001)

    # Estimador Kanemoto
    A_est_Kane[i],r_est_Kane[i],K_est_Kane[i] = Est_Momentos_Kanemoto(env_data_Norm[i,:])

    # Estimador Zabin
    A_est_Zabin[i],K_est_Zabin[i] = Est_Momentos_Zabin_Poor(env_data_Norm[i,:])

    # Estimador EM
    A_est_EM[i],NumIter = RFI_EMParamA(N, 10, env_data_Norm[i,:],A_ini,A_ini*r_ini,9)
    A_errado,K_est_EM[i],NumIter = RFI_EMTwoParamEst(N, 10, env_data_Norm[i,:], A_ini, A_ini*r_ini)

    # Vector de A_ini
    vec_A_ini[i] = A_ini
    vec_r_ini[i] = r_ini
    vec_Sigmag2_ini[i] = Sigmag2_ini

    print (i)


print('FIN')
#fig1,(ax1,ax2) = plt.subplots(num=2,figsize = (7,5),dpi = 140)
fig1,(ax1,ax2,ax3) = plt.subplots(3)

ax1.tick_params(axis='x', labelsize=13)
ax1.tick_params(axis='y', labelsize=13)
ax1.set_ylabel('EST. A',fontsize = 14)
ax1.plot(A_est_Kane,'--o',markersize = 5, color = 'tab:blue', label = 'Kanemoto')
ax1.plot(A_est_Zabin,'--o',markersize = 5, color = 'tab:red', label = 'Zabin-Poor')
ax1.plot(A_est_EM,'--o',markersize = 5, color = 'tab:orange', label = 'EM')
ax1.plot(vec_A_ini,'--o', markersize = 5, color = 'black', label = 'Proposed')
#ax1.plot(A_est_Luc,'--o', markersize = 5, color = 'green', label = 'Est. Propuesto')
ax1.axhline(xmin = 0,xmax = n,y = A, linestyle = '--',color = 'tab:olive',linewidth = 1.5, label = 'True value')
ax1.grid(True)

ax2.tick_params(axis='x', labelsize=13)
ax2.tick_params(axis='y', labelsize=13)
ax2.set_ylabel(r'EST. $\Gamma$',fontsize = 14)
ax2.set_yscale('log')
ax2.plot(r_est_Kane,'--o',markersize = 5, color = 'tab:blue', label = 'Kanemoto')
ax2.plot(K_est_Zabin/A_est_Zabin,'--o',markersize = 5, color = 'tab:red', label = 'Zabin-Poor')
ax2.plot(K_est_EM/A_est_EM,'--o',markersize = 5, color = 'tab:orange', label = 'EM')
ax2.plot(vec_r_ini,'--o', markersize = 5, color = 'black', label = 'Proposed')
#ax2.plot(r_est_Luc,'--o', markersize = 5, color = 'green', label = 'Est. Propuesto')
ax2.axhline(xmin = 0,xmax = n,y = r, linestyle = '--',color = 'tab:olive',linewidth = 1.5, label = 'True value')
ax2.grid(True)

ax3.tick_params(axis='x', labelsize=13)
ax3.tick_params(axis='y', labelsize=13)
ax3.set_ylabel(r'EST. $\sigma_{G}^{2}$',fontsize = 14)
ax3.plot(vec_Sigmag2_ini,'--o', markersize = 5, color = 'black', label = 'Proposed')
#ax3.plot(Sigmag2_est_Luc,'--o', markersize = 5, color = 'green', label = 'Est. Propuesto')
ax3.axhline(xmin = 0,xmax = n,y = Sigma_G_sq, linestyle = '--',color = 'tab:olive',linewidth = 1.5, label = 'True value')
ax3.set_xlabel('Iterations (N = 1000 Samples)',fontsize = 14)
ax3.grid(True)

"""
#------------------------ A --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha A -----------------------------------------
ax1.annotate(r'$A $', xy=(0.25, A), xytext=(2.5, 0.1),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'))

#------------------------ Gamma --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha Gamma -----------------------------------------
ax2.annotate(r'$\Gamma $', xy=(0.5, r), xytext=(0, 1.5*r),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'))

#------------------------ Sigma_G_sq --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha Sigma_G_sq -----------------------------------------
ax3.annotate(r'$\sigma_{G}^{2} $', xy=(0.5, Sigma_G_sq), xytext=(0, 0.003),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'));
"""

ax1.legend(loc = 1, fontsize = 8)
ax2.legend(loc = 1, fontsize = 8)
ax3.legend(loc = 1, fontsize = 8)

plt.show()