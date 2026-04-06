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
r = 0.05
A = 0.05
Sigma_G_sq = 0.001
#----------------------
n = 50

N = np.arange(n)
N = N[1:]*100 - 1

X = np.arange(n)
X = X[1:]*100

# Inicializacion del vector A_ini
vec_A_ini = np.zeros(n)
vec_r_ini = np.zeros(n)
vec_Sigmag2_ini = np.zeros(n)

for i in range(len(N)):
    env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N[i],Sigma_G_sq)

    # Estimador inicial
    vec_A_ini[i],vec_Sigmag2_ini[i],vec_r_ini[i] = est_inicial(env_data_DesNorm)

print ('FIN')

###########################################################################################

n = 51
veces = 10

N = np.arange(n)
N = N[1:]*100

X = np.arange(n+1)
X = X[1:]*100

# Inicializacion del vector A_ini
vec_A_ini2       = np.zeros((veces,n))
vec_r_ini2       = np.zeros((veces,n))
vec_Sigmag2_ini2 = np.zeros((veces,n))

for j in range(len(N)):
    print (j)
    for i in range(veces):
        env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N[j],Sigma_G_sq)

        # Estimador inicial
        A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)
        vec_A_ini2[i,j]          = A_ini
        vec_Sigmag2_ini2[i,j]    = Sigmag2_ini
        vec_r_ini2[i,j]          = r_ini

print ('FIN')

# ----------- Cálculo del error cuadrático medio -------------------
mse_est_simple        = np.sum((vec_A_ini2 - A)**2,axis=0)/veces
mse_est_simple_Gamma  = np.sum((vec_r_ini2 - r)**2,axis=0)/veces

#########################################################################################3

fig1,(ax1A,ax1B,ax1C) = plt.subplots(3)

ax1A.tick_params(axis='x', labelsize=13)
ax1A.tick_params(axis='y', labelsize=13)
ax1A.set_yscale('log')
ax1A.set_ylabel('Estimated A-index',fontsize = 16)
#ax1A.plot(X[:-2],vec_A_ini[:-1],'--o', markersize = 5, color = 'black', label = 'Est. Propuesto')
ax1A.plot(X[:-2],vec_A_ini[:-1],'--o', markersize = 5, color = 'black')
ax1A.axhline(xmin = 0,xmax = n,y = A, linestyle = '--',color = 'green',linewidth = 2)

#------------------------ A --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha A -----------------------------------------
ax1A.annotate(r'$A $', xy=(100, A), xytext=(1000, A + A),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'))

ax1A.grid(True)

ax1B.tick_params(axis='x', labelsize=13)
ax1B.tick_params(axis='y', labelsize=13)
ax1B.set_yscale('log')
ax1B.set_ylabel('MSE',fontsize = 16)
ax1B.set_xlabel('N [Quantity samples]',fontsize = 16)
#ax1B.plot(X[:-1],mse_est_simple[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'black', label = 'MSE est. propuesto')
ax1B.plot(X[:-1],mse_est_simple[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'black')
ax1B.grid(True)

ax1C.tick_params(axis='x', labelsize=13)
ax1C.tick_params(axis='y', labelsize=13)
ax1C.set_yscale('log')
ax1C.set_ylabel('MSE',fontsize = 16)
ax1C.set_xlabel('N [Quantity samples]',fontsize = 16)
ax1C.plot(X[:-1],mse_est_simple_Gamma[:-1], '--o', fillstyle = 'none', markersize = 5, color = 'black')
ax1C.grid(True)

######################################################################

#ax1A.legend(loc = 1, fontsize = 8)
#ax1B.legend(loc = 1, fontsize = 8)

plt.show()