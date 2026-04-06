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
r = 0.001
A = 0.01
Sigma_G_sq = 0.001
#----------------------
n = 50

"""
N = 1000
env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N,Sigma_G_sq)
print (np.sqrt(np.mean(env_data_Norm**2/2)))
print (np.sqrt(np.mean(env_data_DesNorm**2/2)))
"""

N = np.arange(n)
N = N[1:]*100 - 1

X = np.arange(n)
X = X[1:]*100

# Inicializacion del vector A_ini
vec_A_ini       = np.zeros(n)
vec_r_ini       = np.zeros(n)
vec_Sigmag2_ini = np.zeros(n)
tiempo          = np.zeros(n)

for i in range(len(N)):
    env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N[i],Sigma_G_sq)

    # Estimador inicial
    t0 = tm.time()
    vec_A_ini[i],vec_Sigmag2_ini[i],vec_r_ini[i] = est_inicial(env_data_DesNorm)
    tiempo[i] = tm.time() - t0

print ('FIN')

fig1,ax1 = plt.subplots(1)

ax1.set_ylabel('A Estimado',fontsize = 16)
ax1.set_xlabel('N [Cant. de Muestras]',fontsize = 16)
ax1.plot(X,vec_A_ini[:-1],'--o', markersize = 5, color = 'black', label = 'Est. Propuesto')
ax1.axhline(xmin = 0,xmax = n,y = A, linestyle = '--',color = 'green',linewidth = 2)

#------------------------ A --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha A -----------------------------------------
ax1.annotate(r'$A $', xy=(100, A), xytext=(1000, A + A),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'))

ax1.grid(True)

plt.show()

print (tiempo)
"""
env_data_Norm    = np.zeros((n,N))
env_data_DesNorm = np.zeros((n,N))


# ------------- Generacion de vectores muestras -------------------------------
for i in range(n):
    env_data_Norm[i],env_data_DesNorm[i] = RFI_MakeEnvelopeDataClassA(A,r,10,N,Sigma_G_sq)
"""