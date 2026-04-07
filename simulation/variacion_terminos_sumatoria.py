import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA
from Estimadores import *
from Estimadores_terminos_suma import *

mpl.rc('font',family = 'Times New Roman')

#----------------------
N = 2000
#----------------------

#-------------------------------- Simulacion con A = 0.01 -------------------------------------------------
r = 0.01
A = 0.01
Sigma_G_sq = 0.001

A_est_Luc1       = np.zeros(11)
Sigmag2_est_Luc1 = np.zeros(11)
r_est_Luc1       = np.zeros(11)

env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,20,N,Sigma_G_sq)

A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)

A_est_Luc1[2],Sigmag2_est_Luc1[2],r_est_Luc1[2],Numiter_Luc = Est_Param_ClassA_CDF_2(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[3],Sigmag2_est_Luc1[3],r_est_Luc1[3],Numiter_Luc = Est_Param_ClassA_CDF_3(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[4],Sigmag2_est_Luc1[4],r_est_Luc1[4],Numiter_Luc = Est_Param_ClassA_CDF_4(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[5],Sigmag2_est_Luc1[5],r_est_Luc1[5],Numiter_Luc = Est_Param_ClassA_CDF_5(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[6],Sigmag2_est_Luc1[6],r_est_Luc1[6],Numiter_Luc = Est_Param_ClassA_CDF_6(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[7],Sigmag2_est_Luc1[7],r_est_Luc1[7],Numiter_Luc = Est_Param_ClassA_CDF_7(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[8],Sigmag2_est_Luc1[8],r_est_Luc1[8],Numiter_Luc = Est_Param_ClassA_CDF_8(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[9],Sigmag2_est_Luc1[9],r_est_Luc1[9],Numiter_Luc = Est_Param_ClassA_CDF_9(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc1[10],Sigmag2_est_Luc1[10],r_est_Luc1[10],Numiter_Luc = Est_Param_ClassA_CDF_10(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)


#-------------------------------- Simulacion con A = 0.1 -------------------------------------------------
r = 0.01
A = 0.1
Sigma_G_sq = 0.001

A_est_Luc2       = np.zeros(11)
Sigmag2_est_Luc2 = np.zeros(11)
r_est_Luc2       = np.zeros(11)

env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,20,N,Sigma_G_sq)

A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)

A_est_Luc2[2],Sigmag2_est_Luc2[2],r_est_Luc2[2],Numiter_Luc = Est_Param_ClassA_CDF_2(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[3],Sigmag2_est_Luc2[3],r_est_Luc2[3],Numiter_Luc = Est_Param_ClassA_CDF_3(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[4],Sigmag2_est_Luc2[4],r_est_Luc2[4],Numiter_Luc = Est_Param_ClassA_CDF_4(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[5],Sigmag2_est_Luc2[5],r_est_Luc2[5],Numiter_Luc = Est_Param_ClassA_CDF_5(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[6],Sigmag2_est_Luc2[6],r_est_Luc2[6],Numiter_Luc = Est_Param_ClassA_CDF_6(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[7],Sigmag2_est_Luc2[7],r_est_Luc2[7],Numiter_Luc = Est_Param_ClassA_CDF_7(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[8],Sigmag2_est_Luc2[8],r_est_Luc2[8],Numiter_Luc = Est_Param_ClassA_CDF_8(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[9],Sigmag2_est_Luc2[9],r_est_Luc2[9],Numiter_Luc = Est_Param_ClassA_CDF_9(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc2[10],Sigmag2_est_Luc2[10],r_est_Luc2[10],Numiter_Luc = Est_Param_ClassA_CDF_10(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)

#-------------------------------- Simulacion con A = 0.8 -------------------------------------------------
r = 0.01
A = 0.8
Sigma_G_sq = 0.001

A_est_Luc3       = np.zeros(11)
Sigmag2_est_Luc3 = np.zeros(11)
r_est_Luc3       = np.zeros(11)

env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,20,N,Sigma_G_sq)

A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)

A_est_Luc3[2],Sigmag2_est_Luc3[2],r_est_Luc3[2],Numiter_Luc = Est_Param_ClassA_CDF_2(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[3],Sigmag2_est_Luc3[3],r_est_Luc3[3],Numiter_Luc = Est_Param_ClassA_CDF_3(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[4],Sigmag2_est_Luc3[4],r_est_Luc3[4],Numiter_Luc = Est_Param_ClassA_CDF_4(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[5],Sigmag2_est_Luc3[5],r_est_Luc3[5],Numiter_Luc = Est_Param_ClassA_CDF_5(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[6],Sigmag2_est_Luc3[6],r_est_Luc3[6],Numiter_Luc = Est_Param_ClassA_CDF_6(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[7],Sigmag2_est_Luc3[7],r_est_Luc3[7],Numiter_Luc = Est_Param_ClassA_CDF_7(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[8],Sigmag2_est_Luc3[8],r_est_Luc3[8],Numiter_Luc = Est_Param_ClassA_CDF_8(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[9],Sigmag2_est_Luc3[9],r_est_Luc3[9],Numiter_Luc = Est_Param_ClassA_CDF_9(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc3[10],Sigmag2_est_Luc3[10],r_est_Luc3[10],Numiter_Luc = Est_Param_ClassA_CDF_10(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)


#-------------------------------- Simulacion con A = 1.5 -------------------------------------------------
r = 0.01
A = 1.5
Sigma_G_sq = 0.001

A_est_Luc4       = np.zeros(11)
Sigmag2_est_Luc4 = np.zeros(11)
r_est_Luc4       = np.zeros(11)

env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,20,N,Sigma_G_sq)

A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm)

A_est_Luc4[2],Sigmag2_est_Luc4[2],r_est_Luc4[2],Numiter_Luc = Est_Param_ClassA_CDF_2(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[3],Sigmag2_est_Luc4[3],r_est_Luc4[3],Numiter_Luc = Est_Param_ClassA_CDF_3(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[4],Sigmag2_est_Luc4[4],r_est_Luc4[4],Numiter_Luc = Est_Param_ClassA_CDF_4(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[5],Sigmag2_est_Luc4[5],r_est_Luc4[5],Numiter_Luc = Est_Param_ClassA_CDF_5(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[6],Sigmag2_est_Luc4[6],r_est_Luc4[6],Numiter_Luc = Est_Param_ClassA_CDF_6(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[7],Sigmag2_est_Luc4[7],r_est_Luc4[7],Numiter_Luc = Est_Param_ClassA_CDF_7(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[8],Sigmag2_est_Luc4[8],r_est_Luc4[8],Numiter_Luc = Est_Param_ClassA_CDF_8(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[9],Sigmag2_est_Luc4[9],r_est_Luc4[9],Numiter_Luc = Est_Param_ClassA_CDF_9(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)
A_est_Luc4[10],Sigmag2_est_Luc4[10],r_est_Luc4[10],Numiter_Luc = Est_Param_ClassA_CDF_10(env_data_DesNorm,[A_ini,Sigmag2_ini,r_ini],1000,0.00001)

fig1,ax1 = plt.subplots(1)

ax1.set_ylabel(r'$\hat{A}$',fontsize = 14)
#ax1.set_yscale('log')
ax1.set_xlim(xmin = 2, xmax = 10)
ax1.plot(A_est_Luc1,'--o', fillstyle = 'none',markersize = 5, color = 'black', label = 'A = 0.01')
ax1.plot(A_est_Luc2,'--o', fillstyle = 'none',markersize = 5, color = 'tab:green', label = 'A = 0.1')
ax1.plot(A_est_Luc3,'--o', fillstyle = 'none',markersize = 5, color = 'tab:blue', label = 'A = 0.8')
ax1.plot(A_est_Luc4,'--o', fillstyle = 'none',markersize = 5, color = 'tab:orange', label = 'A = 1.5')
#ax1.set_xlabel(r'Cantidad de términos en la sumatoria $CDF_{te\acute{o}rica}$',fontsize = 14) en español
ax1.set_xlabel(r'Number of terms in $CDF_{theoretical}$',fontsize = 14) # en inglés
ax1.grid(True, which = 'both')

ax1.tick_params(axis="x", labelsize=11)
ax1.tick_params(axis="y", labelsize=11)
ax1.legend(loc = 'best', fontsize = 8)

plt.show()