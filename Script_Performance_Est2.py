import numpy as np
import math
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA
from Estimadores import *

matlab = loadmat("/home/lucas/Escritorio/matlab3.mat")
env_data_DesNorm = matlab["env_data_DesNorm"]
env_data_Norm = matlab["env_data_Norm"]


# r = 0.001
# A = 0.1
# Sigmag2 = 0.001
# cte = Sigmag2*(1+1/r)

# paso = 0.1
# x_min = -100
# x_max = 40
# N_point = (x_max-x_min)/paso

# X = np.linspace(x_min,x_max,N_point)

# long = len(X)

n = 10

# Inicializacion de vectores de estimaciones estimador lucas
A_est_Luc       = np.zeros(n)
Sigmag2_est_Luc = np.zeros(n)
r_est_Luc       = np.zeros(n)

# Inicializacion de vectores de estimaciones estimador Kanemoto
A_est_Kane = np.zeros(n)
r_est_Kane = np.zeros(n)
K_est_Kane = np.zeros(n)

# Inicializacion de vectores de estimaciones estimador Zabin
A_est_Zabin = np.zeros(n)
K_est_Zabin = np.zeros(n)

# Inicializacion de vectores de estimaciones estimador EM
A_est_EM = np.zeros(n)

# Inicializacion del vector A_ini
vec_A_ini = np.zeros(n)

print (np.shape(env_data_Norm))
print (np.shape(env_data_DesNorm))

#env_data = np.zeros([n,long])
for i in range(n):
    #env_data = RFI_MakeEnvelopeDataClassA(A,r,10,long,Sigmag2)

    # Estimador inicial
    A_ini,Sigmag2_ini,r_ini = est_inicial(env_data_DesNorm[i])

    # Estimador mio
    A_est_Luc[i],Sigmag2_est_Luc[i],r_est_Luc[i],Numiter_Luc = Est_Param_ClassA_CDF(env_data_DesNorm[i,:],[A_ini,Sigmag2_ini,r_ini],1000,0.0001)

    # Estimador Kanemoto
    A_est_Kane[i],r_est_Kane[i],K_est_Kane[i] = Est_Momentos_Kanemoto(env_data_Norm[i,:])

    # Estimador Zabin
    A_est_Zabin[i],K_est_Zabin[i] = Est_Momentos_Zabin_Poor(env_data_Norm[i,:])

    # Estimador EM
    A_est_EM[i],NumIter = RFI_EMParamA(5000, 10, env_data_Norm[i,:],A_ini,A_ini*r_ini,9)

    # Vector de A_ini
    vec_A_ini[i] = A_ini

    print (i)

# Grafica de los valores de A estimados
"""plt.figure(1)
plt.plot()
plot(A_est_Kane,'r'),grid
hold on
plot(A_est_Zabin,'b')
plot(A_est_EM,'g')
plot(A_estLuc,'k')
"""

print('FIN')
plt.figure(1)
plt.plot(A_est_Luc,'rp',markersize = 4)
plt.plot(A_est_Kane,'bo',markersize = 8)
plt.plot(A_est_Zabin,'w*',markersize = 4)
plt.plot(A_est_EM,'gD',markersize = 4)
plt.plot(vec_A_ini)
plt.grid(True)
plt.show()

A_est_Luc_prom = np.cumsum(A_est_Luc)
A_est_Kane_prom = np.cumsum(A_est_Kane)
A_est_Zabin_prom = np.cumsum(A_est_Zabin)
A_est_EM_prom = np.cumsum(A_est_EM)

for i in range(n):
    A_est_Luc_prom[i] = A_est_Luc_prom[i]/(i+1)
    A_est_Kane_prom[i] = A_est_Kane_prom[i]/(i+1)
    A_est_Zabin_prom[i] = A_est_Zabin_prom[i]/(i+1)
    A_est_EM_prom[i] = A_est_EM_prom[i]/(i+1)

plt.figure(2)
plt.plot(A_est_Luc_prom,'ro',markersize = 4)
plt.plot(A_est_Kane_prom,'bp',markersize = 4)
plt.plot(A_est_Zabin_prom,'ks',markersize = 4)
plt.plot(A_est_EM_prom,'gD',markersize = 4)
plt.grid(True)
plt.show()

print ('FIN2')