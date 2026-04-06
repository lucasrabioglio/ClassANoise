from ClassA import RFI_MakeEnvelopeDataClassA
#from Estimadores import Est_Momentos_Kanemoto
#import Estimadores as Est

import numpy as np
import matplotlib.pyplot as plt
import sys


# Valores de los distintos parametros
# para utilizar la funcion
# RFI_MakeEnvelopeDataClassA
# -----------------------------------
A = 0.1
r = 0.001
M = 10
N = 300
Sigmag2 = 0.001
# -----------------------------------

# Inicializacion del vector que 
# contendra las muestras de Ruido
# de Envolvente de Class A
#-------------------------------
env_data = np.zeros(N)
# -------------------------------


# Llamado a funcion para generar las muestras de ruido
# de envolvente
# ----------------------------------------------------
env_data = RFI_MakeEnvelopeDataClassA(A,r,M,N,Sigmag2)
# ----------------------------------------------------

# Dado que Potencia = Envelope^2/2,
# se calcula la misma
# ---------------------------------
potencia = (env_data**2)/2
# ---------------------------------


# Grafico de los valores de ruido de
# envolvente
# ----------------------------------
plt.figure(1)
plt.plot(env_data,'ro',markersize=3)
plt.ylabel('env_data')
plt.title('Muestras de Ruido')
plt.grid(True)
# ----------------------------------


# Grafico de los valores de potencia 
# correspondientes a los valores de 
# env_data
# ----------------------------------
plt.figure(2)
plt.plot(potencia,'ro',markersize=3)
plt.title('Potencia de Ruido')
plt.ylabel('Potencia')
plt.grid(True)
#plt.show()
# ----------------------------------


# Calculo del logaritmo en base 10 
# de la potencia, es decir, potencia
# en dBW (respecto a 1 WATT)
# ----------------------------------
potencia_log = 10*np.log10(potencia)
# ----------------------------------

minimo = min(potencia_log)
maximo = max(potencia_log)
paso = (maximo - minimo)/(N-1)

x = np.linspace(minimo-2*paso,maximo+2*paso,N+2)

# Calculo del Histograma del logaritmo de la potencia (PDF experimental)
# ---------------------------------------------------------------------
histo, bin_edge = np.histogram(potencia_log,N+2,(minimo-2*paso,maximo+2*paso),density = True)

fig3, ax3 = plt.subplots(num=3)
ax3.plot(potencia_log,'bo',markersize=3)
ax3.hlines(x,0,N,color = 'r',linestyle = ':')
ax3.hlines(bin_edge,0,N,color = 'b',linestyle = '--')

fig10, ax10 = plt.subplots(num=10)
ax10.vlines(bin_edge,0,0.2, color = 'b', linestyle = '--')
ax10.vlines(x,0,0.2, color = 'r', linestyle = ':')
ax10.grid(True)

fig4, ax4 = plt.subplots(num=4)
l1 = ax4.vlines(bin_edge[::1],0,0.2,color = 'b',linestyle = '--')
#xhisto = np.linspace(minimo-2*paso,maximo+2*paso,N+2)
l2 = ax4.plot(x,histo,'ro',markersize=3)
ax4.vlines(x,0,0.2,color = 'r',linestyle = ':')
ax4.grid(True)
#print(bin_edge)
# ---------------------------------------------------------------------

cumulative_function = np.cumsum(histo)
cumulative_function = cumulative_function/max(cumulative_function)
cumulative = 1 - cumulative_function/max(cumulative_function)
print(cumulative_function)
print(cumulative)

fig5, ax5 = plt.subplots(num=5)
ax5.vlines(bin_edge,0,1,color = 'b',linestyle = '--')
ax5.step(bin_edge[:N+2],cumulative,where = 'post',linewidth = 3)
ax5.grid(True)

fig20, ax20 = plt.subplots(num=20)
ax20.vlines(bin_edge,0,1,color = 'b',linestyle = '--')
ax20.step(x,cumulative_function/max(cumulative_function),where = 'post',linewidth = 3)
ax20.grid(True)

#  Grafico de la CDF experimental (Funcion Distribucion de Probabilidad Acumulada)
# -------------------------------------------------------------------------------------
fig6, ax6 = plt.subplots(num=6)
ax6_var = ax6.hist(potencia_log,x,density=True, histtype='step',cumulative=-1, label='Empirical',linewidth = 4)
ax6.vlines(x,0,1)
Xcdf = ax6_var[1]
ax6.grid(True)
# -------------------------------------------------------------------------------------

fig7, ax7 = plt.subplots(num=7)
ax7.hist(potencia_log,bin_edge,density=True, histtype='step',cumulative=-1, label='Empirical',linewidth = 2)
ax7.step(x,cumulative,linewidth = 2,where = 'post')
ax7.vlines(bin_edge,0,1)
ax7.grid(True)


fig30, ax30 = plt.subplots(num=30)
ax30_var = ax30.hist(potencia_log,bin_edge,density=True, histtype='step',cumulative=1, label='Empirical',linewidth = 2)
#ax30.step(bin_edge[:N+2],cumulative,linewidth = 2,where = 'post')
ax30.vlines(bin_edge,0,1,color = 'b',linestyle = '--')
ax30.vlines(x,0,1,color = 'r',linestyle = ':')
ax30.grid(True)
print('Figura 30')
print(ax30_var)
# Mostrar todos los
# graficos (plots)
# -----------------
plt.show()
# -----------------
print(x)
print(Xcdf)