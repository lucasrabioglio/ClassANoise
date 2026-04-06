from ClassA import RFI_MakeEnvelopeDataClassA
from func_min import func_min

import numpy as np
import matplotlib.pyplot as plt
import sys


# Valores de los distintos parametros para utilizar la funcion RFI_MakeEnvelopeDataClassA
# -----------------------------------
A = 0.1
r = 0.001
M = 10
N = 300
Sigmag2 = 0.001
# -----------------------------------

# Inicializacion del vector que contendra las muestras de Ruido de Envolvente de Class A
#-------------------------------
env_data_Norm    = np.zeros(N)
env_data_DesNorm = np.zeros(N)
# -------------------------------

# Llamado a funcion para generar las muestras de ruido de envolvente
# ----------------------------------------------------
env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,M,N,Sigmag2)
# ----------------------------------------------------

# Dado que Potencia = Envelope^2/2, se calcula la misma
# ---------------------------------
potencia = (env_data_DesNorm**2)/2
# ---------------------------------

# Calculo del logaritmo en base 10 de la potencia, es decir, potencia en dBW (respecto a 1 WATT)
# ----------------------------------
potencia_log = 10*np.log10(potencia)
# ----------------------------------

minimo = min(potencia_log)
maximo = max(potencia_log)
paso = (maximo - minimo)/(N-1)

x = np.linspace(minimo-2*paso,maximo+2*paso,N+2)

# Calculo del Histograma del logaritmo de la potencia (PDF experimental)
# ---------------------------------------------------------------------
histo, bin_edge = np.histogram(potencia_log,N+1,(minimo-2*paso,maximo+2*paso),density = True)

bin_edge2 = bin_edge[:N+1]

fig1, ax1 = plt.subplots(num=1)
ax1.plot(potencia_log,'ro')
ax1.hlines(x,0,N,color = 'r',linestyle = ':')
ax1.grid(True)

fig2, ax2 = plt.subplots(num=2)
ax2.vlines(x,0,0.2,color = 'r',linestyle = ':',linewidth = 3)
ax2.vlines(bin_edge,0,0.2,color = 'b',linestyle = '--')
ax2.plot(bin_edge[:N+1],histo,'ro',markersize = 5)
ax2.grid(True)

cumulative = np.cumsum(histo)
Xcdf = cumulative/max(cumulative)
Xcdf_aux = 1 - Xcdf

fig3, ax3 = plt.subplots(num=3)
ax3.step(bin_edge[:N+1],Xcdf_aux,where = 'post',linewidth = 3)
ax3.vlines(bin_edge,0,1,color = 'b')
ax3.grid(True)

plt.show()
print('FIN')



#Y = func_min(bin_edge2,Xcdf_aux,A,Sigmag2,r)
#print(Y)