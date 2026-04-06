import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA

mpl.rc('font',family = 'Times New Roman')

N = 10000
#----------------------
r = 0.01
A = 0.05
Sigma_G_sq = 0.001
#----------------------
env_data = np.zeros(N)
env_data_Norm,env_data_DesNorm = RFI_MakeEnvelopeDataClassA(A,r,10,N,Sigma_G_sq)
potencia = (env_data_DesNorm**2)/2

# ---------------------------------
# Calculo del logaritmo en base 10 de la potencia, es decir, potencia en dBW (respecto a 1 WATT)
# ----------------------------------
potencia_log = 10*np.log10(potencia)
minimo = min(potencia_log)
maximo = max(potencia_log)
paso = (maximo - minimo)/(N-1)

lim_inf = minimo
lim_sup = maximo
num_ptos = N

#----------------------------------------------
D = np.log(10)/10
#----------------------
X = np.linspace(lim_inf,lim_sup,num_ptos)
X3 = 10**(X/10)
#------------------------------------------

#----------------------
L0 = 1/Sigma_G_sq
L1 = r/(Sigma_G_sq*(1/A + r))
L2 = r/(Sigma_G_sq*(2/A + r))
L3 = r/(Sigma_G_sq*(3/A + r))
L4 = r/(Sigma_G_sq*(4/A + r))
L5 = r/(Sigma_G_sq*(5/A + r))
L6 = r/(Sigma_G_sq*(6/A + r))
L7 = r/(Sigma_G_sq*(7/A + r))
L8 = r/(Sigma_G_sq*(8/A + r))
L9 = r/(Sigma_G_sq*(9/A + r))
L10 = r/(Sigma_G_sq*(10/A + r))
#-----------------------
p0 = np.exp(-A)
p1 = A*np.exp(-A)
p2 = (A**2)*np.exp(-A)/2
p3 = (A**3)*np.exp(-A)/6
p4 = (A**4)*np.exp(-A)/np.math.factorial(4)
p5 = (A**5)*np.exp(-A)/np.math.factorial(5)
p6 = (A**6)*np.exp(-A)/np.math.factorial(6)
p7 = (A**7)*np.exp(-A)/np.math.factorial(7)
p8 = (A**8)*np.exp(-A)/np.math.factorial(8)
p9 = (A**9)*np.exp(-A)/np.math.factorial(9)
p10 = (A**10)*np.exp(-A)/np.math.factorial(10)
#-----------------------
T0 = L0*D*np.exp(D*X - L0*np.exp(D*X))
T1 = L1*D*np.exp(D*X - L1*np.exp(D*X))
T2 = L2*D*np.exp(D*X - L2*np.exp(D*X))
T3 = L3*D*np.exp(D*X - L3*np.exp(D*X))
T4 = L4*D*np.exp(D*X - L4*np.exp(D*X))
T5 = L5*D*np.exp(D*X - L5*np.exp(D*X))
T6 = L6*D*np.exp(D*X - L6*np.exp(D*X))
T7 = L7*D*np.exp(D*X - L7*np.exp(D*X))
T8 = L8*D*np.exp(D*X - L8*np.exp(D*X))
T9 = L9*D*np.exp(D*X - L9*np.exp(D*X))
T10 = L10*D*np.exp(D*X - L10*np.exp(D*X))
#-------------------------
V0 = p0*T0
V1 = p1*T1
V2 = p2*T2
V3 = p3*T3
V4 = p4*T4
V5 = p5*T5
V6 = p6*T6
V7 = p7*T7
V8 = p8*T8
V9 = p9*T9
V10 = p10*T10
#-------------------------
V = V0 + V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10
#-------------------------------------------------------

#-------------------- CDF ------------------------------
#-------------------------------------------------------
G0 = np.exp(-L0*X3)
G1 = np.exp(-L1*X3)
G2 = np.exp(-L2*X3)
G3 = np.exp(-L3*X3)
G4 = np.exp(-L4*X3)
G5 = np.exp(-L5*X3)
G6 = np.exp(-L6*X3)
G7 = np.exp(-L7*X3)
G8 = np.exp(-L8*X3)
G9 = np.exp(-L9*X3)
G10 = np.exp(-L10*X3)

W0 = p0*G0
W1 = p1*G1
W2 = p2*G2
W3 = p3*G3
W4 = p4*G4
W5 = p5*G5
W6 = p6*G6
W7 = p7*G7
W8 = p8*G8
W9 = p9*G9
W10 = p10*G10

W = W0 + W1 + W2 + W3 + W4 + W5 + W6 + W7 + W8 + W9 + W10
#--------------------------------------------------------

#----------- Generación de datos experimentales ------------
# Inicializacion del vector que 
# contendra las muestras de Ruido
# de Envolvente de Class A
#-------------------------------

# -------------------------------

# Llamado a funcion para generar las muestras de ruido
# de envolvente
# ----------------------------------------------------

# ----------------------------------------------------

# Dado que Potencia = Envelope^2/2,
# se calcula la misma
# ---------------------------------

# ----------------------------------


x = np.linspace(minimo-2*paso,maximo+2*paso,N+2)

# Calculo del Histograma del logaritmo de la potencia (PDF experimental)
# ---------------------------------------------------------------------
histo, bin_edge = np.histogram(potencia_log,N+1,(minimo-2*paso,maximo+2*paso),density = True)

bin_edge2 = bin_edge[:N+1]

cumulative_function = np.cumsum(histo)
cumulative_function = cumulative_function/max(cumulative_function)
cumulative = 1 - cumulative_function/max(cumulative_function)

Fcosto = (cumulative[:-1] - W)**2
#----------------- Gráficos -----------------------------
#--------------------------------------------------------
"""
#---------------------- pdf teórica ---------------------
fig1,ax1 = plt.subplots(num=1,figsize = (7,5),dpi = 140)
ax1.plot(X,V,'k',linewidth = 1)
ax1.set_xlabel('Potencia (dBW)',fontsize = 18)
ax1.set_ylabel('Densidad de Probabilidad',fontsize = 18)
ax1.set_title('Función Densidad de Probabilidad',fontsize = 20)
"""
#---------------------- pdf teórica ---------------------
fig1,ax1 = plt.subplots(num=1,figsize = (7,5),dpi = 140)
ax1.plot(X,W,'k',linewidth = 3)
ax1.set_xlabel('Potencia (dBW)',fontsize = 18)
ax1.set_ylabel('Densidad de Probabilidad',fontsize = 18)
ax1.set_title('Función Distribución de Probabilidad Excedida',fontsize = 20)
"""
#---------------------- Texto caja de parámetros ----------------------
#----------------------------------------------------------------------
textstr = '\n'.join((
    r'$\Gamma=%.3f$' % (r, ),
    r'$\mathrm{A}=%.1f$' % (A, ),
    r'$\sigma_G^2=%.3f$' % (Sigma_G_sq, )))

props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad = 0.6)

ax1.text(0.8, 0.92, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

ax1.tick_params(labelsize = 12)
#-----------------------------------------------------------------------
"""
#---------------------- Gráfico CDF teórica ----------------------------
fig2,ax2 = plt.subplots(num=2,figsize = (7,5),dpi = 140)
#ax2.plot(X,W,'k',linewidth = 1.5)
ax2.set_xlabel('Potencia (dBW)',fontsize = 18)
ax2.set_ylabel('Distribución de Probabilidad',fontsize = 18)
ax2.set_title('Función Distribución de Probabilidad Excedida',fontsize = 20)

ax2.step(bin_edge[:N+1],cumulative,where = 'post',linewidth = 3)

#---------------------- Texto caja de parámetros ----------------------
#----------------------------------------------------------------------
textstr = '\n'.join((
    r'$\Gamma=%.2f$' % (r, ),
    r'$\mathrm{A}=%.2f$' % (A, ),
    r'$\sigma_G^2=%.3f$' % (Sigma_G_sq, )))

props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad = 0.6)

ax2.text(0.8, 0.92, textstr, transform=ax2.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

ax2.tick_params(labelsize = 12)
#-----------------------------------------------------------------------

#---------------------- Funcion de costo ---------------------
fig3,ax3 = plt.subplots(num=3,figsize = (7,5),dpi = 140)
ax3.plot(X,Fcosto,'k',linewidth = 1)
ax3.set_xlabel('Potencia (dBW)',fontsize = 18)
ax3.set_ylabel('Fcosto',fontsize = 18)
ax3.set_title('Función de Costo',fontsize = 20)
# ----------------------------------------------------------------------------

ax1.grid()
ax2.grid()
ax3.grid()

plt.show()