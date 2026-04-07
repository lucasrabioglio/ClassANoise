import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA

mpl.rc('font',family = 'Times New Roman')

N = 1000
#----------------------
r = 0.001
A = 0.5
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

lim_inf = -80
lim_sup = 30
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
Vaux = V1 + V2 + V3 + V4 + V5 + V6
#-------------------------------------------------------

x = np.linspace(minimo-2*paso,maximo+2*paso,N+2)

# Calculo del Histograma del logaritmo de la potencia (PDF experimental)
# ---------------------------------------------------------------------
histo, bin_edge = np.histogram(potencia_log,N+1,density = False)
tt = np.count_nonzero(histo)
step = (maximo - minimo)/tt
print (step)
"""
plt.subplot(1,2,1)
plt.hist(bin_edge[:-1],N,weights=histo)
plt.grid()

plt.subplot(1,2,2)
histo1,bin_edge1 = np.histogram(potencia_log,tt,density=True)
plt.hist(bin_edge1[:-1],tt,weights=histo1)

plt.grid()
plt.show()
"""
histo1,bin_edge1 = np.histogram(potencia_log,tt,density=True)

print (np.cumsum(histo))

bin_edge2 = bin_edge[:N+1]

#----------------- Gráficos -----------------------------
#--------------------------------------------------------
#---------------------- pdf teórica ---------------------
fig1,ax1 = plt.subplots(num=1,figsize = (7,5),dpi = 140)
ax1.plot(X,V,'--r',linewidth = 1)
ax1.set_xlabel('Potencia (dBW)',fontsize = 18)
ax1.set_ylabel('Densidad de Probabilidad',fontsize = 18)
#ax1.set_title('Función Densidad de Probabilidad',fontsize = 20)

ax1.fill_between(X,V0,color="tab:blue",alpha=0.3)
ax1.fill_between(X,Vaux,color="tab:blue",alpha=0.3)

ax1.hist(bin_edge1[:-1],tt,weights=histo1,edgecolor = 'black',linewidth = 0.15, color = 'steelblue')

#ax1.axvline(x=-20.5,ymin=0,ymax=1,linestyle = '--',color = 'green',linewidth = 1)


#---------------------- Texto caja de parámetros ----------------------
#----------------------------------------------------------------------
textstr = '\n'.join((
    r'$\Gamma=%.3f$' % (r, ),
    r'$\mathrm{A}=%.1f$' % (A, ),
    r'$\sigma_G^2=%.3f$' % (Sigma_G_sq, )))

props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad = 0.6)

ax1.text(0.8, 0.92, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

"""
#------------------------ Texto caja Zona 1 --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha Zona 1 -----------------------------------------
ax1.annotate('$Zona_{1}$', fontsize = 14, xy=(-30, 0.02), xytext=(-50, 0.0437),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='red',alpha = 0.2,color = 'red'))
#---------------------------------------------------------------------------------

#------------------------ Texto caja Zona 2 --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha Zona 2 -----------------------------------------
ax1.annotate('$Zona_{2}$', fontsize = 14, xy=(4, 0.028), xytext=(-15.4, 0.0388),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=-90,angleB=0"),
                                   bbox=dict(boxstyle="round", fc='tab:blue',alpha = 0.2,color = 'blue'))
#---------------------------------------------------------------------------------

#------------------------ Alfa --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha Alfa -----------------------------------------
ax1.annotate(r'$\alpha$', xy=(-20.5, 0.06), xytext=(-15, 0.07),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=-90,angleB=0"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'));
#---------------------------------------------------------------------------------
"""
ax1.tick_params(labelsize = 12)
#-----------------------------------------------------------------------

ax1.grid()

plt.show()