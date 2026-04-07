from curses import window
from curses.textpad import rectangle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ClassA import RFI_MakeEnvelopeDataClassA

mpl.rc('font',family = 'Times New Roman')

N = 100000
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
valor_medio = np.mean(potencia_log)

# ------------ Figura en escala logarítmica ------------------------------------------------
# ------------------------------------------------------------------------------------------
fig1,ax1 = plt.subplots(num=1,figsize = (7,5),dpi = 140)
#fig1,ax1 = plt.subplots(1)
ax1.plot(potencia_log,'.',linewidth = 1,color = 'tab:blue')
ax1.set_xlabel('Sample',fontsize = 18)
ax1.set_ylabel('X [dBW]',fontsize = 18)
ax1.axhline(y=valor_medio,xmin=0,xmax=N,linestyle = '--', color = 'green', linewidth = 2)
# ------------------------------------------------------------------------------------------


fig2,ax2 = plt.subplots(1)
ax2.plot(potencia,'.',linewidth = 1,color = 'tab:blue')
ax2.set_xlabel('Muestras',fontsize = 18)
ax2.set_ylabel('Potencia [W]',fontsize = 18)

#---------------------- Texto caja de parámetros ----------------------
#----------------------------------------------------------------------
textstr = '\n'.join((
    r'$\Gamma=%.3f$' % (r, ),
    r'$\mathrm{A}=%.1f$' % (A, ),
    r'$\sigma_G^2=%.3f$' % (Sigma_G_sq, )))

props = dict(boxstyle='round', facecolor='white', alpha=0.8, pad = 0.6)

ax1.text(0.8, 0.92, textstr, transform=ax1.transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

#------------------------ Texto caja Zona 1 --------------------------------------
#---------------------------------------------------------------------------------
ax1.text(0.1, 0.05, '$Zone_{1}$',transform=ax1.transAxes, fontsize = 16,
        bbox={'boxstyle':'round', 'facecolor': 'red', 'alpha': 0.2, 'pad': 0.4})

#------------------------ Texto caja Zona 1 --------------------------------------
#---------------------------------------------------------------------------------
ax1.text(0.1, 0.93, '$Zone_{2}$',transform=ax1.transAxes, fontsize = 16,
        bbox={'boxstyle':'round', 'facecolor': 'blue', 'alpha': 0.2, 'pad': 0.4})


#------------------------ Alfa --------------------------------------
#---------------------------------------------------------------------------------
#------------------------- Flecha Alfa -----------------------------------------
ax1.annotate(r'$\alpha$', fontsize = 12, xy=(50, valor_medio), xytext=(5, valor_medio + 10),
                   arrowprops=dict(arrowstyle="->",
                                   connectionstyle="angle3,angleA=0,angleB=-90"),
                                   bbox=dict(boxstyle="round", fc='tab:green',alpha = 0.5,color = 'tab:green'));


#ax1.grid()

plt.show()

plt.psd(env_data_DesNorm,Fs=1e6,window=windows_none(env_data_DesNorm))
plt.grid(True)
plt.show()