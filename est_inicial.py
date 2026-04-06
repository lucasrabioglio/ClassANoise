import numpy as np
from scipy.special import factorial

def est_inicial(env_data):
    N = len(env_data)

    potencia     = env_data**2/2
    potencia_log = 10*np.log10(potencia)

    valor_medio = np.mean(potencia)

    j = 0
    k = 0

    potencia_gauss        = []
    potencia_log_gauss    = []
    potencia_no_gauss     = []
    potencia_log_no_gauss = []

    for i in range(N):
        if (potencia[i] <= valor_medio):
            potencia_gauss     = np.append(potencia_gauss,potencia[i])
            potencia_log_gauss = np.append(potencia_log_gauss,potencia_log[i])
            j = j + 1
        else:
            potencia_no_gauss     = np.append(potencia_no_gauss,potencia[i])
            potencia_log_no_gauss = np.append(potencia_log_no_gauss,potencia_log[i]) 
            k = k + 1
    
    valor_medio_no_gauss = np.mean(potencia_log_no_gauss)
    valor_medio_gauss    = np.mean(potencia_log_gauss)

    A_ini       = len(potencia_no_gauss)/N
    Sigmag2_ini = np.mean(potencia_gauss)
    r_ini       = 1/(((10**((valor_medio_no_gauss - valor_medio_gauss)/10)) - 1)*A_ini)

    return A_ini,Sigmag2_ini,r_ini