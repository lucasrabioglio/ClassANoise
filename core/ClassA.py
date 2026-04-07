import numpy as np
import math
from scipy.special import factorial
import numpy.random as random

from core.momento import momento

def RFI_MakeEnvelopeDataClassA(A,r,M,N,Sigmag2):
    """
    # Funcion generadora de muestras de envolvente de Ruido de Middleton de Clase A dados A,r,M,N,Sigmag2
    # INPUTS :
    #          A        : Parametro A del Ruido Clase A
    #          r        : Parametro gamma del Ruido de Clase A
    #          M        : Cantidad de terminos a ser considerados en la sumatoria de la pdf de Ruido de Clase A
    #          N        : Cantidad de muestras a ser generadas
    #          Sigmag2  : Potencia de Ruido Gaussiano o Background Noise
    # OUTPUTS:
    #          env_data : Vector de 1xN muestras de envolvente de Ruido Clase A generadas 
    """
    # Generacion de los pesos de las distintas Gaussianas segun el valor de M
    # --------------------------------------------------
    m_vec = np.array(list(range(0,M)))
    pdf_weights = math.exp(-A)*A**m_vec/factorial(m_vec)
    # --------------------------------------------------

    # Normalizacion del vector de pesos/probabilidades para usar en la funcion random.choice
    # ----------------------------------------
    pdf_weights = pdf_weights/sum(pdf_weights)
    # ----------------------------------------

    # Generacion de vactor con los valores de M segun las probabilidades dadas por pdf_weights para cada valor de M
    # --------------------------------------------------------------
    selectionMat = random.choice(m_vec,N,replace=True,p=pdf_weights)
    #print(selectionMat)
    # --------------------------------------------------------------

    # Inicializacion de los vectores env_data y noise data
    # ----------------------
    env_data = np.zeros(N)
    noise_data = np.zeros(N)
    # ----------------------

    # Inicializacion del vector de desviaciones estandar
    # --------------------
    sigma_sq = np.zeros(M)
    # --------------------

    # Bucle for para la generacion de los valores de envolvente
    # Estos valores son generados de forma Normalizada, luego hay que desnormalizar con respecto al Ruido Gaussiano y r
    # -------------------------------------------------------------------
    for m in range(M):
        sigma_sq[m]        = (m/A + r) / (1 + r)
        #print(sigma_sq)
        inds               = np.where(selectionMat == m)
        inds               = inds[0]
        #print(inds)
        mean               = np.zeros(2)
        #print(mean)
        cov                = sigma_sq[m]*np.identity(2)
        #print(cov)
        dim                = len(inds)
        #print(dim)
        noise_data         = random.multivariate_normal(mean, cov, dim).T
        #print(noise_data)
        env_data[inds]     = (noise_data[0]**2 + noise_data[1]**2)**(1/2)
        #print(env_data)
    # -------------------------------------------------------------------

    # Datos env_data Normalizados
    env_data_Norm = env_data/np.sqrt(np.mean(env_data**2/2))

    # Calculo de constante de desnormalizacion
    # -------------------
    cte = Sigmag2*(1+1/r)
    # -------------------

    # Desnormalizacion
    # --------------------------------
    env_data_DesNorm = math.sqrt(cte)*env_data
    # --------------------------------
    
    return env_data_Norm,env_data_DesNorm
