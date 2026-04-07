import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.special import factorial
import numpy.matlib as np_matlib
from estimators.RFI_EMParamK_AFixed import RFI_EMParamK_AFixed
from estimators.RFI_EMCalculateObjFunc import RFI_EMCalculateObjFunc
from scipy.io import loadmat

from core.ClassA import RFI_MakeEnvelopeDataClassA
from core.func_min import func_min
from core.momento import momento

def Est_Momentos_Kanemoto(env_data):
    """
    Estima los valores de los parametros A y r
    """
    m2 = momento(env_data,2)
    m4 = momento(env_data,4)
    m6 = momento(env_data,6)

    A = 9*(m4 - 2*m2**2)**3/(2*(m6 + 12*m2**3 -9*m2*m4)**2)
    r = 2*m2*(m6 + 12*m2**3 - 9*m2*m4)/(3*(m4 - 2*m2**2)**2) - 1
    K = 3*m2*(m4 - 2*m2**2)/(m6 - 9*m2*m4 + 12*m2**3) - 9*(m4 - 2*m2**2)**3/(2*(m6 - 9*m2*m4 + 12*m2**3)**2)

    return A,r,K

def Est_Momentos_Zabin_Poor(env_data):
    """
    -------------- Estimador de Zabin-Poor ---------------

    -- Utiliza el metodo de Zabin-Poor para realizar la --
    --       estimacion de los parametros A y r         --
    ------------------------------------------------------
       [A_est,K_est] = Est_Momentos_Zabin_Poor(env_data)
    ------------------------------------------------------
    """
    m4 = momento(env_data,4)
    m6 = momento(env_data,6)

    A = (m4/2-1)**3/(m6/6-1.5*m4+2)**2
    K = (m4/2-1)/(m6/6-1.5*m4+2)-A

    return A,K

def Est_Param_ClassA_CDF(env_data,Vi,num_iter,func_eval):
    """
    -------------------------------------------------------------------------------------------------------------------------
    --             Función con la cual se obtiene una estimación de los parámetros de ruido                                --                                               
    --             de Middleton de Clase A a partir de una funcion a minimizar y un vector                                 -- 
    --                         con los puntos iniciales para comenzar la iteración:                                        --
    --       A_est,gauss_noise_est,r_est = Est_Param_ClassA_propio(env_data,vec_ini,num_iter,func_eval,Sigmag2,r_org)      --
    --                                                                                                                     --
    -- Entradas:                                                                                                           --
    --         env_data: Valores de ruido de Middleton de Clase A de envolvente utilizados para la estimación              --
    --          vec_ini: Es el vector fila inicial con el cual se formará el Simplex Inicial                               --
    --         num_iter: Número de iteraciones a realizar por el método hasta alcanzar el mínimo                           --
    --        func_eval: Valor mínimo de la evaluación de la función del punto óptimo para cortar la iteración             --
    --          Sigmag2: Potencia de Ruido Gaussiano o de fondo en W.                                                      --
    --            r_org: Parámetro r(Gamma) original para desnormalizar los datos de entrada env_data                      --
    -------------------------------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------------------------------                                                                                                                     
    --                     La iteración corta si el numero de iteraciones supera num_iter                                  -- 
    --                      o si feval(func,[A_est,gauss_noise_est,r_est]) < a func_eval                                   --
    -------------------------------------------------------------------------------------------------------------------------
    -------------------------------------------------------------------------------------------------------------------------
    -- Salidas:                                                                                                            --
    --                   A_est: Es el valor del parámetro A del ruido de Middleton de Clase A estimado                     --
    --         gauss_noise_est: Es la potencia(en mW) de ruido gaussiano o de fondo estimado                               --
    --                   r_est: Es el valor del parámetro r(Gamma) del ruido de Middleton de Clase A estimado              --
    --                                                                                                                     --
    ------------------ Esta función utiliza el método de Nelder-Mead para minimizar la función objetivo ---------------------
    """

    #-- Calculo de la potencia de ruido --
    potencia = env_data**2/2
    #----------------------------------------------

    #-- Potencia en dBW(respecto a 1W) --
    potencia_log = 10*np.log10(potencia)
    #------------------------------------

    #-- Generación del vector de edges para la función histcounts --
    N = len(env_data)
    minimo = min(potencia_log)
    maximo = max(potencia_log)
    paso = (maximo - minimo)/(N-1)
    #---------------------------------------------------------------

    #----- Cálculo del histograma (CDF) de potencia_log -----
    histo, bin_edge = np.histogram(potencia_log,N+1,(minimo-2*paso,maximo+2*paso),density = True)
    #--------------------------------------------------------

    X_cdf = bin_edge[:N+1]
    CUM = np.cumsum(histo)
    CDF = CUM/max(CUM)
    CDF_inv = 1 - CDF

    V = np.zeros((4,3))
    Vi[1] = 1/Vi[1]
    #V = np.array([(Vi),(Vi[0]*1.05,Vi[1],Vi[2]),(Vi[0],Vi[1]*1.05,Vi[2]),(Vi[0],Vi[1],Vi[2]*1.05)])
    V = np.array([(Vi),(Vi[0]*1.3,Vi[1],Vi[2]),(Vi[0],Vi[1]*1.3,Vi[2]),(Vi[0],Vi[1],Vi[2]*1.3)])

    size = np.shape(V)
    Y = np.zeros(size[0])


    for i in range(size[0]):
        Y[i] = func_min(X_cdf,CDF_inv,V[i])

    Aux = np.max(Y)
    Vord = np.zeros(size)
    Yord = np.zeros(size[0])

    for i in range(size[0]):
        pos_minYaux = np.where(Y == np.min(Y))
        pos_minY = pos_minYaux[0]
        Vord[i,:] = V[pos_minY[0],:]
        Yord[i] = Y[pos_minY[0]]
        Y[pos_minY[0]] = Aux + 1

    iter = 0

    n = 0

    while (iter < num_iter and Yord[0] > func_eval and n<10):
        #print('Iteracion:',iter,'Valor Yord:',Yord[0])
        M = np.array([(sum(Vord[0:3,0])/3),(sum(Vord[0:3,1])/3),(sum(Vord[0:3,2])/3)])
        R = 2*M - Vord[3,:]
        fr = func_min(X_cdf,CDF_inv,R)
        
        if (fr < Yord[1]):
            # CASO 1
            if (Yord[0] < fr):
                Vord[3,:] = R
            else:
                E = 2*R - M
                fe = func_min(X_cdf,CDF_inv,E)
                if (fe < Yord[0]):
                    Vord[3,:] = E
                else:
                    Vord[3,:] = R
        else:
            # CASO 2
            if (fr < Yord[3]):
                Vord[3,:] = R
            else:
                C = (Vord[3,:] + M)/2
                fc = func_min(X_cdf,CDF_inv,C)
                C2 = (M + R)/2
                fc2 = func_min(X_cdf,CDF_inv,C2)
                if (fc2 < fc):
                    C = C2
                    fc = fc2
                if (fc < Yord[3]):
                    Vord[3,:] = C
                else:
                    S = (Vord[0,:] + Vord[3,:])/2
                    Vord[3,:] = S
                    J = (Vord[0,:] + Vord[1,:])/2
                    Vord[1,:] = J
                    H = (Vord[0,:] + Vord[2,:])/2
                    Vord[2,:] = H
        
        for i in range(size[0]):
            Y[i] = func_min(X_cdf,CDF_inv,Vord[i,:])

        Aux = np.max(Y)
        
        for i in range(size[0]):
            posMin = np.argmin(Y)
            V[i] = Vord[posMin]
            Yord[i] = Y[posMin]
            Y[posMin] = Aux + 1

        Vord[:,:] = V[:,:]
        iter += 1
        
        if ((Yord[1] - Yord[0]) < (1E-18)):
            n = n - 1
        else:
            n = 0

        A = V[0,0]
        Sigmag2 = 1/V[0,1]
        r = V[0,2]

    return A,Sigmag2,r,iter

def RFI_EMTwoParamEst(N, M, z, A_init, K_init):
    #-------------------------------------------
    #---- Funcion para estimar los valores de A y K utilizando
    #---- el algoritmo EM ------------------------------------

    # Dummy Initialization
    K_prev = 10
    A_prev = 10
    # Change of variables
    A = A_init
    Kp = K_init
    # Initialization
    NumIter = 0
    # Conversion vector de entrada de 1D a 2D
    #z = z.reshape(1,N)

    while (abs((K_prev - Kp) / K_prev)  + abs((A_prev - A)/A_prev) > 10e-7 and NumIter < 100):   # convergence criterion, incremental error less than 10^-7
        # Niter < 100 used to avoid infinite loop which can occur due to
        # inconsistent input envelope data or in-accuracies in calculations.
        K_prev = Kp
        A_prev = A
        K_possible = []
        A_possible = []

        # Calculate a_ij's as defined in equation (8) in [1]
        a_ij = np.zeros ((N, M))
        j    = np.array(list(range(0,M)))
        pi_j = exp(-A)*(A**j)/factorial(j)
        for i in range(N):
            for j in range(M):
                h_j       = 2*z[i]*(A + Kp)/(j + Kp)*exp(-1*(z[i]**2)*(A + Kp)/(j + Kp))
                a_ij[i,j] = pi_j[j]*h_j
            a_ij[i,:]     = a_ij[i,:]/np.sum(a_ij[i,:])

        beta1 = np.sum((1 + z**2)*a_ij[:,0].T)
        beta2 = np.sum((1 + z**2)*(a_ij[:,1:].dot((1/np.array(list(range(1,M)),dtype=float)).reshape(1,M-1).T)).T)
        beta3 = np.sum((z**2)*a_ij[:,0].T)
        beta4 = np.sum((z**2)*(a_ij[:,1:].dot((((1/np.array(list(range(1,M)),dtype = float))**2).reshape(1,M-1)).T)).T)
        beta5 = np.sum((z**2)*(a_ij[:,1:].dot((1/np.array(list(range(1,M)))).T)).T)

        alpha = np.sum(a_ij*(np_matlib.repmat(np.array(list(range(0,M))),N,1)))

        c1    = (beta4**2)*(N**2) - beta2*beta4*N*(beta5 + N) + beta4*N*((beta5 + N)**2)
        c2    = (-beta2*beta4*(N**2) - beta2*beta3*beta4*N - beta2*beta4*alpha*N) + (2*beta3*beta4*N - beta1*beta4*N + (beta2**2)*N + 2*beta4*alpha*N)*(beta5 + N) - (beta2*N)*((beta5 + N)**2)
        c3    = (beta4*N**3 + 2*beta3*beta4*N**2 - beta1*beta4*N**2 - beta1*beta4*N**2 + 2*beta4*alpha*N**2) + (beta3**2*beta4*N + beta2**2*beta3*N - beta1*beta3*beta4*N - beta1*beta4*alpha*N + 2*beta3*beta4*alpha*N + beta4*alpha**2*N) + (-beta2*N**2 - 3*beta2*beta3*N + 2*beta1*beta2*N - beta2*alpha*N)*(beta5 + N) + (N**2 * beta3*N - beta1*N)*(beta5+N)**2
        c4    = (-2*beta2*beta3*N**2 - 2*beta2*beta3**2*N + 2*beta1*beta2*beta3*N - 2*beta2*beta3*alpha*N) + (2*beta3*N**2 - beta1*N**2 - 3*beta1*beta3*N + 2*beta3**2*N + beta1**2*N - beta1*alpha*N + 2*beta3*alpha*N)*(beta5 + N)
        c5    = (beta3*N**3 - 2*beta1*beta3*N**2 + 2*beta3**2*N**2 + 2*beta3*alpha*N**2) + (beta1**2*beta3*N - 2*beta1*beta3**2*N + beta3**2*N - 2*beta1*beta3*alpha*N + 2*beta3**2*alpha*N + beta3*alpha**2*N)

        r = np.real(np.roots(np.array([c1,c2,c3,c4,c5])))
        r_ind = np.where((r > 9.09e-7) & (r < 1.1e-2) & (np.imag(r) == 0) & (np.real(r) > 0))
        k_r = r[r_ind[0]] 

        j_vec = np_matlib.repmat(np.array(list(range(0,M))),N,1)
        z_vec = np_matlib.repmat((z.reshape(1,N)).T,1,M)

        k_r = np.append(k_r,[9.09e-7,1.1e-2])
        zeta = []
        A_corr = []
        if (len(k_r) > 0):
            for i in range(len(k_r)):
                zeta.append(np.sum(np.divide(z_vec**2*a_ij,j_vec + k_r[i])))
            
            A_corr = np.divide(N*k_r + zeta*k_r - alpha - N - np.sqrt((N*k_r + zeta*k_r - alpha - N)**2 + 4*(N + np.array(zeta))*alpha*k_r),-2*(N + np.array(zeta)))
            indA = np.where((A_corr > 9.09e-3) & (A_corr < 1.1))

            k_r = k_r[indA[0]]
            A_corr = A_corr[indA[0]]
        
        K_possible = np.append(K_possible,k_r)
        A_possible = np.append(A_possible,A_corr)

        # Estimate A at boundary values of K
        K1 = RFI_EMParamK_AFixed(N,M,z,9.09e-3,Kp)   # Estimate A at K = 9.09*10^-3
        K2 = RFI_EMParamK_AFixed(N,M,z,1.1,Kp)      # Estimate A at K = 1.1

        if (len(K1) > 0):
            K_possible = np.append(K_possible,K1)
            A_possible = np.append(A_possible,9.09e-3*np.ones(K1.size))

        if (len(K2) > 0):
            K_possible = np.append(K_possible,K2)
            A_possible = np.append(A_possible,1.1*np.ones(K2.size))
        
        K_possible = np.append(K_possible,[1.1e-2,1.1e-2,9.09e-7,9.09e-7])
        A_possible = np.append(A_possible,[1.1,9.09e-3,1.1,9.09e-3])
        
        j_vec = np_matlib.repmat(np.array(list(range(0,M))),N,1)
        z_vec = np_matlib.repmat((z.reshape(1,N)).T,1,M)

        alpha = np.sum(a_ij*j_vec)

        Obj_func = []
        for i in range(len(K_possible)):
            A_i = A_possible[i]
            K_i = K_possible[i]
            Obj_func.append(RFI_EMCalculateObjFunc(N,M,z,A_i,K_i))
        
        obj_max = max(Obj_func)
        ind     = Obj_func.index(obj_max)

        Kp = K_possible[ind]
        A  = A_possible[ind]
        NumIter = NumIter + 1

        A_est = A
        K_est = Kp

    return A_est,K_est,NumIter

def RFI_EMParamA (N, M, z, A_init, K, epsilon):
    A      = A_init
    A_prev = 10
    NIter  = 0

    while (abs((A_prev - A)/A_prev) > 10e-7 and NIter < 100):
        A_prev = A

        a_ij = np.zeros ((N, M))
        j    = np.array(list(range(0,M)))
        pi_j = exp(-A)*(A**j)/factorial(j)

        for i in range(N):
            for j in range(M):
                h_j       = 2*z[i]*(A + K)/(j + K)*exp(-1*(z[i]**2)*(A + K)/(j + K))
                a_ij[i,j] = pi_j[j]*h_j
            a_ij[i,:]     = a_ij[i,:]/np.sum(a_ij[i,:])

        beta = np.sum(a_ij.dot((np.array(list(range(0,M)),dtype=float)).reshape(1,M).T))
        #phi  = np.sum(((z**2).reshape(1,N).T)*(a_ij.dot(1/np.array(list(range(K,M+K)),dtype=float).reshape(1,M).T)))
        phi  = np.sum(((z**2).reshape(1,N).T)*(a_ij.dot(1/np.arange(K,M+K).reshape(1,M).T)))

        a1 = -1 * N - phi                                         
        a2 = a1 * K + beta + N                                   
        a3 = beta * K                                          
        A = (a2 + np.sqrt(a2**2 - 4*a1*a3))/(-2*a1)

        if (A  < (10e-2 / ( 1 + epsilon))):
            A = 10e-2 / (1 + epsilon)                        
        else:
            if (A > (1 + epsilon)):
                A = 1 + epsilon
   
        NIter = NIter +1; 

    A_est = A

    return A_est,NIter

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
            #potencia_gauss[j] = potencia[i]
            #potencia_log_gauss[j] = potencia_log[i]
            j = j + 1
        else:
            potencia_no_gauss     = np.append(potencia_no_gauss,potencia[i])
            potencia_log_no_gauss = np.append(potencia_log_no_gauss,potencia_log[i]) 
            #potencia_no_gauss[k] = potencia[i]
            #potencia_log_no_gauss[k] = potencia_log[i]
            k = k + 1
    
    valor_medio_no_gauss = np.mean(potencia_log_no_gauss)
    valor_medio_gauss    = np.mean(potencia_log_gauss)

    A_ini       = len(potencia_no_gauss)/N
    Sigmag2_ini = np.mean(potencia_gauss)
    r_ini       = 1/(((10**((valor_medio_no_gauss - valor_medio_gauss)/10)) - 1)*A_ini)

    return A_ini,Sigmag2_ini,r_ini

def est_inicial_densidad(env_data):
    N = len(env_data)

    potencia     = env_data**2/2
    potencia_log = 10*np.log10(potencia)

    valor_medio = np.mean(potencia)
    valor_medio_log = np.mean(potencia_log)

    j = 0
    k = 0

    potencia_gauss        = []
    potencia_log_gauss    = []
    potencia_no_gauss     = []
    potencia_log_no_gauss = []
    potencia_densidad     = []

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

    for i in range(N):
        if ((potencia_log[i] <= valor_medio_no_gauss) and (potencia_log[i] >= valor_medio_gauss)):
            potencia_densidad = np.append(potencia_densidad,potencia[i])

    A_ini       = len(potencia_no_gauss)/N
    Sigmag2_ini = np.mean(potencia_gauss)
    r_ini       = 1/(((10**((valor_medio_no_gauss - valor_medio_gauss)/10)) - 1)*A_ini)

    densidad = len(potencia_densidad)/N

    return A_ini,Sigmag2_ini,r_ini,densidad,valor_medio,valor_medio_gauss,valor_medio_no_gauss,valor_medio_log

"""

A = 0.2
r = 0.001
M = 10
N = 5000
Sigmag2 = 0.01

# A = 0.02
# r = 0.5
# M = 10
# N = 3000
# Sigmag2 = 0.001
# -----------------------------------

# Inicializacion del vector que contendra las muestras de Ruido de Envolvente de Class A
#-------------------------------
env_data = np.zeros(N)
# -------------------------------

# Llamado a funcion para generar las muestras de ruido de envolvente
# ----------------------------------------------------
env_data = RFI_MakeEnvelopeDataClassA(A,r,M,N,Sigmag2)

matlab = loadmat("/home/lucas/Escritorio/matlab2.mat")
z_matlab = matlab["env_data_DesNorm"]
z_matlab = z_matlab[0]

env_data = z_matlab

A_init = 0.05
K_init = 1

A_est_Luc,Sigmag2_est_Luc,r_est_Luc,Numiter_Luc = Est_Param_ClassA_CDF(env_data,[0.5,0.012,0.0015],1000,0.0001)
A_est_Kane,r_est_Kane,K_est_Kane = Est_Momentos_Kanemoto(env_data)
A_est_Zabin,K_est_Zabin = Est_Momentos_Zabin_Poor(env_data)
A_est_EM,K_est_EM,NumIter = RFI_EMTwoParamEst(N, M, env_data, A_init, K_init)

print('A_est_Luc:',A_est_Luc,'Sigmag2_est_Luc:',Sigmag2_est_Luc,'r_est_Luc:',r_est_Luc,'Num Iter:',Numiter_Luc)
print('A_est_Kane:',A_est_Kane,'r_est_Kane:',r_est_Kane,'K_est_Kane:',K_est_Kane)
print('A_est_Zabin:',A_est_Zabin,'K_est_Zabin:',K_est_Zabin)
print('A_est_EM:',A_est_EM,'K_est_EM:',K_est_EM,'Num Iter:',NumIter)

print ('FIN')
"""