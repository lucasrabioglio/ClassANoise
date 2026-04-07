import numpy as np
from math import *
from scipy.special import factorial

def RFI_EMParamK_AFixed(N, M, z, A, Kp):
    # [K_est] = RFI_EMParamK_AFixed(N, M, z, A, Kp)
    # Estimates the parameter K (= impulsive index * gaussian factor) of a Middleton Class A model
    # using the Expectation Maximization (EM) algorithm developed in [1] for
    # a known impulsive index (A)
    # Inputs:  N - Length of the input data vector
    #          M - Number of terms in the infinite summation of the envelope
    #              density to consider in the calculations. M = 10 terms were
    #              observed to be sufficient to capture most information about the
    #              Class A model in most cases.
    #          z - Vector of observed envelope values for the noise data
    #                     assumed to follow the Middleton Class A model.
    #          A - Value (known)  of the parameter A (impulsive index)
    #          K_p - Initial value of the parameter K (= impulsive index * gaussian factor)
    #                   (see [1] for details)
    # Outputs: K_est  - Estimate of the parameter K (= impulsive index * gaussian factor) of the
    #                   Middleton Class A model
    #
    # References:
    # [1] S. M. Zabin and H. V. Poor, �Efficient estimation of Class A noise parameters
    #     via the EM [Expectation-Maximization] algorithms�, IEEE Transaction on Information
    #     Theory, vol. 37, no. 1, pp. 60-72, Jan. 1991.
    #
    # Copyright (c) The University of Texas
    # Please see the file Copyright.txt that came with this release for details
    # Programmers: Kapil Gulati   (gulati@ece.utexas.edu)
    #              Arvind Sujeeth (arvind.sujeeth@mail.utexas.edu)
    #

    a_ij = np.zeros ((N, M))
    # Calculate a_ij's as defined in equation (8) in [1]
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

    # Equation (32) in [1]
    d1 = beta4
    d2 = 2 * A * beta4 - beta2
    d3 = beta4 * (A**2) - beta2*A + beta3 - beta1 + N
    d4 = 2 * beta3 * A - beta1 * A
    d5 = beta3 * (A**2)

    r     = np.roots(np.array([d1,d2,d3,d4,d5]))
    r_ind = np.where((r > 9.09e-7) & (r < 1.1e-2) & (np.imag(r) == 0))
    K_est = np.real(r[r_ind[0]])     # Estimate of K       

    return K_est