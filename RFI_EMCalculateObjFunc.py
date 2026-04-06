import numpy as np
from math import *
from scipy.special import factorial
import numpy.matlib as np_matlib

def RFI_EMCalculateObjFunc (N, M, z, A, K):
    """ [obj_func] = RFI_EMCalculateObjFunc (N, M, z, A, K)
        Returns the value of the objective function derived by the Expectation
        maximization algorithm in [1] for Middleton Class A model
        Inputs:  N - Length of the input data vector
                 M - Number of terms in the infinite summation of the envelope
                     density to consider in the calculations. M = 10 terms were
                     observed to be sufficient to capture most information about the
                     Class A model in most cases.
                 z - Vector of observed envelope values for the noise data
                     assumed to follow the Middleton Class A model.
                A - Value of the parameter A (impulsive index)
                K - Value of the parameter K (= impulsive index * Gaussian factor)
                    (see [1] for details)
    Outputs: onj_func  - Value of the objective function based on equation
                         (9) in [1]

    References:
    [1] S. M. Zabin and H. V. Poor, �Efficient estimation of Class A noise parameters
         via the EM [Expectation-Maximization] algorithms�, IEEE Transaction on Information
         Theory, vol. 37, no. 1, pp. 60-72, Jan. 1991.

    Copyright (c) The University of Texas
    Please see the file Copyright.txt that came with this release for details
    Programmers: Kapil Gulati   (gulati@ece.utexas.edu)
                 Arvind Sujeeth (arvind.sujeeth@mail.utexas.edu)
    """
    a_ij = np.zeros ((N, M))

    # Calculate a_ij's as defined in equation (8) in [1]
    j    = np.array(list(range(0,M)))
    pi_j = exp(-A)*(A**j)/factorial(j)

    for i in range(N):
        for j in range(M):
            h_j       = 2*z[i]*(A + K)/(j + K)*exp(-1*(z[i]**2)*(A + K)/(j + K))
            a_ij[i,j] = pi_j[j]*h_j

        a_ij[i,:]     = a_ij[i,:]/np.sum(a_ij[i,:])

    j_m1_fact = []
    for i in range(M):
        j_m1_fact.append(factorial(i))

    j_vec        = np_matlib.repmat(np.array(list(range(0,M))),N,1)
    jm1_fact_vec = np.matlib.repmat(j_m1_fact,N,1)
    z_vec = np_matlib.repmat((z.reshape(1,N)).T,1,M)

    # Objective function (without removing data/parameter independent terms) - Equation (9) in [1]
    Q1 = a_ij*(-1*A + j_vec*np.log(A) - np.log(jm1_fact_vec)) + a_ij*(np.log(2*z_vec) + np.log(np.divide(A+K,j_vec+K)) - np.divide(z_vec**2*(A+K),j_vec+K))

    obj_func = np.sum(Q1)

    return obj_func