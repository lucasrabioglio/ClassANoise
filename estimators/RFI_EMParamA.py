import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.special import factorial
import numpy.matlib as np_matlib

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
        #phi  = np.sum(((z**2).reshape(1,N).T)*(a_ij.dot(1/np.array(list(range(K,M+K))).reshape(1,M).T)))
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

N = 5
M = 3
z = np.array([1,2,3,1,2.5])
A_init = 1
K = 0.1
epsilon = 0.1

t = np.arange(K,M+K)
#print ('t')

A_est,NIter = RFI_EMParamA (N, M, z, A_init, K, epsilon)
#print ('A_est:',A_est)



"""        
while (abs ((A_prev - A) / A_prev) > (10^-7) && NIter < 100)    % convergence criterion, incremental error less than 10^-7.
    # Niter < 100 used to avoid infinite loop which can occur due to inconsistent input envelope data or in-accuracies in calculations.
    A_prev = A;

    a_ij = zeros (N, M);
    j = [1:M];
    pi_j = exp(-1*A) * (A.^(j-1)) ./ factorial(j-1);
    for i = 1:N
        for j=1:M
            h_j  = 2 * z(i) * (A + K) / (j - 1 + K) * exp( -1 * (z(i)^2) * (A + K) / (j - 1 + K));
            a_ij(i,j) = pi_j(j) * h_j;
        end
        rtot = sum(a_ij(i,:));
        a_ij(i,:) = a_ij(i,:) ./ rtot;
    end
    beta = sum(a_ij * [0:M-1].');                               
    phi = sum( ((z.^2).') .* (a_ij *  (1./([K:M-1+K].'))));     

    a1 = -1 * N - phi;                                          % Equation 15, [1]
    a2 = a1 * K + beta + N;                                     % Equation 15, [1]
    a3 = beta * K;                                              % Equation 15, [1]
    A = (a2 + sqrt(a2^2 - 4*a1*a3))/(-2*a1);                % Equation 16, [1] (corrected)

    if (A  < 10^-2 / ( 1 + epsilon) )                       % Limiting parameter A_est to the range
        A = 10^-2 / (1 + epsilon);                          % [10^-2 / (1+epsilon), (1+epsilon)]
    else
        if (A > 1 + epsilon)
            A = 1 + epsilon;
        end
    end
    NIter = NIter +1;                                       % Increment Number of Iterations
end

A_est = A;                                                  % Estimate of the impulsive index

return;
"""