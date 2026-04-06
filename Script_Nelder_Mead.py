from Nelder_Mead import Nelder_Mead_function
import numpy as np 

R = np.zeros(5)
P = np.array([5,4,3,2,1])
R = Nelder_Mead_function(P)
print('---------')
print(R)
