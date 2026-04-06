import numpy as np

def Nelder_Mead_function(A):
    # Funcion de minimizacion utilizando el metodo de Nelder-Mead

    resul = np.zeros(5)
    fn = lambda x: A - x
    Pinit = 5
    for i in range(5):
        print(i)
        P = Pinit/2
        resul = fn(P)
        print(resul)
        Pinit = P
    
    return resul
    