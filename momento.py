import numpy as np

def momento(Vec,k): return sum(Vec**k)/len(Vec)
"""
# Función que calcula el momento de orden k no centrado del vector Vec
#                resultado = momento(Vec,k)
# Vec: Vector de numeros al cual se pretende calcular el momento de orden k
# k: Entero que determina el orden del momento a calcular
"""