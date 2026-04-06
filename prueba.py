import numpy as np

V = np.array([1,2,3,1])
print('V:',V)

minV = min(V)
print('Min:',minV)

pos_minV = np.where(V == minV)
print(type(pos_minV))
print('Posicion minimo:',pos_minV[0])

minimos = pos_minV[0]
print('Posiciones de los minimos:',minimos)
print(type(minimos))

p_min = minimos[0]
print('Primer minimo:',p_min)

print(np.sum((V-2)**2))
