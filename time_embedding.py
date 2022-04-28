import numpy as np 

def omega(k):
    return 1/(10000)**(2*k / 128)

def embedding(t):
    vector = []
    for i in range(128):
        if i % 2 == 0:
            vector.append(np.sin(omega(i//2) * t)) 
        else:
            vector.append(np.cos(omega(i//2) * t))
    vector = np.array(vector)
    return vector 
