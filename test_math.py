import numpy as np

a = np.random.randint(10, size=(2, 4, 1))
print(a)

b = np.random.randint(5, size=(2, 4, 3))
print(b)

c = a * b
print(a*b)