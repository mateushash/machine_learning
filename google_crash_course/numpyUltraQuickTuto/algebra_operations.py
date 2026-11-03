import numpy as np

random_01 = np.random.random((6,))
print(random_01)

random_23 = random_01 + 2.0
print(random_23)

random_1020 = np.random.randint(low=10, high=21, size=(6,2))
print(random_1020)

random_3060 = random_1020 * 3.0
print(random_3060)
