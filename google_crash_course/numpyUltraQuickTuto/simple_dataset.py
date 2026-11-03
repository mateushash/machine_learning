import numpy as np

feature = np.arange(6, 21)
print(feature)

label = 3 * feature + 4
print(label)

noise = np.random.randint(low=-2, high=2, size=(21-6,))
#noise = (np.random.random([15]) * 4) - 2
print(noise)

label = label + noise
print(label)
