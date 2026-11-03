import numpy as np

one_dimensional_array = np.array([1.2, 2.4, 3.5, 4.7, 6.1])
print(one_dimensional_array)

two_dimensional_array = np.array([[1,2], [3,4], [5,6]])
print(two_dimensional_array)

zeros_array = np.zeros(3)
ones_array = np.ones(3)

print(zeros_array)
print(ones_array)

sq_int = np.arange(5, 12)
print(sq_int)

rand_1_10 = np.random.randint(low=1, high=11, size=(2,2))
print(rand_1_10)

