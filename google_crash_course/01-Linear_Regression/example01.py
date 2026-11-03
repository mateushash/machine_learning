import matplotlib.pyplot as plt

def model(x):
    return (-4.6*x) + 34

def l2_loss(actual, prediction):
    return (actual-prediction)*(actual-prediction)


x = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]
y = [18, 15, 18, 16, 15, 14, 24]

plt.scatter(x, y, color='blue', marker='o')

plt.title("Linear Regression")
plt.xlabel("Feature")
plt.ylabel("Label")

plt.grid(True)
plt.show()

print(model(2.370))
print(l2_loss(24, model(2.370)))

