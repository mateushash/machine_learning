def model_function(x, w, b):
    return w * x + b

def weight_derivative(x, y, w, b):
    n = len(x)
    w_d = (1/n) * sum((model_function(x[i], w, b) - y[i]) * 2 * x[i] for i in range(n))

    return w_d

def bias_derivative(x, y, w, b):
    n = len(x)
    b_d = (2/n) * sum((model_function(x[i], w, b) - y[i]) for i in range(n))

    return b_d

def mse(x, y, w, b):
    n = len(x)
    mse = sum((y[i] - model_function(x[i], w, b))**2 for i in range(n)) / n
    
    return mse

w0 = 0
b0 = 0

x = [3.5, 3.69, 3.44, 3.43, 4.34, 4.42, 2.37]
y = [18, 15, 18, 16, 15, 14, 24]
yo = [model_function(i, w0, b0) for i in x]
learning_rate = 0.01

gradient_list = []

for i in range(122):
    gradient_list.append([i+1, w0, b0, mse(x, y, w0, b0)])
    new_w = w0 - (learning_rate * weight_derivative(x, y, w0, b0))
    new_b = b0 - (learning_rate * bias_derivative(x, y, w0, b0))
    w0 = new_w
    b0 = new_b

for row in gradient_list:
    print(f"Iteration: {row[0]}, Weight: {row[1]:.4f}, Bias: {row[2]:.4f}, MSE: {row[3]:.4f}")
