import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


X = np.array([[1, 1], [1, 8], [8, 1], [8, 8]])
y = np.array([[1], [8], [8], [1]])

y_norm = (y - y.min()) / (y.max() - y.min())

np.random.seed(42)
W1 = np.random.uniform(-0.5, 0.5, (2, 2))
W2 = np.random.uniform(-0.5, 0.5, (2, 1))
b1 = np.zeros((1, 2))
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 5000
error_threshold = 0.01
mlp_errors = []

for epoch in range(epochs):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    error = y_norm - final_output
    mse = np.mean(error ** 2)
    mlp_errors.append(mse)

    d_output = error * sigmoid_derivative(final_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    if mse < error_threshold:
        print(f"MLP: Сходимость достигнута на эпохе {epoch}, ошибка={mse:.4f}")
        break

final_output_rescaled = final_output * (y.max() - y.min()) + y.min()
print("\n--- Результаты обучения MLP ---")
for i in range(len(X)):
    print(f"Вход: {X[i]} → Выход: {final_output_rescaled[i][0]:.2f}")

W = np.random.uniform(-0.5, 0.5, (2, 1))
b = np.zeros((1, 1))
slp_errors = []

for epoch in range(2000):
    net_input = np.dot(X, W) + b
    output = sigmoid(net_input)

    error = y_norm - output
    mse = np.mean(error ** 2)
    slp_errors.append(mse)

    d_output = error * sigmoid_derivative(output)

    W += X.T.dot(d_output) * lr
    b += np.sum(d_output, axis=0, keepdims=True) * lr

output_rescaled = output * (y.max() - y.min()) + y.min()
print("\n--- Результаты обучения однослойного персептрона ---")
for i in range(len(X)):
    print(f"Вход: {X[i]} → Выход: {output_rescaled[i][0]:.2f}")

def predict(a, b):
    hidden_input = np.dot([a, b], W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)
    rescaled = final_output * (y.max() - y.min()) + y.min()
    return rescaled.item()

print("\n--- Проверка работы сети ---")
test_inputs = [[1, 8], [8, 8], [4, 8], [8, 4]]
for inp in test_inputs:
    result = predict(inp[0], inp[1])
    print(f"Вход: {inp} → Выход сети: {result:.2f}")

plt.plot(mlp_errors, label="MLP")
plt.plot(slp_errors, label="SLP")
plt.xlabel("Эпоха")
plt.ylabel("Ошибка (MSE)")
plt.title("Сравнение ошибок MLP и SLP")
plt.legend()
plt.show()