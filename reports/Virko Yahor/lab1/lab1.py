import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(output):
    return output * (1.0 - output)


class MLP:
    def __init__(self, input_size=2, hidden_size=2, output_size=1):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.random.randn(hidden_size) * 0.5
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.random.randn(output_size) * 0.5

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        error = y - output
        delta2 = error * sigmoid_derivative(output)
        error_hidden = np.dot(delta2, self.W2.T)
        delta1 = error_hidden * sigmoid_derivative(self.a1)

        self.W2 += np.dot(self.a1.T, delta2) * lr
        self.b2 += np.sum(delta2, axis=0) * lr
        self.W1 += np.dot(X.T, delta1) * lr
        self.b1 += np.sum(delta1, axis=0) * lr

    def train(self, X, y, epochs=10000, lr=0.5, Ee=0.01):
        error_history = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = X[i:i + 1]
                y_i = y[i:i + 1]
                output = self.forward(x_i)
                self.backward(x_i, y_i, output, lr)
                total_error += np.sum((y_i - output) ** 2)

            error_history.append(total_error)

            if total_error <= Ee:
                return epoch + 1, total_error, error_history
        return epochs, total_error, error_history


class SingleLayerPerceptron:
    def __init__(self, input_size=2, output_size=1):
        self.W = np.random.randn(input_size, output_size) * 0.5
        self.b = np.random.randn(output_size) * 0.5

    def forward(self, X):
        self.z = np.dot(X, self.W) + self.b
        self.output = sigmoid(self.z)
        return self.output

    def backward(self, X, y, output, lr):
        error = y - output
        delta = error * sigmoid_derivative(output)
        self.W += np.dot(X.T, delta) * lr
        self.b += np.sum(delta, axis=0) * lr

    def train(self, X, y, epochs=10000, lr=0.5, Ee=0.01):
        error_history = []
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                x_i = X[i:i + 1]
                y_i = y[i:i + 1]
                output = self.forward(x_i)
                self.backward(x_i, y_i, output, lr)
                total_error += np.sum((y_i - output) ** 2)

            error_history.append(total_error)

            if total_error <= Ee:
                return epoch + 1, total_error, error_history
        return epochs, total_error, error_history


X = np.array([
    [0, 0],
    [0, -6],
    [-6, 0],
    [-6, -6]
], dtype=float)

y_raw = np.array([[0], [-6], [-6], [0]], dtype=float)
y_scaled = (y_raw + 6) / 6

print("Обучение MLP (2-2-1)")
mlp = MLP()
epochs_mlp, error_mlp, history_mlp = mlp.train(X, y_scaled, epochs=2000, lr=0.5, Ee=0.01)
print(f"Эпох до сходимости: {epochs_mlp}")
print(f"Итоговая суммарная ошибка: {error_mlp:.6f}")
predictions_mlp = mlp.forward(X)
acc_mlp = np.mean((predictions_mlp > 0.5) == (y_scaled > 0.5))
print(f"Accuracy: {acc_mlp * 100}%")

print("\nОбучение Однослойного Персептрона")
slp = SingleLayerPerceptron()
epochs_slp, error_slp, history_slp = slp.train(X, y_scaled, epochs=2000, lr=0.5, Ee=0.01)
print(f"Эпох до сходимости: {epochs_slp} (или достигнут максимум)")
print(f"Итоговая суммарная ошибка: {error_slp:.6f}")
predictions_slp = slp.forward(X)
acc_slp = np.mean((predictions_slp > 0.5) == (y_scaled > 0.5))
print(f"Accuracy: {acc_slp * 100}%")

print("\nРежим функционирования (MLP)")
test_inputs = [
    [0, 0], [0, -6], [-6, 0], [-6, -6],
    [-3, -3], [-1, -5], [-5, -1], [1, -1]
]

for a, b in test_inputs:
    inp = np.array([[a, b]], dtype=float)
    pred_scaled = mlp.forward(inp)[0][0]
    pred_raw = pred_scaled * 6 - 6
    pred_class = 0 if pred_scaled < 0.5 else -6
    print(f"Вход: ({a:>3}, {b:>3}) | Выход: {pred_raw:>6.2f} | Класс: {pred_class}")



plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')

epochs_range_mlp = range(1, len(history_mlp) + 1)
epochs_range_slp = range(1, len(history_slp) + 1)

ax.plot(epochs_range_mlp, history_mlp, color='#1f77b4', linewidth=2.5, label='MLP (2-2-1)')
ax.plot(epochs_range_slp, history_slp, color='#d62728', linewidth=2.5, label='Однослойный персептрон')

ax.fill_between(epochs_range_mlp, history_mlp, alpha=0.15, color='#1f77b4')
ax.fill_between(epochs_range_slp, history_slp, alpha=0.15, color='#d62728')

ax.axhline(y=0.01, color='#2ca02c', linestyle='--', linewidth=2, label='Порог остановки (Ee = 0.01)')

ax.set_yscale('log')

ax.set_title('График сходимости нейронных сетей (Задача XOR)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Номер эпохи', fontsize=13, labelpad=10)
ax.set_ylabel('Суммарная ошибка MSE (логарифмическая шкала)', fontsize=13, labelpad=10)

ax.legend(fontsize=12, loc='upper right', frameon=True, facecolor='white', framealpha=0.9, shadow=True)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()