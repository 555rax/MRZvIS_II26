import numpy as np
import matplotlib.pyplot as plt

C0 = -9.0
C1 = 6.0
TARGET_ERROR = 0.01
MAX_EPOCHS = 10000
LEARNING_RATE = 0.5


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(s):
    return s * (1.0 - s)

def normalize(val, c_min=C0, c_max=C1):

    return (val - c_min) / (c_max - c_min)


def denormalize(val, c_min=C0, c_max=C1):
    return val * (c_max - c_min) + c_min


X_raw = np.array([
    [C0, C0],
    [C0, C1],
    [C1, C0],
    [C1, C1]
])
y_raw = np.array([C0, C1, C1, C0]).reshape(-1, 1)

X_norm = normalize(X_raw)
y_norm = normalize(y_raw)



class MLP:
    def __init__(self, lr):
        self.lr = lr

        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(1, 2) * 0.5
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.5

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step_online(self, x, y_true):

        pred = self.forward(x)


        error = y_true - pred
        delta2 = error * sigmoid_derivative(pred)


        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)


        self.W2 += self.lr * np.dot(self.a1.T, delta2)
        self.b2 += self.lr * delta2
        self.W1 += self.lr * np.dot(x.T, delta1)
        self.b1 += self.lr * delta1

        return 0.5 * np.sum(error ** 2)

    def calc_total_error(self, X, y):
        total_err = 0.0
        for i in range(len(X)):
            pred = self.forward(X[i:i + 1])
            total_err += 0.5 * np.sum((y[i:i + 1] - pred) ** 2)
        return total_err


class SLP:
    def __init__(self, lr):
        self.lr = lr
        self.W = np.random.randn(2, 1) * 0.5
        self.b = np.random.randn(1, 1) * 0.5

    def forward(self, x):
        self.z = np.dot(x, self.W) + self.b
        self.a = sigmoid(self.z)
        return self.a

    def train_step_online(self, x, y_true):
        pred = self.forward(x)
        error = y_true - pred
        delta = error * sigmoid_derivative(pred)

        self.W += self.lr * np.dot(x.T, delta)
        self.b += self.lr * delta

        return 0.5 * np.sum(error ** 2)

    def calc_total_error(self, X, y):
        total_err = 0.0
        for i in range(len(X)):
            pred = self.forward(X[i:i + 1])
            total_err += 0.5 * np.sum((y[i:i + 1] - pred) ** 2)
        return total_err


print("Запуск обучения многослойного персептрона (MLP 2-2-1)...")
mlp = MLP(lr=LEARNING_RATE)
mlp_history = []
mlp_converged = False

for epoch in range(MAX_EPOCHS):
    indices = np.arange(len(X_norm))
    np.random.shuffle(indices)

    epoch_error = 0.0
    for i in indices:
        x_sample = X_norm[i:i + 1]
        y_sample = y_norm[i:i + 1]
        epoch_error += mlp.train_step_online(x_sample, y_sample)

    mlp_history.append(epoch_error)

    if epoch_error <= TARGET_ERROR:
        print(f"✓ MLP: Сходимость достигнута на эпохе {epoch + 1}. Итоговая ошибка: {epoch_error:.5f}")
        mlp_converged = True
        break

if not mlp_converged:
    print(f"✗ MLP: Не сошелся за {MAX_EPOCHS} эпох. Итоговая ошибка: {mlp_history[-1]:.5f}")

print("\nЗапуск обучения однослойного персептрона (SLP 2-1)...")
slp = SLP(lr=LEARNING_RATE)
slp_history = []
slp_converged = False

for epoch in range(MAX_EPOCHS):
    indices = np.arange(len(X_norm))
    np.random.shuffle(indices)

    epoch_error = 0.0
    for i in indices:
        x_sample = X_norm[i:i + 1]
        y_sample = y_norm[i:i + 1]
        epoch_error += slp.train_step_online(x_sample, y_sample)

    slp_history.append(epoch_error)

    if epoch_error <= TARGET_ERROR:
        print(f"✓ SLP: Сходимость достигнута на эпохе {epoch + 1}. Итоговая ошибка: {epoch_error:.5f}")
        slp_converged = True
        break

if not slp_converged:
    print(f"✗ SLP: Не сошелся за {MAX_EPOCHS} эпох. Итоговая ошибка: {slp_history[-1]:.5f}")


def evaluate_network(model, test_cases, model_name):
    print(f"\n--- Проверка работы: {model_name} ---")
    print(f"{'Вход (A, B)':<15} | {'Выход ŷ':<10} | {'Ближайший класс'}")
    print("-" * 48)

    for a, b in test_cases:
        x_test = normalize(np.array([[a, b]]))
        out_norm = model.forward(x_test)
        out_real = denormalize(out_norm)[0][0]

        dist_c0 = abs(out_real - C0)
        dist_c1 = abs(out_real - C1)
        closest_class = C0 if dist_c0 < dist_c1 else C1

        print(f"({a:>3}, {b:>3})        | {out_real:>8.2f}   | {closest_class}")


test_inputs = [
    (-9, -9), (-9, 6), (6, -9), (6, 6),
    (0, 0), (-5, 3), (2, -7), (6, 0)
]

evaluate_network(mlp, test_inputs, "Многослойный персептрон (MLP)")
evaluate_network(slp, test_inputs, "Однослойный персептрон (SLP)")

plt.figure(figsize=(10, 5))
plt.plot(mlp_history, label='MLP (2-2-1)', color='blue', linewidth=2)
plt.plot(slp_history, label='SLP (2-1)', color='red', linestyle='dashed', linewidth=2)
plt.axhline(y=TARGET_ERROR, color='green', linestyle=':', linewidth=1.5, label=f'Порог ошибки ({TARGET_ERROR})')

plt.title('Динамика суммарной ошибки обучения')
plt.xlabel('Эпоха')
plt.ylabel('Суммарная ошибка (Es)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(2000, MAX_EPOCHS))
plt.show()