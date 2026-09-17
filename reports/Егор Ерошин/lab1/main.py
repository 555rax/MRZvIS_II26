import numpy as np

c0 = -9.0
c1 = -8.0

LEARNING_RATE = 1.0
MAX_EPOCHS = 50000
TARGET_ERROR = 0.005

X_raw = np.array([
    [-9.0, -9.0],
    [-9.0, -8.0],
    [-8.0, -9.0],
    [-8.0, -8.0]
], dtype=float)

y_raw = np.array([
    [-9.0],
    [-8.0],
    [-8.0],
    [-9.0]
], dtype=float)


def normalize_x(X):
    return (X - c0) / (c1 - c0)


def normalize_y(y):
    return (y - c0) / (c1 - c0) * 0.8 + 0.1


def denormalize_y(y_norm):
    return c0 + (y_norm - 0.1) / 0.8 * (c1 - c0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


class SingleLayerPerceptron:
    def __init__(self, input_dim=2):
        np.random.seed(42)
        self.W = np.random.uniform(-0.5, 0.5, (input_dim, 1))
        self.b = np.random.uniform(-0.5, 0.5, (1, 1))

    def forward(self, x):
        return sigmoid(np.dot(x, self.W) + self.b)

    def train_online(self, X, y, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, target_error=TARGET_ERROR):
        for epoch in range(1, max_epochs + 1):
            total_error = 0.0

            for i in range(len(X)):
                x_i = X[i:i + 1]
                y_i = y[i:i + 1]

                out = self.forward(x_i)
                err = 0.5 * (y_i[0, 0] - out[0, 0]) ** 2
                total_error += err

                # Градиент ошибки по сигмоиде
                delta = (out - y_i) * out * (1.0 - out)

                self.W -= lr * np.dot(x_i.T, delta)
                self.b -= lr * delta

            if total_error <= target_error:
                return epoch, total_error
        return max_epochs, total_error


class MultilayerPerceptron:
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1):
        np.random.seed(42)
        self.W1 = np.random.uniform(-1.0, 1.0, (input_dim, hidden_dim))
        self.b1 = np.random.uniform(-1.0, 1.0, (1, hidden_dim))
        self.W2 = np.random.uniform(-1.0, 1.0, (hidden_dim, output_dim))
        self.b2 = np.random.uniform(-1.0, 1.0, (1, output_dim))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        return a2, a1

    def train_online(self, X, y, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, target_error=TARGET_ERROR):
        for epoch in range(1, max_epochs + 1):
            total_error = 0.0
            for i in range(len(X)):
                x_i = X[i:i + 1]
                y_i = y[i:i + 1]

                a2, a1 = self.forward(x_i)

                err = 0.5 * (y_i[0, 0] - a2[0, 0]) ** 2
                total_error += err

                delta2 = (a2 - y_i) * a2 * (1.0 - a2)
                delta1 = np.dot(delta2, self.W2.T) * a1 * (1.0 - a1)

                self.W2 -= lr * np.dot(a1.T, delta2)
                self.b2 -= lr * delta2
                self.W1 -= lr * np.dot(x_i.T, delta1)
                self.b1 -= lr * delta1

            if total_error <= target_error:
                return epoch, total_error
        return max_epochs, total_error


X_norm = normalize_x(X_raw)
y_norm = normalize_y(y_raw)

# Обучение SLP
slp = SingleLayerPerceptron()
slp_epochs, slp_err = slp.train_online(X_norm, y_norm)

# Обучение MLP
mlp = MultilayerPerceptron()
mlp_epochs, mlp_err = mlp.train_online(X_norm, y_norm)


def evaluate(model, name):
    correct = 0
    print(f"\n--- Проверка модели {name} ---")
    print("A\tB\tИстинный\tПредсказанный ŷ\tКласс")
    print("-" * 55)
    for i in range(len(X_raw)):
        x_norm_i = normalize_x(X_raw[i:i + 1])
        if isinstance(model, MultilayerPerceptron):
            out_norm, _ = model.forward(x_norm_i)
        else:
            out_norm = model.forward(x_norm_i)

        y_pred = denormalize_y(out_norm[0, 0])
        pred_class = c0 if abs(y_pred - c0) < abs(y_pred - c1) else c1
        target_class = y_raw[i, 0]

        if pred_class == target_class:
            correct += 1

        print(f"{X_raw[i, 0]:.1f}\t{X_raw[i, 1]:.1f}\t{target_class:.1f}\t\t{y_pred:.4f}\t\t{pred_class:.1f}")

    accuracy = (correct / len(X_raw)) * 100.0
    return accuracy


slp_acc = evaluate(slp, "Однослойный персептрон (SLP)")
mlp_acc = evaluate(mlp, "Многослойный персептрон (MLP 2-2-1)")

print("\n" + "=" * 60)
print("СРАВНИТЕЛЬНЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Критерий                       | SLP            | MLP (2-2-1)")
print(f"------------------------------------------------------------")
print(f"Эпох до сходимости              | {slp_epochs:<14} | {mlp_epochs:<14}")
print(f"Достижение критерия остановки   | Нет (Превышен) | Да")
print(f"Итоговая суммарная ошибка (Es)  | {slp_err:<14.6f} | {mlp_err:<14.6f}")
print(f"Точность (Accuracy)            | {slp_acc:<14.1f}%| {mlp_acc:<14.1f}%")


def user_inference_mode(model):
    print("\n" + "=" * 60)
    print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ ОБУЧЕННОЙ СЕТИ (MLP)")
    print("=" * 60)
    print("Введите пару чисел A и B из диапазона [-10, 10]. Для выхода введите 'q'.")

    # Демонстрационные тестовые значения
    test_inputs = [(-9.0, -9.0), (-9.0, -8.0), (-8.5, -8.2), (0.0, 5.0)]

    for A, B in test_inputs:
        x_user = np.array([[A, B]])
        x_norm_user = normalize_x(x_user)
        out_norm, _ = model.forward(x_norm_user)
        y_hat = denormalize_y(out_norm[0, 0])
        predicted_class = c0 if abs(y_hat - c0) < abs(y_hat - c1) else c1

        print(f"Вход: A = {A:.2f}, B = {B:.2f} -> Выход ŷ = {y_hat:.4f} (Ближе к классу {predicted_class:.1f})")


user_inference_mode(mlp)
