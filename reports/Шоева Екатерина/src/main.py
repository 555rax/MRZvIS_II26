import numpy as np
import matplotlib.pyplot as plt

C0 = 5
C1 = -2

RAW_DATA = [
    (C0, C0, C0),
    (C0, C1, C1),
    (C1, C0, C1),
    (C1, C1, C0),
]

LEARNING_RATE = 0.5
EE = 0.01
MAX_EPOCHS = 10000
SEED = 42


def encode_value(x):
    return 1.0 if x == C0 else 0.0


def decode_value(y):
    return C0 if y >= 0.5 else C1


def encode_arbitrary(x):
    return 1.0 if abs(x - C0) < abs(x - C1) else 0.0


def build_training_set():
    X = []
    Y = []
    for a, b, target in RAW_DATA:
        X.append([encode_value(a), encode_value(b)])
        Y.append([encode_value(target)])
    return np.array(X), np.array(Y)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1.0 - y)


class MLP221:
    def __init__(self, lr=LEARNING_RATE, seed=SEED):
        rng = np.random.default_rng(seed)
        self.lr = lr

        self.W1 = rng.uniform(-0.5, 0.5, size=(2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, size=(2,))
        self.W2 = rng.uniform(-0.5, 0.5, size=(2, 1))
        self.b2 = rng.uniform(-0.5, 0.5, size=(1,))

    def forward(self, x):

        hidden_in = x @ self.W1 + self.b1
        hidden_out = sigmoid(hidden_in)

        out_in = hidden_out @ self.W2 + self.b2
        out = sigmoid(out_in)
        return hidden_out, out

    def train_step(self, x, target):

        hidden_out, out = self.forward(x)

        error = target - out
        delta_out = error * sigmoid_derivative(out)

        error_hidden = delta_out @ self.W2.T
        delta_hidden = error_hidden * sigmoid_derivative(hidden_out)

        self.W2 += self.lr * np.outer(hidden_out, delta_out)
        self.b2 += self.lr * delta_out

        self.W1 += self.lr * np.outer(x, delta_hidden)
        self.b1 += self.lr * delta_hidden

        return float(np.sum(error ** 2))

    def predict(self, x):
        _, out = self.forward(np.array(x, dtype=float))
        return float(out[0])

    def train(self, X, Y, ee=EE, max_epochs=MAX_EPOCHS):
        history = []
        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            idx = np.arange(len(X))
            for i in idx:
                Es += self.train_step(X[i], Y[i])
            history.append(Es)
            if Es <= ee:
                return epoch, Es, history
        return max_epochs, Es, history


class SinglePerceptron:
    def __init__(self, lr=LEARNING_RATE, seed=SEED):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W = rng.uniform(-0.5, 0.5, size=(2,))
        self.b = rng.uniform(-0.5, 0.5)

    def forward(self, x):
        net = x @ self.W + self.b
        return sigmoid(net)

    def train_step(self, x, target):
        out = self.forward(x)
        error = target[0] - out
        delta = error * sigmoid_derivative(out)

        self.W += self.lr * delta * x
        self.b += self.lr * delta

        return float(error ** 2)

    def predict(self, x):
        out = self.forward(np.array(x, dtype=float))
        return float(out)

    def train(self, X, Y, ee=EE, max_epochs=MAX_EPOCHS):
        history = []
        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            for i in range(len(X)):
                Es += self.train_step(X[i], Y[i])
            history.append(Es)
            if Es <= ee:
                return epoch, Es, history
        return max_epochs, Es, history


def accuracy(model, X, Y):
    correct = 0
    for i in range(len(X)):
        pred = model.predict(X[i])
        pred_class = 1.0 if pred >= 0.5 else 0.0
        if pred_class == Y[i][0]:
            correct += 1
    return correct / len(X)


def print_truth_table(model, title):
    print(f"\nРезультат работы сети «{title}» на обучающей выборке:")
    print(f"{'A':>5} {'B':>5} {'target':>8} {'y_hat':>10} {'class':>8}")
    for a, b, target in RAW_DATA:
        x = [encode_value(a), encode_value(b)]
        y_hat = model.predict(x)
        cls = decode_value(y_hat)
        print(f"{a:>5} {b:>5} {target:>8} {y_hat:>10.4f} {cls:>8}")


def plot_error_history(mlp_history, sp_history, n_samples,
                        filename="error_plot.png"):
    mlp_mse = np.array(mlp_history) / n_samples
    sp_mse = np.array(sp_history) / n_samples

    plt.figure(figsize=(6, 4.5))
    plt.plot(range(1, len(mlp_mse) + 1), mlp_mse, label="MLP")
    plt.plot(range(1, len(sp_mse) + 1), sp_mse, label="SLP")
    plt.title("Сравнение ошибок MLP и SLP")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (MSE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nГрафик сохранён в файл: {filename}")
    plt.show()


def interactive_mode(model, model_name):
    print(f"\n--- Режим работы обученной сети ({model_name}) ---")
    print("Введите пару чисел A, B из диапазона [-10, 10] (или 'q' для выхода).")
    while True:
        raw = input("Введите A B через пробел: ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            print("Завершение режима ввода.")
            break
        try:
            parts = raw.replace(",", " ").split()
            a, b = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            print("Некорректный ввод. Пример: 3.5 -7")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Значения должны быть в диапазоне [-10, 10].")
            continue

        x = [encode_arbitrary(a), encode_arbitrary(b)]
        y_hat = model.predict(x)
        nearest_class = "c0" if y_hat >= 0.5 else "c1"
        nearest_value = C0 if y_hat >= 0.5 else C1
        print(f"  Вход сети (после кодирования по близости к c0/c1): {x}")
        print(f"  Выход сети y_hat = {y_hat:.4f}")
        print(f"  Ближе к классу {nearest_class} (значение {nearest_value})")


def main():
    X, Y = build_training_set()

    print("Обучающая выборка (закодированная):")
    for x, y, (a, b, t) in zip(X, Y, RAW_DATA):
        print(f"  A={a:>3} B={b:>3} -> вход={x} target={y} (реальный={t})")

    print("\n" + "=" * 60)
    print("Обучение многослойного персептрона (2-2-1)")
    print("=" * 60)
    mlp = MLP221()
    mlp_epochs, mlp_Es, mlp_history = mlp.train(X, Y, ee=EE, max_epochs=MAX_EPOCHS)
    mlp_acc = accuracy(mlp, X, Y)
    converged_mlp = mlp_Es <= EE

    print(f"Эпох потребовалось: {mlp_epochs}"
          f"{' (достигнут критерий Es<=Ee)' if converged_mlp else ' (лимит эпох, критерий НЕ достигнут)'}")
    print(f"Итоговая суммарная ошибка Es = {mlp_Es:.6f}")
    print(f"Точность (accuracy) = {mlp_acc:.2f}")
    print_truth_table(mlp, "MLP 2-2-1")

    print("\n" + "=" * 60)
    print("Обучение однослойного персептрона (без скрытого слоя)")
    print("=" * 60)
    sp = SinglePerceptron()
    sp_epochs, sp_Es, sp_history = sp.train(X, Y, ee=EE, max_epochs=MAX_EPOCHS)
    sp_acc = accuracy(sp, X, Y)
    converged_sp = sp_Es <= EE

    print(f"Эпох потребовалось: {sp_epochs}"
          f"{' (достигнут критерий Es<=Ee)' if converged_sp else ' (лимит эпох, критерий НЕ достигнут)'}")
    print(f"Итоговая суммарная ошибка Es = {sp_Es:.6f}")
    print(f"Точность (accuracy) = {sp_acc:.2f}")
    print_truth_table(sp, "Однослойный персептрон")

    print("\n" + "=" * 60)
    print("Сравнение сетей")
    print("=" * 60)
    print(f"{'Критерий':<35}{'MLP 2-2-1':>15}{'Однослойный':>15}")
    print(f"{'Эпох до сходимости':<35}{mlp_epochs:>15}{sp_epochs:>15}")
    print(f"{'Критерий Es<=Ee достигнут':<35}{str(converged_mlp):>15}{str(converged_sp):>15}")
    print(f"{'Итоговая Es':<35}{mlp_Es:>15.6f}{sp_Es:>15.6f}")
    print(f"{'Accuracy':<35}{mlp_acc:>15.2f}{sp_acc:>15.2f}")


    plot_error_history(mlp_history, sp_history, n_samples=len(X))

    print("\nВыберите сеть для интерактивного режима:")
    print("  1 - MLP 2-2-1")
    print("  2 - Однослойный персептрон")
    choice = input("Ваш выбор (1/2, Enter для пропуска): ").strip()
    if choice == "1":
        interactive_mode(mlp, "MLP 2-2-1")
    elif choice == "2":
        interactive_mode(sp, "Однослойный персептрон")
    else:
        print("Интерактивный режим пропущен.")


if __name__ == "__main__":
    main()
