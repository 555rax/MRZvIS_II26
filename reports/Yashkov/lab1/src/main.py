import numpy as np

c0, c1 = -3, -8

X = np.array([
    [-3, -3],
    [-3, -8],
    [-8, -3],
    [-8, -8]
], dtype=float)

Y_raw = np.array([
    [-3],
    [-8],
    [-8],
    [-3]
], dtype=float)


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def denormalize(y, xmin, xmax):
    return xmin + y * (xmax - xmin)


Xn = normalize(X, c0, c1)
Y = normalize(Y_raw, c0, c1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(y):
    return y * (1.0 - y)


class MLP221:

    def __init__(self, lr=0.5, seed=7):
        rng = np.random.RandomState(seed)

        self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
        self.T1 = rng.uniform(-0.1, 0.1, (1, 2))

        self.W2 = rng.uniform(-0.1, 0.1, (2, 1))
        self.T2 = rng.uniform(-0.1, 0.1, (1, 1))

        self.lr = lr

    def forward(self, x):
        x = x.reshape(1, -1)

        y1 = sigmoid(x @ self.W1 - self.T1)
        y2 = sigmoid(y1 @ self.W2 - self.T2)

        return y1, y2

    def train_step(self, x, e):
        x = x.reshape(1, -1)
        e = e.reshape(1, -1)

        y1, y2 = self.forward(x)

        delta2 = (y2 - e) * dsigmoid(y2)
        delta1 = (delta2 @ self.W2.T) * dsigmoid(y1)

        self.W2 -= self.lr * (y1.T @ delta2)
        self.T2 += self.lr * delta2

        self.W1 -= self.lr * (x.T @ delta1)
        self.T1 += self.lr * delta1

        return 0.5 * np.sum((y2 - e) ** 2)

    def predict(self, x):
        return self.forward(x)[1][0, 0]

    def fit(self, X, Y, Ee=0.01, max_epochs=20000, seed=42):
        rng = np.random.RandomState(seed)

        for epoch in range(1, max_epochs + 1):

            idx = rng.permutation(len(X))
            Es = 0

            for i in idx:
                Es += self.train_step(X[i], Y[i])

            if Es <= Ee:
                return epoch, Es

        return max_epochs, Es


class SingleLayerPerceptron:

    def __init__(self, lr=0.5, seed=7):
        rng = np.random.RandomState(seed)

        self.W = rng.uniform(-0.1, 0.1, (2, 1))
        self.T = rng.uniform(-0.1, 0.1, (1, 1))

        self.lr = lr

    def forward(self, x):
        x = x.reshape(1, -1)
        return sigmoid(x @ self.W - self.T)

    def train_step(self, x, e):
        x = x.reshape(1, -1)
        e = e.reshape(1, -1)

        y = self.forward(x)

        delta = (y - e) * dsigmoid(y)

        self.W -= self.lr * (x.T @ delta)
        self.T += self.lr * delta

        return 0.5 * np.sum((y - e) ** 2)

    def predict(self, x):
        return self.forward(x)[0, 0]

    def fit(self, X, Y, Ee=0.01, max_epochs=20000, seed=42):
        rng = np.random.RandomState(seed)

        for epoch in range(1, max_epochs + 1):

            idx = rng.permutation(len(X))
            Es = 0

            for i in idx:
                Es += self.train_step(X[i], Y[i])

            if Es <= Ee:
                return epoch, Es

        return max_epochs, Es


def accuracy(model, X, Y):
    correct = 0

    for x, y in zip(X, Y):

        prediction = model.predict(x)

        predicted_class = int(prediction >= 0.5)
        correct_class = int(y[0])

        if predicted_class == correct_class:
            correct += 1

    return correct / len(X)


def show_results(model, name):

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print(
        f"{'A':>8}"
        f"{'B':>8}"
        f"{'Ожидается':>15}"
        f"{'Выход':>15}"
        f"{'Класс':>12}"
    )

    print("-" * 60)

    for x_raw, x_norm, y_raw in zip(X, Xn, Y_raw):

        y_norm = model.predict(x_norm)

        y_hat = denormalize(
            y_norm,
            c0,
            c1
        )

        nearest = c0 if y_norm < 0.5 else c1

        print(
            f"{x_raw[0]:>8.0f}"
            f"{x_raw[1]:>8.0f}"
            f"{y_raw[0]:>15.0f}"
            f"{y_hat:>15.4f}"
            f"{nearest:>12}"
        )


def run_inference(model, A, B, name):

    x = normalize(
        np.array([A, B], dtype=float),
        c0,
        c1
    )

    y_norm = model.predict(x)

    y_hat = denormalize(
        y_norm,
        c0,
        c1
    )

    nearest = c0 if y_norm < 0.5 else c1

    print(
        f"[{name}] "
        f"A={A}, B={B} -> "
        f"ŷ={y_hat:.3f} -> "
        f"ближе к классу {nearest}"
    )


if __name__ == "__main__":

    print("=" * 60)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Вариант 10: c0 = -3, c1 = -8")
    print("=" * 60)

    print("""
Таблица истинности:

 A       B       A XOR B
-3      -3         -3
-3      -8         -8
-8      -3         -8
-8      -8         -3
""")

    mlp = MLP221()
    mlp_epochs, mlp_Es = mlp.fit(Xn, Y)

    slp = SingleLayerPerceptron()
    slp_epochs, slp_Es = slp.fit(Xn, Y)

    mlp_acc = accuracy(mlp, Xn, Y)
    slp_acc = accuracy(slp, Xn, Y)

    print("=" * 60)
    print("РЕЗУЛЬТАТЫ ОБУЧЕНИЯ")
    print("=" * 60)

    print(
        f"\nMLP 2-2-1:"
        f"\nКоличество эпох: {mlp_epochs}"
        f"\nСуммарная ошибка Es: {mlp_Es:.5f}"
        f"\nAccuracy: {mlp_acc:.2f}"
    )

    print(
        f"\nОднослойный персептрон:"
        f"\nКоличество эпох: {slp_epochs}"
        f"\nСуммарная ошибка Es: {slp_Es:.5f}"
        f"\nAccuracy: {slp_acc:.2f}"
    )

    show_results(
        mlp,
        "MLP 2-2-1"
    )

    show_results(
        slp,
        "ОДНОСЛОЙНЫЙ ПЕРСЕПТРОН"
    )

    print()
    print("=" * 60)
    print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ")
    print("=" * 60)

    print("Введите два числа A и B из диапазона [-10; 10].")
    print("Например: -3 -8")
    print("Чтобы выйти, нажмите Enter.")

    while True:

        raw = input(
            "\nВведите A B (пусто — выход): "
        ).strip()

        if not raw:
            print("Программа завершена.")
            break

        try:

            A, B = map(float, raw.split())

            if not (-10 <= A <= 10 and -10 <= B <= 10):
                print("Числа должны быть в диапазоне [-10; 10].")
                continue

            run_inference(
                mlp,
                A,
                B,
                "MLP"
            )

            run_inference(
                slp,
                A,
                B,
                "SLP"
            )

        except ValueError:
            print("Введите два числа через пробел.")