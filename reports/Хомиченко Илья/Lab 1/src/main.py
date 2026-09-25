import numpy as np

c0, c1 = -4, -9

X = np.array([
    [c0, c0],
    [c0, c1],
    [c1, c0],
    [c1, c1]
], dtype=float)

Y_raw = np.array([
    [c0],
    [c1],
    [c1],
    [c0]
], dtype=float)


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def denormalize(y_norm, xmin, xmax):
    return xmin + y_norm * (xmax - xmin)


Xn = normalize(X, c0, c1)
Y = normalize(Y_raw, c0, c1)


def sigmoid(S):
    S = np.clip(S, -60, 60)
    return 1.0 / (1.0 + np.exp(-S))


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

            Es = sum(
                self.train_step(X[i], Y[i])
                for i in idx
            )

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
        return sigmoid(
            x.reshape(1, -1) @ self.W - self.T
        )

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

            Es = sum(
                self.train_step(X[i], Y[i])
                for i in idx
            )

            if Es <= Ee:
                return epoch, Es

        return max_epochs, Es


def accuracy(model, Xn, Y, thr=0.5):
    correct = 0

    for x, y in zip(Xn, Y):
        pred_class = int(model.predict(x) >= thr)
        true_class = int(y[0] >= thr)

        correct += int(
            pred_class == true_class
        )

    return correct / len(Xn)


def run_inference(model, A, B, c0, c1, name="Model"):
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
        f"[{name}] A={A}, B={B} "
        f"-> y_hat={y_hat:.3f} "
        f"-> closest class {nearest}"
    )

    return y_hat, nearest


if __name__ == "__main__":
    mlp = MLP221()

    mlp_epochs, mlp_Es = mlp.fit(
        Xn,
        Y
    )

    slp = SingleLayerPerceptron()

    slp_epochs, slp_Es = slp.fit(
        Xn,
        Y
    )

    print(
        "MLP 2-2-1:",
        mlp_epochs,
        "epochs,",
        round(mlp_Es, 5),
        "error,",
        "accuracy =",
        accuracy(mlp, Xn, Y)
    )

    print(
        "SLP:",
        slp_epochs,
        "epochs,",
        round(slp_Es, 5),
        "error,",
        "accuracy =",
        accuracy(slp, Xn, Y)
    )

    while True:
        raw = input(
            "Enter A B (empty line to exit): "
        ).strip()

        if not raw:
            break

        A, B = map(
            float,
            raw.split()
        )

        if not (-10 <= A <= 10 and -10 <= B <= 10):
            print("A and B must be in range [-10, 10]")
            continue

        run_inference(
            mlp,
            A,
            B,
            c0,
            c1,
            "MLP"
        )

        run_inference(
            slp,
            A,
            B,
            c0,
            c1,
            "SLP"
        )