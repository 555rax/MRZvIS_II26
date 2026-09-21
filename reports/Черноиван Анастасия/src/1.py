import numpy as np

np.random.seed(42)

C0, C1 = -8.0, 3.0

X_raw = np.array([
    [-8.0, -8.0],
    [-8.0,  3.0],
    [ 3.0, -8.0],
    [ 3.0,  3.0],
], dtype=float)

Y_raw = np.array([-8.0, 3.0, 3.0, -8.0], dtype=float)  # A xor B по варианту


def normalize(t):
    return (t - C0) / (C1 - C0)

def denormalize(o):
    return C0 + (C1 - C0) * o

X = normalize(X_raw)
Y = normalize(Y_raw)

EE = 0.01
MAX_EPOCHS = 20000
LR = 5.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_deriv(o):
    return o * (1.0 - o)


class MLP:
    def __init__(self, lr=LR):
        self.lr = lr
        self.W1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.random.uniform(-0.5, 0.5, 2)
        self.W2 = np.random.uniform(-0.5, 0.5, 2)
        self.b2 = np.random.uniform(-0.5, 0.5)

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        h = sigmoid(z1)
        z2 = self.W2 @ h + self.b2
        o = sigmoid(z2)
        return h, o

    def train_epoch(self, X, Y):
        total_error = 0.0
        for i in range(len(X)):
            x = X[i]
            t = Y[i]
            h, o = self.forward(x)

            err = t - o
            total_error += err ** 2

            delta_out = err * sigmoid_deriv(o)
            grad_W2 = delta_out * h
            grad_b2 = delta_out

            delta_hidden = delta_out * self.W2 * sigmoid_deriv(h)
            grad_W1 = np.outer(delta_hidden, x)
            grad_b1 = delta_hidden

            self.W2 += self.lr * grad_W2
            self.b2 += self.lr * grad_b2
            self.W1 += self.lr * grad_W1
            self.b1 += self.lr * grad_b1

        return total_error

    def predict(self, x):
        _, o = self.forward(x)
        return o


def train_mlp(n_restarts=10):
    best = None
    for attempt in range(n_restarts):
        net = MLP()
        for epoch in range(1, MAX_EPOCHS + 1):
            Es = net.train_epoch(X, Y)
            if Es <= EE:
                return net, epoch, Es
        if best is None or Es < best[2]:
            best = (net, MAX_EPOCHS, Es)
    return best


class SingleLayer:
    def __init__(self, lr=LR):
        self.lr = lr
        self.w = np.random.uniform(-0.5, 0.5, 2)
        self.b = np.random.uniform(-0.5, 0.5)

    def forward(self, x):
        z = self.w @ x + self.b
        return sigmoid(z)

    def train_epoch(self, X, Y):
        total_error = 0.0
        for i in range(len(X)):
            x = X[i]
            t = Y[i]
            o = self.forward(x)
            err = t - o
            total_error += err ** 2

            delta = err * sigmoid_deriv(o)
            self.w += self.lr * delta * x
            self.b += self.lr * delta

        return total_error

    def predict(self, x):
        return self.forward(x)


def train_single():
    net = SingleLayer()
    for epoch in range(1, MAX_EPOCHS + 1):
        Es = net.train_epoch(X, Y)
        if Es <= EE:
            return net, epoch, Es
    return net, MAX_EPOCHS, Es

def evaluate(net):
    correct = 0
    for i in range(len(X)):
        o = net.predict(X[i])
        y_pred = denormalize(o)
        target = Y_raw[i]
        pred_class = C0 if abs(y_pred - C0) < abs(y_pred - C1) else C1
        true_class = target
        if pred_class == true_class:
            correct += 1
    return correct / len(X)


def run_inference(net, A, B):
    x = normalize(np.array([A, B], dtype=float))
    o = net.predict(x)
    y_hat = denormalize(o)
    nearest_class = C0 if abs(y_hat - C0) < abs(y_hat - C1) else C1
    return y_hat, nearest_class

if __name__ == "__main__":
    print("=== Многослойный персептрон 2-2-1 ===")
    mlp, mlp_epochs, mlp_es = train_mlp()
    mlp_acc = evaluate(mlp)
    print(f"Эпох до сходимости: {mlp_epochs}")
    print(f"Итоговая суммарная ошибка Es: {mlp_es:.6f}")
    print(f"Accuracy: {mlp_acc * 100:.1f}%")

    print("\n=== Однослойный персептрон ===")
    sl, sl_epochs, sl_es = train_single()
    sl_acc = evaluate(sl)
    print(f"Эпох (макс. {MAX_EPOCHS}, критерий Es<={EE} {'достигнут' if sl_es <= EE else 'НЕ достигнут'}): {sl_epochs}")
    print(f"Итоговая суммарная ошибка Es: {sl_es:.6f}")
    print(f"Accuracy: {sl_acc * 100:.1f}%")

    print("\n=== Режим функционирования обученного MLP (проверочные примеры) ===")
    test_pairs = [(-8, -8), (-8, 3), (3, -8), (3, 3), (-5, 0), (1, -2)]
    for A, B in test_pairs:
        y_hat, cls = run_inference(mlp, A, B)
        print(f"A={A:>4}, B={B:>4} -> y_hat={y_hat:7.3f}, ближайший класс = {cls}")

    print("\n=== Интерактивный режим функционирования ===")
    print("Введите пару чисел A и B из диапазона [-10; 10] (или 'q' для выхода).")
    while True:
        raw = input("A = ")
        if raw.strip().lower() == "q":
            break
        try:
            A = float(raw)
            B = float(input("B = "))
        except ValueError:
            print("Ошибка: введите числа.")
            continue

        if not (-10 <= A <= 10 and -10 <= B <= 10):
            print("Внимание: значения вне диапазона [-10; 10], результат может быть некорректным.")

        y_hat, cls = run_inference(mlp, A, B)
        print(f"ŷ = {y_hat:.3f}  ->  ближе к классу {cls} "
              f"({'c0' if cls == C0 else 'c1'})")