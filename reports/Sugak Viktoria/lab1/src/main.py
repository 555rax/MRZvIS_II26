import numpy as np

np.random.seed(42)

C0 = 7
C1 = -7

X_raw = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]], dtype=float)
Y_raw = np.array([[C0], [C1], [C1], [C0]], dtype=float)

RANGE_MIN, RANGE_MAX = -10, 10

def normalize(v):
    return (v - RANGE_MIN) / (RANGE_MAX - RANGE_MIN)

def denormalize(v):
    return v * (RANGE_MAX - RANGE_MIN) + RANGE_MIN

X = normalize(X_raw)
Y = normalize(Y_raw)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(out):
    return out * (1.0 - out)

def closest_class(y):
    return C0 if abs(y - C0) <= abs(y - C1) else C1


class MLP:
    def __init__(self, lr=5.0):
        self.lr = lr
        self.W1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, 2))
        self.W2 = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def forward(self, x):
        x = x.reshape(1, -1)
        self.x = x
        self.a1 = sigmoid(x @ self.W1 + self.b1)
        self.a2 = sigmoid(self.a1 @ self.W2 + self.b2)
        return self.a2

    def backward(self, target):
        target = target.reshape(1, -1)
        error_out = target - self.a2
        delta_out = error_out * sigmoid_deriv(self.a2)
        delta_hid = (delta_out @ self.W2.T) * sigmoid_deriv(self.a1)

        self.W2 += self.lr * (self.a1.T @ delta_out)
        self.b2 += self.lr * delta_out
        self.W1 += self.lr * (self.x.T @ delta_hid)
        self.b1 += self.lr * delta_hid

        return np.sum(error_out ** 2)

    def train(self, X, Y, Ee=0.001, max_epochs=100000):
        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            for i in np.random.permutation(len(X)):
                self.forward(X[i])
                Es += self.backward(Y[i])
            if Es <= Ee:
                return epoch, Es
        return max_epochs, Es

    def predict(self, x):
        return self.forward(np.array(x)).flatten()[0]


class SingleLayerPerceptron:
    def __init__(self, lr=5.0):
        self.lr = lr
        self.W = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b = np.random.uniform(-0.5, 0.5, (1, 1))

    def forward(self, x):
        x = x.reshape(1, -1)
        self.x = x
        self.a = sigmoid(x @ self.W + self.b)
        return self.a

    def backward(self, target):
        target = target.reshape(1, -1)
        error = target - self.a
        delta = error * sigmoid_deriv(self.a)
        self.W += self.lr * (self.x.T @ delta)
        self.b += self.lr * delta
        return np.sum(error ** 2)

    def train(self, X, Y, Ee=0.001, max_epochs=100000):
        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            for i in np.random.permutation(len(X)):
                self.forward(X[i])
                Es += self.backward(Y[i])
            if Es <= Ee:
                return epoch, Es
        return max_epochs, Es

    def predict(self, x):
        return self.forward(np.array(x)).flatten()[0]


def evaluate(model, name):
    print(f"\n{name}")
    correct = 0
    for i in range(len(X)):
        y_pred = denormalize(model.predict(X[i]))
        pred_class = closest_class(y_pred)
        target_class = Y_raw[i][0]
        ok = pred_class == target_class
        correct += ok
        print(f"A={X_raw[i][0]:>3.0f} B={X_raw[i][1]:>3.0f} | "
              f"y_hat={y_pred:>7.3f} | класс {pred_class:>3.0f} | "
              f"ожид {target_class:>3.0f} | {'OK' if ok else 'WRONG'}")
    return correct / len(X)


def run_network(model, A, B):
    y = denormalize(model.predict(normalize(np.array([A, B], dtype=float))))
    return y, closest_class(y)


Ee = 0.001
MAX_EPOCHS = 100000

mlp = MLP()
mlp_epochs, mlp_Es = mlp.train(X, Y, Ee, MAX_EPOCHS)
mlp_acc = evaluate(mlp, "MLP 2-2-1")

slp = SingleLayerPerceptron()
slp_epochs, slp_Es = slp.train(X, Y, Ee, MAX_EPOCHS)
slp_acc = evaluate(slp, "Однослойный персептрон")

print("\nСРАВНЕНИЕ")
print(f"{'Модель':<25}{'Эпох':<10}{'Es':<12}{'Accuracy'}")
print(f"{'MLP 2-2-1':<25}{mlp_epochs:<10}{mlp_Es:<12.6f}{mlp_acc*100:.0f}%")
print(f"{'Однослойный':<25}{slp_epochs:<10}{slp_Es:<12.6f}{slp_acc*100:.0f}%")

print("\nРЕЖИМ ФУНКЦИОНИРОВАНИЯ (MLP)")
for A, B in [(7, 7), (7, -7), (-7, 7), (-7, -7), (8, -6), (0, 0)]:
    y, cls = run_network(mlp, A, B)
    print(f"A={A:>4} B={B:>4} | y_hat={y:>7.3f} | класс {cls}")

print("\nВВОД ПОЛЬЗОВАТЕЛЯ ('q' для выхода)")
while True:
    a_in = input("A: ").strip()
    if a_in.lower() == 'q':
        break
    b_in = input("B: ").strip()
    if b_in.lower() == 'q':
        break
    try:
        y, cls = run_network(mlp, float(a_in), float(b_in))
        print(f"y_hat={y:.3f} | класс {cls}\n")
    except ValueError:
        print("Некорректный ввод\n")