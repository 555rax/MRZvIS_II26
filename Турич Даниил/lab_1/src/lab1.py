import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Параметры варианта №6
# ------------------------------
C0, C1 = -9.0, 6.0
TARGET_ERROR = 0.01
MAX_EPOCHS = 20000
LEARNING_RATE = 0.5
SEED = 6


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(s):
    return s * (1.0 - s)


def normalize(v):
    return (v - C0) / (C1 - C0)


def denormalize(v):
    return v * (C1 - C0) + C0


# ------------------------------
# Данные XOR для классов -9 и 6
# ------------------------------
X_raw = np.array([
    [C0, C0],
    [C0, C1],
    [C1, C0],
    [C1, C1]
], dtype=float)

y_raw = np.array([
    [C0],
    [C1],
    [C1],
    [C0]
], dtype=float)

# Входы всегда нормализуем в [0, 1]
X = normalize(X_raw)

# Диагностическая нормализация целей
y_norm = normalize(y_raw)


# ------------------------------
# MLP 2-2-1
# ------------------------------
class MLP:
    def __init__(self, lr=LEARNING_RATE, seed=SEED):
        self.lr = lr
        rng = np.random.RandomState(seed)

        # По условию: веса и смещения из U(-0.1, 0.1)
        self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
        self.b1 = rng.uniform(-0.1, 0.1, (1, 2))
        self.W2 = rng.uniform(-0.1, 0.1, (2, 1))
        self.b2 = rng.uniform(-0.1, 0.1, (1, 1))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step_online(self, x, y_true):
        pred = self.forward(x)

        # Для MSE: dL/dp = (p - t), но в коде удобнее error = t - p
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


# ------------------------------
# Однослойный персептрон 2-1
# ------------------------------
class SLP:
    def __init__(self, lr=LEARNING_RATE, seed=SEED):
        self.lr = lr
        rng = np.random.RandomState(seed)

        self.W = rng.uniform(-0.1, 0.1, (2, 1))
        self.b = rng.uniform(-0.1, 0.1, (1, 1))

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


# ------------------------------
# Обучение online
# ------------------------------
def train_model(model, X_train, y_train, max_epochs=MAX_EPOCHS,
                target_error=TARGET_ERROR, shuffle_seed=42):
    rng = np.random.RandomState(shuffle_seed)
    history = []
    converged = False
    epoch = 0

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(X_train))

        for i in order:
            model.train_step_online(X_train[i:i + 1], y_train[i:i + 1])

        # Суммарная ошибка на всех 4 примерах после эпохи
        es = model.calc_total_error(X_train, y_train)
        history.append(es)

        if es <= target_error:
            converged = True
            break

    return {
        "epochs": epoch,
        "Es": es,
        "converged": converged,
        "history": np.asarray(history)
    }


# ------------------------------
# Оценка качества
# ------------------------------
def evaluate_model(model, y_true_raw, use_raw_targets):
    pred_norm = model.forward(X).ravel()

    if use_raw_targets:
        pred_out = pred_norm
    else:
        pred_out = denormalize(pred_norm)

    y_true = y_true_raw.ravel()

    classes = np.where(
        np.abs(pred_out - C0) <= np.abs(pred_out - C1),
        C0,
        C1
    )

    accuracy = float(np.mean(classes == y_true))
    mae = float(np.mean(np.abs(pred_out - y_true)))

    return {
        "accuracy": accuracy,
        "mae": mae,
        "pred_out": pred_out,
        "pred_norm": pred_norm,
        "classes": classes
    }


def inference(model, a, b, use_raw_targets=False):
    x_test = normalize(np.array([[a, b]], dtype=float))
    pred_norm = float(model.forward(x_test).ravel()[0])

    if use_raw_targets:
        pred_out = pred_norm
    else:
        pred_out = float(denormalize(pred_norm))

    closest_class = C0 if abs(pred_out - C0) <= abs(pred_out - C1) else C1

    return pred_norm, pred_out, closest_class


# ------------------------------
# Обучение четырёх конфигураций
# ------------------------------
configs = [
    ("MLP raw targets", MLP(seed=SEED), y_raw, True),
    ("Single raw targets", SLP(seed=SEED), y_raw, True),
    ("MLP scaled diagnostic", MLP(seed=SEED), y_norm, False),
    ("Single scaled diagnostic", SLP(seed=SEED), y_norm, False),
]

results = {}

print("Конфигурация                 | Эпохи | Es          | Accuracy | MAE     | Es<=Ee")
print("-" * 82)

for name, model, target, use_raw in configs:
    fit = train_model(model, X, target)
    quality = evaluate_model(model, y_raw, use_raw)

    results[name] = {
        "model": model,
        "fit": fit,
        "quality": quality,
        "use_raw": use_raw
    }

    print(
        f"{name:<28} | {fit['epochs']:>5} | "
        f"{fit['Es']:>11.6f} | "
        f"{quality['accuracy'] * 100:>7.1f}% | "
        f"{quality['mae']:>7.4f} | "
        f"{'да' if fit['converged'] else 'нет'}"
    )


# ------------------------------
# Режим функционирования: 8 примеров
# ------------------------------
test_pairs = [
    (-9, -9), (-9, 6), (6, -9), (6, 6),
    (-6, 4), (0, 0), (6, 0), (-10, 10)
]

mlp_scaled = results["MLP scaled diagnostic"]["model"]

print("\nРежим функционирования MLP scaled diagnostic:")
print(f"{'A':>5} {'B':>5} | {'y_norm':>9} | {'y_raw':>9} | {'class':>6}")
print("-" * 48)

for a, b in test_pairs:
    pred_norm, pred_out, cls = inference(
        mlp_scaled, a, b, use_raw_targets=False
    )
    print(
        f"{a:>5} {b:>5} | "
        f"{pred_norm:>9.4f} | "
        f"{pred_out:>9.4f} | "
        f"{cls:>6}"
    )


# ------------------------------
# График 1: сходимость MLP
# ------------------------------
mlp_raw_hist = results["MLP raw targets"]["fit"]["history"]
mlp_scaled_hist = results["MLP scaled diagnostic"]["fit"]["history"]

plt.figure(figsize=(10, 5))
plt.plot(mlp_raw_hist, label="Raw-target MSE")
plt.plot(mlp_scaled_hist, label="Normalized-target MSE")
plt.axhline(
    y=TARGET_ERROR,
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label="Ee = 0.01"
)
plt.yscale("log")
plt.title("MLP convergence: direct vs normalized targets")
plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.xlim(0, MAX_EPOCHS)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------
# График 2: восстановление исходной шкалы
# ------------------------------
pred_scaled = results["MLP scaled diagnostic"]["quality"]["pred_out"]

labels = ["(-9,-9)", "(-9,6)", "(6,-9)", "(6,6)"]
x_pos = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 5))
plt.bar(x_pos - width / 2, y_raw.ravel(), width, label="Target")
plt.bar(x_pos + width / 2, pred_scaled, width, label="MLP prediction")

plt.title("Scaled MSE predictions (seed 6)")
plt.ylabel("Original scale")
plt.xticks(x_pos, labels)
plt.grid(True, axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()