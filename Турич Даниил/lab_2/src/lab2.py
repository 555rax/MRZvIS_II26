import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Параметры варианта №6
# ------------------------------
C0, C1 = -9.0, 6.0
TARGET_ERROR = 0.01
MAX_EPOCHS = 20000
LEARNING_RATE = 0.5
SEEDS = [6, 7, 11, 14, 18]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(s):
    return s * (1.0 - s)


def normalize(v):
    return (v - C0) / (C1 - C0)


def denormalize(v):
    return v * (C1 - C0) + C0


# ------------------------------
# Данные XOR
# ------------------------------
X_raw = np.array([
    [C0, C0],
    [C0, C1],
    [C1, C0],
    [C1, C1]
], dtype=float)

y_raw = np.array([C0, C1, C1, C0], dtype=float).reshape(-1, 1)

# Входы всегда нормализуем (в тексте: xn = (x+9)/15)
X_norm = normalize(X_raw)

# Бинарные метки для BCE: tn = (t+9)/15 -> 0 и 1
y_bin = normalize(y_raw)


# ------------------------------
# Класс MLP 2-2-1
# ------------------------------
class MLP:
    def __init__(self, lr=LEARNING_RATE, seed=6, loss_type='mse'):
        self.lr = lr
        self.loss_type = loss_type
        rng = np.random.RandomState(seed)

        # Инициализация U(-0.1, 0.1)
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

        # Градиенты для разных функций потерь
        if self.loss_type == 'bce':
            # BCE + Sigmoid: dL/dz = p - t
            delta2 = pred - y_true
        else:
            # MSE + Sigmoid: dL/dz = (p - t) * p * (1 - p)
            delta2 = (pred - y_true) * sigmoid_derivative(pred)

        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)

        # Обновление весов (градиентный спуск)
        self.W2 -= self.lr * np.dot(self.a1.T, delta2)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * np.dot(x.T, delta1)
        self.b1 -= self.lr * delta1

    def calc_total_loss(self, X, y):
        # Вычисление суммарной ошибки Es на всех 4 примерах
        pred = self.forward(X)
        if self.loss_type == 'bce':
            p = np.clip(pred, 1e-12, 1 - 1e-12)
            return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            return float(0.5 * np.sum((pred - y) ** 2))

    def fit(self, X, y, max_epochs=MAX_EPOCHS, ee=TARGET_ERROR):
        history = []
        converged = False
        epoch = 0

        for epoch in range(1, max_epochs + 1):
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            for i in indices:
                self.train_step_online(X[i:i + 1], y[i:i + 1])

            es = self.calc_total_loss(X, y)
            history.append(es)

            if es <= ee:
                converged = True
                break

        return {
            "epochs": epoch,
            "Es": es,
            "converged": converged,
            "history": np.asarray(history)
        }


# ------------------------------
# Оценка и инференс
# ------------------------------
def evaluate_model(model, y_true_raw, is_raw_target):
    pred_norm = model.forward(X_norm).ravel()

    if is_raw_target:
        pred_out = pred_norm
    else:
        pred_out = denormalize(pred_norm)

    y_true = y_true_raw.ravel()

    # Ближайший класс
    classes = np.where(np.abs(pred_out - C0) <= np.abs(pred_out - C1), C0, C1)

    accuracy = float(np.mean(classes == y_true))
    mae = float(np.mean(np.abs(pred_out - y_true)))

    return accuracy, mae, pred_out


def inference(model, a, b, is_raw_target=False):
    x_test = normalize(np.array([[a, b]], dtype=float))
    pred_norm = float(model.forward(x_test).ravel()[0])

    if is_raw_target:
        pred_out = pred_norm
    else:
        pred_out = float(denormalize(pred_norm))

    closest_class = C0 if abs(pred_out - C0) <= abs(pred_out - C1) else C1
    return pred_norm, pred_out, closest_class


# ------------------------------
# Запуск экспериментов по 5 сидам
# ------------------------------
configs = [
    ("A: original MSE", 'mse', y_raw, True),
    ("B: BCE", 'bce', y_bin, False),
    ("Scaled MSE diagnostic", 'mse', y_bin, False)
]

results = {name: [] for name, _, _, _ in configs}

print("=" * 80)
print(f"{'Конфигурация':<22} | {'Seed':<4} | {'Эпохи':<6} | {'Es':<8} | {'Accuracy':<8} | {'MAE':<6} | {'Сходимость'}")
print("=" * 80)

for name, loss_type, target, is_raw in configs:
    for seed in SEEDS:
        model = MLP(lr=LEARNING_RATE, seed=seed, loss_type=loss_type)
        fit_info = model.fit(X_norm, target)
        acc, mae, pred_out = evaluate_model(model, y_raw, is_raw)

        results[name].append({
            "model": model,
            "fit": fit_info,
            "acc": acc,
            "mae": mae,
            "pred_out": pred_out
        })

        conv_str = "да" if fit_info['converged'] else "нет"
        print(
            f"{name:<22} | {seed:<4} | {fit_info['epochs']:<6} | {fit_info['Es']:.4f}   | {acc * 100:>6.1f}%  | {mae:.4f} | {conv_str}")
    print("-" * 80)

# ------------------------------
# График 1: Кривые обучения (Seed 6)
# ------------------------------
plt.figure(figsize=(8, 5))
for name in ["A: original MSE", "B: BCE", "Scaled MSE diagnostic"]:
    h = results[name][0]["fit"]["history"]
    plt.plot(range(1, len(h) + 1), h, label=name)

plt.axhline(y=TARGET_ERROR, color='gray', linestyle='--', linewidth=1.5, label='Ee=0.01')
plt.yscale('log')
plt.xlim(0, MAX_EPOCHS)
plt.xlabel('Epoch')
plt.ylabel('Loss (different units)')
plt.title('Convergence; loss magnitudes not directly comparable')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------
# График 2: Устойчивость (Эпохи по сидам)
# ------------------------------
epochs_A = [r["fit"]["epochs"] for r in results["A: original MSE"]]
epochs_B = [r["fit"]["epochs"] for r in results["B: BCE"]]

x = np.arange(len(SEEDS))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, epochs_A, width, label='A: raw MSE', color='tab:blue', hatch='//')
plt.bar(x + width / 2, epochs_B, width, label='B: BCE', color='tab:orange')
plt.axhline(y=MAX_EPOCHS, color='gray', linestyle='--', alpha=0.5)

plt.xticks(x, SEEDS)
plt.xlabel('Seed')
plt.ylabel(f'Epochs ({MAX_EPOCHS} = limit)')
plt.title('Устойчивость по пяти случайным инициализациям')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------
# График 3: Разделяющая поверхность
# ------------------------------
def plot_decision_boundary(ax, model, is_raw, title):
    xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = normalize(grid_raw)

    preds = model.forward(grid_norm).reshape(xx.shape)

    if is_raw:
        Z = preds
        vmin, vmax = C0, C1
    else:
        Z = denormalize(preds)
        vmin, vmax = C0, C1

    contour = ax.contourf(xx, yy, Z, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)

    # Обучающие точки
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw.ravel(), edgecolors='w',
               s=80, cmap='coolwarm', vmin=C0, vmax=C1, zorder=10)

    ax.set_title(title)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    return contour


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
mse_model = results["A: original MSE"][0]["model"]
bce_model = results["B: BCE"][0]["model"]

c1 = plot_decision_boundary(ax1, mse_model, True, "A: original-scale MSE")
c2 = plot_decision_boundary(ax2, bce_model, False, "B: BCE with sigmoid")

fig.colorbar(c2, ax=ax2, label='Output sigmoid value')
plt.tight_layout()
plt.show()

# ------------------------------
# График 4: Сравнение MAE
# ------------------------------
mae_mse = results["A: original MSE"][0]["mae"]
mae_bce = results["B: BCE"][0]["mae"]
mae_scaled = results["Scaled MSE diagnostic"][0]["mae"]

labels = ['MSE raw', 'BCE', 'MSE scaled']
maes = [mae_mse, mae_bce, mae_scaled]

plt.figure(figsize=(7, 5))
plt.bar(labels, maes, color='tab:blue')
plt.ylabel('MAE on original scale')
plt.title('Numeric-scale reconstruction accuracy (seed 6)')
plt.tight_layout()
plt.show()

# ------------------------------
# Режим функционирования: 8 примеров (Для лучшей модели BCE)
# ------------------------------
print("\n" + "=" * 55)
print("Режим функционирования (Модель B: BCE, seed 6):")
print(f"{'A':>4} {'B':>4} | {'y_norm':>8} | {'y_raw':>8} | {'Класс':>6}")
print("-" * 55)

test_pairs = [
    (-9, -9), (-9, 6), (6, -9), (6, 6),
    (-6, 4), (0, 0), (6, 0), (-10, 10)
]

for a, b in test_pairs:
    pred_norm, pred_out, cls = inference(bce_model, a, b, is_raw_target=False)
    print(f"{a:>4} {b:>4} | {pred_norm:>8.4f} | {pred_out:>8.4f} | {cls:>6}")
print("=" * 55)