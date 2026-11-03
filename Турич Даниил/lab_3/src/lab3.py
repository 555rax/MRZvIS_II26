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

# Нормализация входов и целевых меток в диапазон [0, 1]
X_norm = normalize(X_raw)
y_norm = normalize(y_raw)


# ------------------------------
# Класс MLP 2-2-1
# ------------------------------
class MLP:
    def __init__(self, lr=LEARNING_RATE, seed=6, activation='sigmoid'):
        self.lr = lr
        self.activation = activation
        rng = np.random.RandomState(seed)

        # В листинге ЛР3 инициализация от -1 до 1
        self.W1 = rng.uniform(-1.0, 1.0, (2, 2))
        self.b1 = rng.uniform(-1.0, 1.0, (1, 2))
        self.W2 = rng.uniform(-1.0, 1.0, (2, 1))
        self.b2 = rng.uniform(-1.0, 1.0, (1, 1))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1

        if self.activation == 'sigmoid':
            self.a1 = sigmoid(self.z1)
        else:
            self.a1 = np.maximum(0.0, self.z1)  # ReLU

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step_online(self, x, y_true):
        pred = self.forward(x)

        # BCE + Sigmoid на выходе
        delta2 = pred - y_true

        # Градиент скрытого слоя
        if self.activation == 'sigmoid':
            d_hidden = sigmoid_derivative(self.a1)
        else:
            d_hidden = (self.z1 > 0).astype(float)  # Производная ReLU

        delta1 = np.dot(delta2, self.W2.T) * d_hidden

        # Обновление весов
        self.W2 -= self.lr * np.dot(self.a1.T, delta2)
        self.b2 -= self.lr * delta2
        self.W1 -= self.lr * np.dot(x.T, delta1)
        self.b1 -= self.lr * delta1

    def calc_total_loss(self, X, y):
        pred = self.forward(X)
        p = np.clip(pred, 1e-12, 1 - 1e-12)
        # BCE Loss
        return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))

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

    def count_dead_neurons(self, X):
        if self.activation != 'relu':
            return 0
        z1 = np.dot(X, self.W1) + self.b1
        # Если нейрон выдает <= 0 для ВСЕХ 4 примеров, он "мертв"
        return int(np.all(z1 <= 0, axis=0).sum())


# ------------------------------
# Оценка качества
# ------------------------------
def evaluate_model(model):
    pred_norm = model.forward(X_norm).ravel()
    pred_out = denormalize(pred_norm)
    y_true = y_raw.ravel()

    classes = np.where(np.abs(pred_out - C0) <= np.abs(pred_out - C1), C0, C1)
    accuracy = float(np.mean(classes == y_true))
    mae = float(np.mean(np.abs(pred_out - y_true)))

    return accuracy, mae


# ------------------------------
# Запуск экспериментов по 5 сидам
# ------------------------------
activations = ['sigmoid', 'relu']
results = {act: [] for act in activations}

for act in activations:
    print(f"\n{'=' * 80}")
    print(f"Конфигурация: {act.upper()}")
    print(
        f"{'Seed':<5} | {'Эпохи':<6} | {'Es':<8} | {'Accuracy':<8} | {'MAE':<6} | {'Мёртвые ReLU':<12} | {'Сходимость'}")
    print("-" * 80)

    for seed in SEEDS:
        model = MLP(lr=LEARNING_RATE, seed=seed, activation=act)
        fit_info = model.fit(X_norm, y_norm)
        acc, mae = evaluate_model(model)
        dead = model.count_dead_neurons(X_norm)

        results[act].append({
            "model": model,
            "fit": fit_info,
            "acc": acc,
            "mae": mae,
            "dead": dead
        })

        conv_str = "да" if fit_info['converged'] else "нет"
        print(
            f"{seed:<5} | {fit_info['epochs']:<6} | {fit_info['Es']:.4f}   | {acc * 100:>6.1f}%  | {mae:.4f} | {dead:<12} | {conv_str}")

# ------------------------------
# График 1: Кривые обучения (Seed 6)
# ------------------------------
plt.figure(figsize=(8, 5))
for act in activations:
    h = results[act][0]["fit"]["history"]
    plt.plot(range(1, len(h) + 1), h, label=f'Hidden {act}')

plt.axhline(y=TARGET_ERROR, color='gray', linestyle='--', linewidth=1.5, label='Ee=0.01')
plt.yscale('log')
plt.xlim(0, 2500)  # Ограничили ось X для наглядности (как на рисунке в отчете)
plt.xlabel('Epoch')
plt.ylabel('Total BCE')
plt.title('Activation function comparison (seed 6)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------
# График 2: Устойчивость по 5 сидам
# ------------------------------
epochs_sig = [r["fit"]["epochs"] for r in results['sigmoid']]
epochs_relu = [r["fit"]["epochs"] for r in results['relu']]

x = np.arange(len(SEEDS))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, epochs_sig, width, label='Sigmoid', color='tab:blue', hatch='//')
plt.bar(x + width / 2, epochs_relu, width, label='ReLU', color='tab:orange')

plt.xticks(x, SEEDS)
plt.xlabel('Seed')
plt.ylabel('Epochs')
plt.title('Stability across 5 random initializations')
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------
# График 3: Разделяющие поверхности
# ------------------------------
def plot_decision_boundary(ax, model, title):
    xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    grid_raw = np.c_[xx.ravel(), yy.ravel()]
    grid_norm = normalize(grid_raw)

    preds = model.forward(grid_norm).reshape(xx.shape)

    # Используем P(c1) от 0 до 1
    contour = ax.contourf(xx, yy, preds, levels=25, cmap='viridis', vmin=0, vmax=1)

    # Обучающие точки
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=y_norm.ravel(), edgecolors='w',
               s=80, cmap='coolwarm', vmin=0, vmax=1, zorder=10)

    ax.set_title(title)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    return contour


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
model_sig = results['sigmoid'][0]["model"]
model_relu = results['relu'][0]["model"]

c1 = plot_decision_boundary(ax1, model_sig, "Hidden sigmoid")
c2 = plot_decision_boundary(ax2, model_relu, "Hidden ReLU")

fig.colorbar(c2, ax=ax2, label='P(c1)')
plt.tight_layout()
plt.show()

# ------------------------------
# Режим функционирования (8 примеров)
# ------------------------------
print("\n" + "=" * 65)
print("Работа в режиме функционирования (Модель ReLU, seed 6):")
print(f"{'A':>4} {'B':>4} | {'Выход 0..1':>10} | {'Исходная шкала':>14} | {'Класс':>6}")
print("-" * 65)

test_pairs = [
    (-9, -9), (-9, 6), (6, -9), (6, 6),
    (-6, 4), (0, 0), (6, 0), (-10, 10)
]

for a, b in test_pairs:
    x_test = normalize(np.array([[a, b]], dtype=float))
    pred_norm = float(model_relu.forward(x_test).ravel()[0])
    pred_out = float(denormalize(pred_norm))

    closest_class = C0 if abs(pred_out - C0) <= abs(pred_out - C1) else C1
    print(f"{a:>4} {b:>4} | {pred_norm:>10.4f} | {pred_out:>14.4f} | {closest_class:>6}")
print("=" * 65)