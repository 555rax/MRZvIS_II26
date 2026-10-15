import numpy as np
import matplotlib.pyplot as plt

# ====================== ВАРИАНТ 3 ======================
c0, c1 = -9.0, -8.0

LEARNING_RATE = 0.5
MAX_EPOCHS = 15000
TARGET_ERROR = 0.05          # Ee для BCE (суммарная ошибка по 4 примерам)

X_raw = np.array([
    [-9., -9.],
    [-9., -8.],
    [-8., -9.],
    [-8., -8.]
], dtype=float)

y_raw = np.array([
    [-9.],
    [-8.],
    [-8.],
    [-9.]
], dtype=float)

# -------------------- нормализация --------------------
def normalize_x(X):
    return (X - c0) / (c1 - c0)

def normalize_y(y):
    return (y - c0) / (c1 - c0)          # → {0, 1}

def denormalize_y(y_norm):
    return c0 + y_norm * (c1 - c0)

X = normalize_x(X_raw)
y = normalize_y(y_raw)

# -------------------- активации --------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))

def relu(z):
    return np.maximum(0.0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

# ====================== MLP 2-2-1 ======================
class MLP:
    def __init__(self, use_relu_hidden=False, seed=42):
        np.random.seed(seed)
        # небольшие случайные веса
        self.W1 = np.random.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, 2))
        self.W2 = np.random.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))
        self.use_relu = use_relu_hidden

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        if self.use_relu:
            self.a1 = relu(self.z1)
        else:
            self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train(self, X, y, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, target=TARGET_ERROR):
        history = []
        for epoch in range(1, max_epochs + 1):
            total_err = 0.0
            # online (по одному примеру)
            for i in range(len(X)):
                xi = X[i:i+1]
                yi = y[i:i+1]

                a2 = self.forward(xi)

                # BCE
                eps = 1e-12
                a2c = np.clip(a2, eps, 1 - eps)
                err = -(yi * np.log(a2c) + (1 - yi) * np.log(1 - a2c))
                total_err += err[0, 0]

                # δ² (sigmoid + BCE)
                delta2 = a2 - yi

                # δ¹
                if self.use_relu:
                    delta1 = (delta2 @ self.W2.T) * relu_deriv(self.z1)
                else:
                    delta1 = (delta2 @ self.W2.T) * self.a1 * (1 - self.a1)

                # обновление весов
                self.W2 -= lr * (self.a1.T @ delta2)
                self.b2 -= lr * delta2
                self.W1 -= lr * (xi.T @ delta1)
                self.b1 -= lr * delta1

            history.append(total_err)
            if total_err <= target:
                return epoch, total_err, history
        return max_epochs, total_err, history

    def predict_raw(self, x_raw):
        """x_raw shape (1,2) или (2,) в исходной шкале → (norm_out, raw_out, class)"""
        x_norm = normalize_x(np.atleast_2d(x_raw))
        out_norm = self.forward(x_norm)[0, 0]
        out_raw = denormalize_y(out_norm)
        cls = c0 if abs(out_raw - c0) <= abs(out_raw - c1) else c1
        return out_norm, out_raw, cls


# ====================== 5 запусков ======================
seeds = [42, 123, 456, 789, 2024]

def run_series(use_relu, name):
    results = []
    histories = []
    models = []
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    for s in seeds:
        net = MLP(use_relu_hidden=use_relu, seed=s)
        ep, err, hist = net.train(X, y)
        # оценка
        correct = 0
        mae = 0.0
        for i in range(4):
            _, pred_raw, pred_cls = net.predict_raw(X_raw[i])
            mae += abs(pred_raw - y_raw[i, 0])
            if pred_cls == y_raw[i, 0]:
                correct += 1
        acc = correct / 4 * 100
        mae /= 4
        results.append({'epochs': ep, 'Es': err, 'acc': acc, 'mae': mae, 'seed': s})
        histories.append(hist)
        models.append(net)
        print(f"seed={s:4d} | epochs={ep:5d} | Es={err:.6f} | Acc={acc:5.1f}% | MAE={mae:.4f}")
    return results, histories, models

res_A, hist_A, models_A = run_series(False, "Конфигурация A (Sigmoid на всех слоях)")
res_B, hist_B, models_B = run_series(True,  "Конфигурация B (ReLU на скрытом слое)")

# репрезентативные модели (seed=42)
mlp_A = models_A[0]
mlp_B = models_B[0]
ep_A, Es_A = res_A[0]['epochs'], res_A[0]['Es']
ep_B, Es_B = res_B[0]['epochs'], res_B[0]['Es']
mae_A = res_A[0]['mae']
mae_B = res_B[0]['mae']
acc_A = res_A[0]['acc']
acc_B = res_B[0]['acc']

# ====================== таблицы ======================
print("\n" + "="*70)
print("СВОДНАЯ ТАБЛИЦА (репрезентативный запуск seed=42)")
print("="*70)
print(f"{'Метрика':<35} | {'Config A (Sigmoid)':<20} | {'Config B (ReLU)':<20}")
print("-"*70)
print(f"{'Эпох до останова':<35} | {ep_A:<20} | {ep_B:<20}")
print(f"{'Итоговая Es':<35} | {Es_A:<20.6f} | {Es_B:<20.6f}")
print(f"{'Accuracy, %':<35} | {acc_A:<20.1f} | {acc_B:<20.1f}")
print(f"{'MAE (исходная шкала)':<35} | {mae_A:<20.4f} | {mae_B:<20.4f}")

print("\n" + "="*70)
print("СТАТИСТИКА ПО 5 ЗАПУСКАМ")
print("="*70)
def stats(res, key):
    vals = [r[key] for r in res]
    return np.mean(vals), np.std(vals), np.min(vals), np.max(vals)

print(f"{'':<20} | {'Config A':^30} | {'Config B':^30}")
print(f"{'Метрика':<20} | {'mean ± std (min–max)':^30} | {'mean ± std (min–max)':^30}")
print("-"*85)
for key, name in [('epochs','Эпохи'), ('mae','MAE'), ('Es','Es')]:
    mA, sA, mnA, mxA = stats(res_A, key)
    mB, sB, mnB, mxB = stats(res_B, key)
    print(f"{name:<20} | {mA:7.2f}±{sA:5.2f} ({mnA:.1f}–{mxA:.1f}) | {mB:7.2f}±{sB:5.2f} ({mnB:.1f}–{mxB:.1f})")

# ====================== графики ======================
fig = plt.figure(figsize=(16, 12))

# 1. Сходимость
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(hist_A[0], label='A: Sigmoid', color='C0', lw=2)
ax1.plot(hist_B[0], label='B: ReLU', color='C1', lw=2)
ax1.axhline(TARGET_ERROR, color='gray', ls='--', label=f'Ee = {TARGET_ERROR}')
ax1.set_title('1. График сходимости (Es от эпохи)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Суммарная ошибка Es (BCE)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 2. Диаграмма эпох
ax2 = fig.add_subplot(2, 2, 2)
x = np.arange(len(seeds))
w = 0.35
ax2.bar(x - w/2, [r['epochs'] for r in res_A], w, label='A: Sigmoid', color='skyblue')
ax2.bar(x + w/2, [r['epochs'] for r in res_B], w, label='B: ReLU', color='salmon')
ax2.set_xticks(x)
ax2.set_xticklabels(seeds)
ax2.set_title('2. Число эпох по 5 запускам')
ax2.set_xlabel('seed')
ax2.set_ylabel('Эпохи')
ax2.legend()
ax2.grid(True, axis='y', alpha=0.3)

# 3–4. Разделяющие поверхности
def plot_surface(ax, model, title):
    xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([model.forward(normalize_x(p.reshape(1, 2)))[0, 0] for p in grid])
    zz = zz.reshape(xx.shape)
    cf = ax.contourf(xx, yy, zz, levels=30, cmap='RdYlBu_r', alpha=0.85)
    ax.contour(xx, yy, zz, levels=[0.5], colors='k', linewidths=2)
    # точки обучающей выборки
    for i in range(4):
        col = 'red' if y_raw[i, 0] == c0 else 'blue'
        mkr = 'o' if y_raw[i, 0] == c0 else '^'
        ax.scatter(X_raw[i, 0], X_raw[i, 1], c=col, s=180, edgecolors='k',
                   marker=mkr, zorder=5, label=f'класс {y_raw[i,0]:.0f}')
    ax.set_title(title)
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    return cf

ax3 = fig.add_subplot(2, 2, 3)
cf3 = plot_surface(ax3, mlp_A, '3. Разделяющая поверхность — Config A (Sigmoid)')
plt.colorbar(cf3, ax=ax3, label='выход (норм.)')

ax4 = fig.add_subplot(2, 2, 4)
cf4 = plot_surface(ax4, mlp_B, '4. Разделяющая поверхность — Config B (ReLU)')
plt.colorbar(cf4, ax=ax4, label='выход (норм.)')

plt.tight_layout()
plt.savefig('lab3_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ====================== интерактивный режим ======================
print("\n" + "="*60)
print("ИНТЕРАКТИВНЫЙ РЕЖИМ (введите A и B из [-10, 10])")
print("Пустая строка — выход")
print("="*60)

while True:
    try:
        s = input("A B > ").strip()
        if not s:
            break
        a, b = map(float, s.split())
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("  значения должны быть в [-10, 10]")
            continue
        nA, rA, cA = mlp_A.predict_raw([a, b])
        nB, rB, cB = mlp_B.predict_raw([a, b])
        print(f"  Config A: норм={nA:.4f}  исх={rA:.4f}  класс={cA:.0f}")
        print(f"  Config B: норм={nB:.4f}  исх={rB:.4f}  класс={cB:.0f}")
    except Exception as e:
        print("  ошибка ввода:", e)