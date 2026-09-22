import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

C0 = -9.0
C1 = 6.0
# Отличие от лр №1: введены раздельные пороги ошибки. Для логарифмической функции потерь (BCE)
# требуется менее строгий порог, чем для квадратичной (MSE).
TARGET_ERROR_MSE = 0.01
TARGET_ERROR_BCE = 0.05
MAX_EPOCHS = 10000
LEARNING_RATE = 0.5

X_raw = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]])
y_raw = np.array([C0, C1, C1, C0]).reshape(-1, 1)


def normalize(val, c_min=C0, c_max=C1):
    return (val - c_min) / (c_max - c_min)


def denormalize(val, c_min=C0, c_max=C1):
    return val * (c_max - c_min) + c_min


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))


def sigmoid_derivative(s):
    return s * (1.0 - s)


X_norm = normalize(X_raw)
y_norm = normalize(y_raw)


class MLP_MSE:
    def __init__(self, lr, seed=None):
        if seed is not None: np.random.seed(seed)
        self.lr = lr
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(1, 2) * 0.5
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.5

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step(self, x, y_true):
        pred = self.forward(x)
        error = y_true - pred

        delta2 = error * sigmoid_derivative(pred)
        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)

        self.W2 += self.lr * np.dot(self.a1.T, delta2)
        self.b2 += self.lr * delta2
        self.W1 += self.lr * np.dot(x.T, delta1)
        self.b1 += self.lr * delta1
        return 0.5 * np.sum(error ** 2)


#Отличие от лр №1: добавлен отдельный класс сети для работы с бинарной кросс-энтропией (BCE)
class MLP_BCE:
    def __init__(self, lr, seed=None):
        if seed is not None: np.random.seed(seed)
        self.lr = lr
        self.W1 = np.random.randn(2, 2) * 0.5
        self.b1 = np.random.randn(1, 2) * 0.5
        self.W2 = np.random.randn(2, 1) * 0.5
        self.b2 = np.random.randn(1, 1) * 0.5

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def train_step(self, x, y_true):
        pred = self.forward(x)

        # Отличие от лр №1: искусственное ограничение значений предсказания (клиппинг)
        # для предотвращения математической ошибки log(0) при расчете функции потерь
        pred_clipped = np.clip(pred, 1e-15, 1 - 1e-15)

        error = y_true - pred

        # Отличие от лр №1: сокращение производной. Поскольку производная логарифмической функции
        # потерь математически сокращается с производной сигмоиды выходного слоя,
        # локальный градиент (delta2) равен просто ошибке (y_true - pred)
        delta2 = error

        delta1 = np.dot(delta2, self.W2.T) * sigmoid_derivative(self.a1)

        self.W2 += self.lr * np.dot(self.a1.T, delta2)
        self.b2 += self.lr * delta2
        self.W1 += self.lr * np.dot(x.T, delta1)
        self.b1 += self.lr * delta1

        # Отличие от лр №1: возврат значения ошибки рассчитывается по формуле кросс-энтропии, а не суммы квадратов
        bce_loss = - (y_true * np.log(pred_clipped) + (1 - y_true) * np.log(1 - pred_clipped))
        return np.sum(bce_loss)


def train_model(model_class, target_error, title, seed=None):
    model = model_class(lr=LEARNING_RATE, seed=seed)
    history = []

    for epoch in range(MAX_EPOCHS):
        indices = np.arange(len(X_norm))
        np.random.shuffle(indices)

        epoch_error = 0.0
        for i in indices:
            epoch_error += model.train_step(X_norm[i:i + 1], y_norm[i:i + 1])

        history.append(epoch_error)
        if epoch_error <= target_error:
            return model, history, epoch + 1, True

    return model, history, MAX_EPOCHS, False


def calc_mae(model):
    mae = 0.0
    for i in range(len(X_norm)):
        pred_norm = model.forward(X_norm[i:i + 1])
        pred_real = denormalize(pred_norm)[0][0]
        mae += abs(pred_real - y_raw[i][0])
    return mae / len(X_norm)


def calc_accuracy(model):
    correct = 0
    for i in range(len(X_norm)):
        pred_norm = model.forward(X_norm[i:i + 1])[0][0]
        pred_real = denormalize(pred_norm)
        true_val = y_raw[i][0]

        pred_class = C0 if abs(pred_real - C0) < abs(pred_real - C1) else C1
        true_class = C0 if abs(true_val - C0) < abs(true_val - C1) else C1

        if pred_class == true_class:
            correct += 1
    return correct / len(X_norm) * 100


print("=== Обучение эталонных моделей (Конфигурация А - MSE и Конфигурация Б - BCE) ===")
model_mse, hist_mse, ep_mse, conv_mse = train_model(MLP_MSE, TARGET_ERROR_MSE, "MSE", seed=42)
print(f"MSE: Эпох = {ep_mse}, Итоговая ошибка = {hist_mse[-1]:.5f}, Сходимость = {conv_mse}")

model_bce, hist_bce, ep_bce, conv_bce = train_model(MLP_BCE, TARGET_ERROR_BCE, "BCE", seed=42)
print(f"BCE: Эпох = {ep_bce}, Итоговая ошибка = {hist_bce[-1]:.5f}, Сходимость = {conv_bce}")

print("\n=== Запуск серии из 5 экспериментов с разными seed ===")
seeds = [10, 42, 123, 777, 2024]
mse_epochs, bce_epochs = [], []
mse_converged_list, bce_converged_list = [], []

for s in seeds:
    _, _, e_mse, c_mse = train_model(MLP_MSE, TARGET_ERROR_MSE, "MSE", seed=s)
    _, _, e_bce, c_bce = train_model(MLP_BCE, TARGET_ERROR_BCE, "BCE", seed=s)
    mse_epochs.append(e_mse)
    bce_epochs.append(e_bce)
    mse_converged_list.append(c_mse)
    bce_converged_list.append(c_bce)
    status_mse = "✓" if c_mse else "✗"
    status_bce = "✓" if c_bce else "✗"
    print(f"Seed {s:4d}: MSE {status_mse} {e_mse} эпох, BCE {status_bce} {e_bce} эпох.")

mae_mse = calc_mae(model_mse)
mae_bce = calc_mae(model_bce)
acc_mse = calc_accuracy(model_mse)
acc_bce = calc_accuracy(model_bce)

print(f"\n=== Итоговые метрики ===")
print(f"MSE: Accuracy = {acc_mse:.1f}%, MAE = {mae_mse:.4f}")
print(f"BCE: Accuracy = {acc_bce:.1f}%, MAE = {mae_bce:.4f}")

print("\nОткрывается первый график (закройте окно, чтобы появился следующий)...")

plt.figure(figsize=(10, 6))
plt.plot(hist_mse, label='Конфигурация А (MSE)', color='blue', linewidth=2)
plt.plot(hist_bce, label='Конфигурация Б (BCE)', color='red', linewidth=2)
plt.axhline(TARGET_ERROR_MSE, color='blue', linestyle=':', linewidth=1.5, label=f'Порог MSE ({TARGET_ERROR_MSE})')
plt.axhline(TARGET_ERROR_BCE, color='red', linestyle=':', linewidth=1.5, label=f'Порог BCE ({TARGET_ERROR_BCE})')
plt.title('Сходимость: Es от номера эпохи (представительный запуск)')
plt.xlabel('Эпоха')
plt.ylabel('Суммарная ошибка (Es)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, min(2000, MAX_EPOCHS))
plt.show()

print("Открывается второй график (закройте окно, чтобы появился следующий)...")
x = np.arange(len(seeds))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars_mse = ax.bar(x - width / 2, mse_epochs, width, label='MSE', color='skyblue', edgecolor='black')
bars_bce = ax.bar(x + width / 2, bce_epochs, width, label='BCE', color='lightcoral', edgecolor='black')

for bar, conv in zip(bars_mse, mse_converged_list):
    if not conv:
        bar.set_hatch('///')
        bar.set_label('MSE (не сошлась)')
for bar, conv in zip(bars_bce, bce_converged_list):
    if not conv:
        bar.set_hatch('///')
        bar.set_label('BCE (не сошлась)')

ax.set_title('Разброс числа эпох по 5 запускам\n(штриховка = не сошлась за MAX_EPOCHS)')
ax.set_xlabel('Номер запуска (seed)')
ax.set_ylabel('Эпохи')
ax.set_xticks(x)
ax.set_xticklabels([f'Run {i + 1}\n(seed={s})' for i, s in enumerate(seeds)])
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("Открывается третий график (закройте окно, чтобы появился следующий)...")
xx, yy = np.meshgrid(np.linspace(C0 - 2, C1 + 2, 100), np.linspace(C0 - 2, C1 + 2, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_norm = normalize(grid_points)

Z_mse = np.array([model_mse.forward(p.reshape(1, 2))[0][0] for p in grid_norm]).reshape(xx.shape)
Z_bce = np.array([model_bce.forward(p.reshape(1, 2))[0][0] for p in grid_norm]).reshape(xx.shape)

Z_mse_real = denormalize(Z_mse)
Z_bce_real = denormalize(Z_bce)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
cmap = plt.cm.RdYlBu

c1 = ax1.contourf(xx, yy, Z_mse_real, levels=50, cmap=cmap, alpha=0.8)
ax1.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw.flatten(), cmap=cmap, edgecolors='k', s=100, label='Точки XOR')
ax1.set_title('Разделяющая поверхность (MSE)')
ax1.set_xlabel('Вход A')
ax1.set_ylabel('Вход B')
ax1.legend()
fig.colorbar(c1, ax=ax1, label='Выход сети (шкала [c0; c1])')

c2 = ax2.contourf(xx, yy, Z_bce_real, levels=50, cmap=cmap, alpha=0.8)
ax2.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw.flatten(), cmap=cmap, edgecolors='k', s=100, label='Точки XOR')
ax2.set_title('Разделяющая поверхность (BCE)')
ax2.set_xlabel('Вход A')
ax2.set_ylabel('Вход B')
ax2.legend()
fig.colorbar(c2, ax=ax2, label='Выход сети (шкала [c0; c1])')

plt.suptitle('Визуализация разделяющей поверхности для обеих конфигураций', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("Открывается четвёртый график (закройте окно, чтобы появился следующий)...")
plt.figure(figsize=(8, 6))
bars = plt.bar(['Конфигурация А\n(MSE)', 'Конфигурация Б\n(BCE)'], [mae_mse, mae_bce],
               color=['skyblue', 'lightcoral'], width=0.6, edgecolor='black')
plt.title('Сравнение точности восстановления шкалы\nСредняя абсолютная ошибка (MAE) в шкале [c0; c1]',
          fontsize=12, fontweight='bold')
plt.ylabel('MAE', fontsize=11)
plt.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, [mae_mse, mae_bce]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("\n=== РЕЖИМ ФУНКЦИОНИРОВАНИЯ (на базе конфигурации Б - BCE) ===")

test_cases = [
    (-9, -9), (-9, 6), (6, -9), (6, 6),
    (0, 0), (-5, 3), (2, -7)
]

print(f"\n{'Вход (A, B)':<15} | {'ŷ (0;1)':<10} | {'ŷ исходная':<12} | {'Ближайший класс'}")
print("-" * 65)
for a, b in test_cases:
    x_in = normalize(np.array([[a, b]]))
    out_norm = model_bce.forward(x_in)[0][0]
    out_real = denormalize(out_norm)

    dist_c0 = abs(out_real - C0)
    dist_c1 = abs(out_real - C1)
    closest = C0 if dist_c0 < dist_c1 else C1

    print(f"({a:>3}, {b:>3})        | {out_norm:>8.4f}   | {out_real:>10.4f}   | {closest}")

print("\n--- Интерактивный режим (введите 'exit' для выхода) ---")
print("Введите координаты A и B через пробел (например: -9 6)")
while True:
    try:
        user_input = input("\nВходные значения A и B: ")
        if user_input.lower() == 'exit':
            print("Завершение работы.")
            break

        a, b = map(float, user_input.split())
        x_in = normalize(np.array([[a, b]]))
        out_norm = model_bce.forward(x_in)[0][0]
        out_real = denormalize(out_norm)

        dist_c0 = abs(out_real - C0)
        dist_c1 = abs(out_real - C1)
        closest = C0 if dist_c0 < dist_c1 else C1

        print(f"Нормализованный выход : {out_norm:.4f}")
        print(f"Выход в шкале [-9; 6]:   {out_real:.4f}")
        print(f"Ближайший класс:         {closest}")
    except ValueError:
        print("Ошибка ввода. Пожалуйста, введите два числа через пробел или 'exit' для выхода.")