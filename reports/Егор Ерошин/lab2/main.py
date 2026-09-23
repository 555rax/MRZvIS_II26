import numpy as np
import matplotlib.pyplot as plt

c0 = -9.0
c1 = -8.0

LEARNING_RATE = 1.0
MAX_EPOCHS = 10000
TARGET_ERROR_MSE = 0.005
TARGET_ERROR_BCE = 0.4

X_raw = np.array([
    [-9.0, -9.0],
    [-9.0, -8.0],
    [-8.0, -9.0],
    [-8.0, -8.0]
], dtype=float)

y_raw = np.array([
    [-9.0],
    [-8.0],
    [-8.0],
    [-9.0]
], dtype=float)


def normalize_x(X):
    return (X - c0) / (c1 - c0)



def normalize_y_mse(y):
    return (y - c0) / (c1 - c0) * 0.8 + 0.1


def denormalize_y_mse(y_norm):
    return c0 + (y_norm - 0.1) / 0.8 * (c1 - c0)


def normalize_y_bce(y):
    return (y - c0) / (c1 - c0)


def denormalize_y_bce(y_norm):
    return c0 + y_norm * (c1 - c0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))


class MultilayerPerceptron:
    def __init__(self, input_dim=2, hidden_dim=2, output_dim=1, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.uniform(-1.0, 1.0, (input_dim, hidden_dim))
        self.b1 = np.random.uniform(-1.0, 1.0, (1, hidden_dim))
        self.W2 = np.random.uniform(-1.0, 1.0, (hidden_dim, output_dim))
        self.b2 = np.random.uniform(-1.0, 1.0, (1, output_dim))

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = sigmoid(z2)
        return a2, a1

    def train(self, X, y, loss_type='mse', lr=LEARNING_RATE, max_epochs=MAX_EPOCHS, target_error=TARGET_ERROR_MSE):
        epoch_errors = []

        for epoch in range(1, max_epochs + 1):
            total_error = 0.0

            # Градиенты для пакетного или онлайн обучения (здесь онлайн - обновление после каждого примера)
            for i in range(len(X)):
                x_i = X[i:i + 1]
                y_i = y[i:i + 1]

                a2, a1 = self.forward(x_i)

                if loss_type == 'mse':
                    err = 0.5 * (y_i[0, 0] - a2[0, 0]) ** 2
                    total_error += err
                    # Производная MSE + сигмоида
                    delta2 = (a2 - y_i) * a2 * (1.0 - a2)
                elif loss_type == 'bce':
                    # Защита от деления на ноль при логарифме
                    eps = 1e-15
                    a2_clipped = np.clip(a2, eps, 1.0 - eps)
                    err = - (y_i[0, 0] * np.log(a2_clipped[0, 0]) + (1.0 - y_i[0, 0]) * np.log(1.0 - a2_clipped[0, 0]))
                    total_error += err
                    # Красивое сокращение для градиента BCE + Сигмоида: delta2 = (a2 - y_i)
                    delta2 = (a2 - y_i)

                delta1 = np.dot(delta2, self.W2.T) * a1 * (1.0 - a1)

                self.W2 -= lr * np.dot(a1.T, delta2)
                self.b2 -= lr * delta2
                self.W1 -= lr * np.dot(x_i.T, delta1)
                self.b1 -= lr * delta1

            epoch_errors.append(total_error)

            if total_error <= target_error:
                return epoch, total_error, epoch_errors

        return max_epochs, total_error, epoch_errors



X_norm = normalize_x(X_raw)
y_norm_mse = normalize_y_mse(y_raw)
y_norm_bce = normalize_y_bce(y_raw)


n_runs = 5
seeds = [42, 123, 456, 789, 2026]

print("=" * 70)
print("СЕРИЯ ИЗ 5 ЗАПУСКОВ ДЛЯ КОНФИГУРАЦИЙ A И Б")
print("=" * 70)

results_A = []
history_A_runs = []
for seed in seeds:
    mlp_a = MultilayerPerceptron(seed=seed)
    ep, err, hist = mlp_a.train(X_norm, y_norm_mse, loss_type='mse', target_error=TARGET_ERROR_MSE)
    results_A.append((ep, err))
    history_A_runs.append(hist)

results_B = []
history_B_runs = []
for seed in seeds:
    mlp_b = MultilayerPerceptron(seed=seed)
    ep, err, hist = mlp_b.train(X_norm, y_norm_bce, loss_type='bce', target_error=TARGET_ERROR_BCE)
    results_B.append((ep, err))
    history_B_runs.append(hist)

# Обучим по одному репрезентативному представителю для детальной оценки и графиков
mlp_rep_a = MultilayerPerceptron(seed=42)
ep_a, err_a, hist_a = mlp_rep_a.train(X_norm, y_norm_mse, loss_type='mse', target_error=TARGET_ERROR_MSE)

mlp_rep_b = MultilayerPerceptron(seed=42)
ep_b, err_b, hist_b = mlp_rep_b.train(X_norm, y_norm_bce, loss_type='bce', target_error=TARGET_ERROR_BCE)



def evaluate_config(model, loss_type, name):
    correct = 0
    total_abs_error = 0.0
    print(f"\n--- Оценка модели: {name} ---")
    print("A\tB\tИстинный\tŷ (норм)\t\tŷ (исх шкала)\tКласс\t|Ошибка|")
    print("-" * 75)

    for i in range(len(X_raw)):
        x_norm_i = normalize_x(X_raw[i:i + 1])
        out_norm, _ = model.forward(x_norm_i)

        if loss_type == 'mse':
            y_pred = denormalize_y_mse(out_norm[0, 0])
        else:
            y_pred = denormalize_y_bce(out_norm[0, 0])

        pred_class = c0 if abs(y_pred - c0) < abs(y_pred - c1) else c1
        target_class = y_raw[i, 0]

        abs_err = abs(y_pred - target_class)
        total_abs_error += abs_err

        if pred_class == target_class:
            correct += 1

        print(
            f"{X_raw[i, 0]:.1f}\t{X_raw[i, 1]:.1f}\t{target_class:.1f}\t\t{out_norm[0, 0]:.4f}\t\t{y_pred:.4f}\t\t{pred_class:.1f}\t{abs_err:.4f}")

    accuracy = (correct / len(X_raw)) * 100.0
    mean_abs_error = total_abs_error / len(X_raw)
    return accuracy, mean_abs_error


acc_a, mae_a = evaluate_config(mlp_rep_a, 'mse', "Конфигурация А (MSE)")
acc_b, mae_b = evaluate_config(mlp_rep_b, 'bce', "Конфигурация Б (BCE)")

print("\n" + "=" * 65)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (РЕПРЕЗЕНТАТИВНЫЙ ЗАПУСК)")
print("=" * 65)
print(f"Метрика / Параметр             | Конфигурация А (MSE) | Конфигурация Б (BCE)")
print(f"-------------------------------------------------------------------")
print(f"Количество эпох                | {ep_a:<20} | {ep_b:<20}")
print(f"Итоговая суммарная ошибка (Es) | {err_a:<20.6f} | {err_b:<20.6f}")
print(f"Точность (Accuracy)            | {acc_a:<19.1f}%| {acc_b:<19.1f}%")
print(f"Средняя абс. ошибка (MAE)      | {mae_a:<20.4f} | {mae_b:<20.4f}")


def inference_demo(model, loss_type, name):
    print(f"\n--- Режим функционирования: {name} ---")
    test_inputs = [(-9.0, -9.0), (-9.0, -8.0), (-8.5, -8.2), (0.0, 5.0), (-8.2, -8.8)]
    for A, B in test_inputs:
        x_user = np.array([[A, B]])
        x_norm_user = normalize_x(x_user)
        out_norm, _ = model.forward(x_norm_user)

        if loss_type == 'mse':
            y_hat = denormalize_y_mse(out_norm[0, 0])
        else:
            y_hat = denormalize_y_bce(out_norm[0, 0])

        predicted_class = c0 if abs(y_hat - c0) < abs(y_hat - c1) else c1
        print(
            f"Вход: ({A:.1f}, {B:.1f}) -> Выход (норм): {out_norm[0, 0]:.4f} | Выход (исх): {y_hat:.4f} -> Класс: {predicted_class:.1f}")


inference_demo(mlp_rep_a, 'mse', "Конфигурация А")
inference_demo(mlp_rep_b, 'bce', "Конфигурация Б")


fig = plt.figure(figsize=(16, 12))

# 1. График сходимости (суммарная ошибка от эпохи)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(hist_a, label='Конфигурация А (MSE)', color='blue', alpha=0.8)
ax1.plot(hist_b, label='Конфигурация Б (BCE)', color='orange', alpha=0.8)
ax1.axhline(y=TARGET_ERROR_MSE, color='blue', linestyle='--', alpha=0.5, label='Порог Ее (MSE)')
ax1.axhline(y=TARGET_ERROR_BCE, color='orange', linestyle='--', alpha=0.5, label='Порог Ее (BCE)')
ax1.set_title('1. График сходимости (ошибка от эпохи)')
ax1.set_xlabel('Эпоха')
ax1.set_ylabel('Суммарная ошибка Es')
ax1.legend()
ax1.grid(True)

# 2. Диаграмма разброса числа эпох по 5 запускам
ax2 = fig.add_subplot(2, 2, 2)
epochs_A_vals = [res[0] for res in results_A]
epochs_B_vals = [res[1] for res in results_B]
x_indexes = np.arange(len(seeds))
width = 0.35

ax2.bar(x_indexes - width / 2, epochs_A_vals, width, label='Конф. А (MSE)', color='skyblue')
ax2.bar(x_indexes + width / 2, epochs_B_vals, width, label='Конф. Б (BCE)', color='sandybrown')
ax2.set_title('2. Число эпох за 5 независимых запусков (seed)')
ax2.set_xlabel('Номер запуска (seed index)')
ax2.set_ylabel('Количество эпох')
ax2.set_xticks(x_indexes)
ax2.set_xticklabels([str(s) for s in seeds])
ax2.legend()
ax2.grid(True, axis='y')

# 3. Визуализация разделяющей поверхности (Heatmap) для Конфигурации Б
ax3 = fig.add_subplot(2, 2, 3)
xx, yy = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_norm = normalize_x(grid_points)


zz_b = np.array([mlp_rep_b.forward(np.array([pt]))[0][0, 0] for pt in grid_norm])
zz_b = zz_b.reshape(xx.shape)

contour = ax3.contourf(xx, yy, zz_b, levels=25, cmap='coolwarm', alpha=0.7)
fig.colorbar(contour, ax=ax3, label='Выход сети (норм. [0, 1])')

for i in range(len(X_raw)):
    color = 'red' if y_raw[i, 0] == c0 else 'blue'
    marker = 'o' if y_raw[i, 0] == c0 else '^'
    ax3.scatter(X_raw[i, 0], X_raw[i, 1], color=color, s=120, edgecolors='black', marker=marker,
                label=f'Класс {y_raw[i, 0]}')

ax3.set_title('3. Разделяющая поверхность и выборка (Конф. Б - BCE)')
ax3.set_xlabel('Ось A')
ax3.set_ylabel('Ось B')

# 4. Диаграмма сравнения точности восстановления шкалы (MAE)
ax4 = fig.add_subplot(2, 2, 4)
configs = ['Конфигурация А (MSE)', 'Конфигурация Б (BCE)']
mae_values = [mae_a, mae_b]
bars = ax4.bar(configs, mae_values, color=['cornflowerblue', 'coral'])
ax4.set_title('4. Сравнение средней абсолютной ошибки (MAE) в шкале [c0, c1]')
ax4.set_ylabel('MAE')
ax4.grid(True, axis='y')

for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()