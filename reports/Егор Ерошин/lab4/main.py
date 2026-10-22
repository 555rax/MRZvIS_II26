import numpy as np
import matplotlib.pyplot as plt


# --- 1. Генерация ряда ---
def generate_series(a, b, c, d, length=200):
    k = np.arange(length)
    return a * np.cos(b * k) + c * np.sin(d * k)


# Параметры варианта 3
A, B, C, D = 1.5, 0.40, 0.8, 0.90
P_VAR, N_VAR = 6, 5

# Исходный ряд в реальном масштабе
series_orig = generate_series(A, B, C, D)
train_limit = 140
train_series_orig = series_orig[:train_limit]
test_series_orig = series_orig[train_limit:]

# --- 2. Нормализация ---
# Берем min/max ТОЛЬКО из обучающей выборки!
x_min = np.min(train_series_orig)
x_max = np.max(train_series_orig)


def normalize(x):
    return (x - x_min) / (x_max - x_min)


def denormalize(x_norm):
    return x_norm * (x_max - x_min) + x_min


series_norm = normalize(series_orig)


# --- 3. Скользящее окно ---
def create_dataset(data, p):
    X, y = [], []
    for i in range(len(data) - p):
        X.append(data[i:i + p])
        y.append(data[i + p])
    return np.array(X), np.array(y).reshape(-1, 1)


# --- 4. Реализация MLP p-N-1 ---
class MLP:
    def __init__(self, p, n, seed=42):
        np.random.seed(seed)
        # Инициализация весов малыми случайными значениями
        self.W1 = np.random.uniform(-0.5, 0.5, (p, n))
        self.b1 = np.random.uniform(-0.5, 0.5, (1, n))
        self.W2 = np.random.uniform(-0.5, 0.5, (n, 1))
        self.b2 = np.random.uniform(-0.5, 0.5, (1, 1))

    def sigmoid(self, s):
        return 1 / (1 + np.exp(-s))

    def sigmoid_deriv(self, s):
        return s * (1 - s)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2  # Линейный выход f(s) = s

    def backward(self, X, y, output, lr):
        # Ошибка выходного слоя (производная MSE: output - y)
        delta_out = output - y

        # Градиенты выходного слоя
        dW2 = np.dot(self.a1.T, delta_out)
        db2 = np.sum(delta_out, axis=0, keepdims=True)

        # Ошибка скрытого слоя
        delta_h = np.dot(delta_out, self.W2.T) * self.sigmoid_deriv(self.a1)

        # Градиенты скрытого слоя
        dW1 = np.dot(X.T, delta_h)
        db1 = np.sum(delta_h, axis=0, keepdims=True)

        # Обновление весов (Online gradient descent)
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train(self, X, y, lr=0.1, epochs=20000, target_mse=1e-4):
        mse_history = []
        for epoch in range(epochs):
            # Перемешивание выборки в каждой эпохе
            indices = np.arange(len(X))
            np.random.shuffle(indices)

            epoch_errors = []
            for i in indices:
                xi = X[i:i + 1]
                yi = y[i:i + 1]

                output = self.forward(xi)
                self.backward(xi, yi, output, lr)

                error = (output - yi) ** 2
                epoch_errors.append(error[0][0])

            avg_mse = np.mean(epoch_errors)
            mse_history.append(avg_mse)

            if avg_mse <= target_mse:
                print(f"Остановка на эпохе {epoch}, MSE: {avg_mse:.6f}")
                break
        return mse_history


# --- 5. Метрики ---
def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    # sMAPE используется т.к. ряд проходит через 0
    smape = 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return mae, rmse, smape


# --- 6. Базовый эксперимент ---
X_all, y_all = create_dataset(series_norm, P_VAR)
X_train, y_train = X_all[:train_limit - P_VAR], y_all[:train_limit - P_VAR]
X_test, y_test = X_all[train_limit - P_VAR:], y_all[train_limit - P_VAR:]

model = MLP(P_VAR, N_VAR)
history = model.train(X_train, y_train, lr=0.1)

# Получение прогнозов
train_preds_norm = model.forward(X_train)
test_preds_norm = model.forward(X_test)

# Денормализация (пересчет в исходную шкалу)
y_train_orig = denormalize(y_train)
y_test_orig = denormalize(y_test)
train_preds_orig = denormalize(train_preds_norm)
test_preds_orig = denormalize(test_preds_norm)

mae, rmse, smape = calculate_metrics(y_test_orig, test_preds_orig)
print(f"Результаты: MAE={mae:.4f}, RMSE={rmse:.4f}, sMAPE={smape:.2f}%")


# --- 9. Многошаговый (рекурсивный) прогноз ---
def recursive_forecast(model, start_window, steps=30):
    forecast = []
    current_window = start_window.copy()
    for _ in range(steps):
        pred = model.forward(current_window.reshape(1, -1))
        forecast.append(pred[0, 0])
        current_window = np.append(current_window[1:], pred)
    return np.array(forecast)


start_win_idx = train_limit - P_VAR
start_window = series_norm[start_win_idx: train_limit]
rec_preds_norm = recursive_forecast(model, start_window, 30)
rec_preds_orig = denormalize(rec_preds_norm)
rec_actual_orig = series_orig[train_limit: train_limit + 30]

# --- 10. Построение графиков ---
plt.figure(figsize=(14, 10))

# 1. График ряда и одношагового прогноза
plt.subplot(3, 1, 1)
plt.plot(series_orig, label='Исходный ряд', color='lightgray', linewidth=2)
plt.axvline(x=train_limit, color='red', linestyle='--', label='Граница Train/Test (k=140)')
plt.plot(np.arange(P_VAR, train_limit), train_preds_orig, label='Прогноз (Train)', color='blue', alpha=0.7)
plt.plot(np.arange(train_limit, 200), test_preds_orig, label='Прогноз (Test)', color='green')
plt.title('Одношаговый прогноз на всем интервале')
plt.xlabel('k')
plt.ylabel('x(k)')
plt.legend()
plt.grid(True)

# 2. Сходимость обучения
plt.subplot(3, 1, 2)
plt.semilogy(history, color='purple')
plt.title('Сходимость ошибки обучения (MSE на логарифмической шкале Y)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.grid(True)

# 3. Рекурсивный прогноз
plt.subplot(3, 2, 5)
plt.plot(rec_actual_orig, 'o-', label='Эталон', color='black')
plt.plot(rec_preds_orig, 'x--', label='Рекурсивный прогноз', color='orange')
plt.title('Многошаговый прогноз (30 шагов)')
plt.xlabel('Шаг прогноза')
plt.ylabel('x(k)')
plt.legend()
plt.grid(True)

# 4. Диаграмма рассеяния (Scatter plot)
plt.subplot(3, 2, 6)
plt.scatter(y_test_orig, test_preds_orig, color='green', alpha=0.8)
min_val = min(y_test_orig.min(), test_preds_orig.min())
max_val = max(y_test_orig.max(), test_preds_orig.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Идеальное совпадение y=x')
plt.xlabel('Эталон')
plt.ylabel('Прогноз')
plt.title('Диаграмма рассеяния (Test)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# --- 11. Режим функционирования ---
def operational_mode(model, k_index):
    window = series_orig[k_index - P_VAR: k_index]
    win_norm = normalize(window)
    pred_norm = model.forward(win_norm.reshape(1, -1))[0, 0]
    pred_orig = denormalize(pred_norm)
    actual = series_orig[k_index]
    print(f"--- Окно для k={k_index} ---")
    print(f"Исходное окно: {np.round(window, 4)}")
    print(f"Нормализованные входы: {np.round(win_norm, 4)}")
    print(f"Прогноз (норм. шкала): {pred_norm:.4f}")
    print(f"Прогноз (исх. шкала):  {pred_orig:.4f}")
    print(f"Истинное значение x({k_index}): {actual:.4f}")
    print(f"Абсолютная ошибка:    {abs(actual - pred_orig):.4f}\n")


print("\n=== ДЕМОНСТРАЦИЯ РЕЖИМА ФУНКЦИОНИРОВАНИЯ ===")
# 3 окна из обучающей выборки
operational_mode(model, 50)
operational_mode(model, 80)
operational_mode(model, 120)

# 3 окна из тестовой выборки
operational_mode(model, 145)
operational_mode(model, 170)
operational_mode(model, 190)