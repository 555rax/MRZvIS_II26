import numpy as np
import matplotlib.pyplot as plt

a, b_param, c, d = 0.9, 0.33, 0.45, 0.75
p_var, N_var = 7, 4
total_len = 200
train_len = 140
Ee = 1e-4


def generate_series():
    k = np.arange(total_len)
    return a * np.cos(b_param * k) + c * np.sin(d * k)


def create_windows(series, p):
    X, Y = [], []
    for i in range(len(series) - p):
        X.append(series[i: i + p])
        Y.append(series[i + p])
    return np.array(X), np.array(Y)


def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    return mae, rmse, smape


def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_deriv(x): s = sigmoid(x); return s * (1 - s)


class MLP_Regression:
    def __init__(self, p, N, seed=42):
        np.random.seed(seed)
        self.W1 = np.random.randn(p, N) * 0.1
        self.b1 = np.random.randn(N) * 0.1
        self.W2 = np.random.randn(N, 1) * 0.1
        self.b2 = np.random.randn(1) * 0.1

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2
        return self.a2

    def backward(self, x, y_true, lr):
        dZ2 = self.a2 - y_true
        dW2 = np.outer(self.a1, dZ2)
        db2 = dZ2

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_deriv(self.z1)
        dW1 = np.outer(x, dZ1)
        db1 = dZ1

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1


def train_mlp(model, X, Y, lr=0.05, max_epochs=20000):
    history = []
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        idx = np.random.permutation(len(X))
        total_loss = 0

        for i in idx:
            x, y = X[i], Y[i:i + 1]
            pred = model.forward(x)
            loss = 0.5 * (pred - y) ** 2
            total_loss += loss[0]
            model.backward(x, y, lr)

        Es = total_loss / len(X)
        history.append(Es)
        if Es <= Ee:
            break
    return epoch, Es, history



series = generate_series()
train_series = series[:train_len]
test_series = series[train_len:]

s_min, s_max = np.min(train_series), np.max(train_series)


def norm(v): return (v - s_min) / (s_max - s_min)


def denorm(v): return v * (s_max - s_min) + s_min


series_norm = norm(series)
train_norm = series_norm[:train_len]
test_norm = series_norm[train_len:]

X_train, Y_train = create_windows(train_norm, p_var)
X_test, Y_test = create_windows(test_norm, p_var)

print(f"Примеров Train: {len(X_train)}, Test: {len(X_test)}")

print("\n--- Базовый эксперимент (p=7, N=4) ---")
model_base = MLP_Regression(p=p_var, N=N_var, seed=42)
ep, es, hist_base = train_mlp(model_base, X_train, Y_train, lr=0.1)

train_preds = np.array([model_base.forward(x)[0] for x in X_train])
test_preds = np.array([model_base.forward(x)[0] for x in X_test])

mae_tr, rmse_tr, smape_tr = calc_metrics(denorm(Y_train), denorm(train_preds))
mae_te, rmse_te, smape_te = calc_metrics(denorm(Y_test), denorm(test_preds))

print(f"Эпох: {ep}, Es: {es:.6f}")
print(f"Train: MAE={mae_tr:.4f}, RMSE={rmse_tr:.4f}, sMAPE={smape_tr:.2f}%")
print(f"Test:  MAE={mae_te:.4f}, RMSE={rmse_te:.4f}, sMAPE={smape_te:.2f}%")

steps = 30
recursive_preds = []
current_window = list(series_norm[train_len - p_var: train_len])

for _ in range(steps):
    pred = model_base.forward(np.array(current_window))[0]
    recursive_preds.append(pred)
    current_window.append(pred)
    current_window.pop(0)

recursive_preds_raw = denorm(np.array(recursive_preds))
actual_steps = series[train_len: train_len + steps]
rec_errors = np.abs(actual_steps - recursive_preds_raw)

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(series, label='Исходный ряд', color='black')
plt.axvline(x=train_len, color='r', linestyle='--', label='Граница Train/Test')
plt.plot(np.arange(p_var, train_len), denorm(train_preds), label='Train Pred', color='green')
plt.plot(np.arange(train_len + p_var, total_len), denorm(test_preds), label='Test Pred', color='orange')
plt.title("Одношаговый прогноз на всем ряду")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(hist_base, color='blue')
plt.axhline(y=Ee, color='r', linestyle='--')
plt.yscale('log')
plt.title("График сходимости (Log-scale)")
plt.xlabel("Эпоха")
plt.ylabel("MSE")

plt.subplot(2, 2, 3)
plt.scatter(denorm(Y_test), denorm(test_preds), color='purple', alpha=0.6)
plt.plot([min(denorm(Y_test)), max(denorm(Y_test))],
         [min(denorm(Y_test)), max(denorm(Y_test))], color='red', linestyle='--')
plt.title("Диаграмма рассеяния (Эталон - Прогноз, Test)")
plt.xlabel("Эталон")
plt.ylabel("Прогноз")

plt.subplot(2, 2, 4)
plt.plot(actual_steps, label='Эталон', marker='o')
plt.plot(recursive_preds_raw, label='Рекурсивный прогноз 30 шагов', marker='x')
plt.plot(rec_errors, label='Абсолютная ошибка', linestyle='--')
plt.axhline(y=0.05 * (s_max - s_min), color='r', linestyle=':', label='5% Размаха')
plt.title("Рекурсивный прогноз на 30 шагов")
plt.legend()

plt.tight_layout()
plt.show()

print("\n--- Интерактивный режим ---")
while True:
    try:
        user_input = input(f"\nВведите номер шага k (от {p_var} до 199) для взятия окна или 'q' для выхода: ")
        if user_input.lower() == 'q': break

        k = int(user_input)
        if k < p_var or k >= total_len:
            print(f"Ошибка. Введите число от {p_var} до 199.")
            continue

        win_raw = series[k - p_var: k]
        win_norm = norm(win_raw)
        actual_val = series[k]

        pred_n = model_base.forward(win_norm)[0]
        pred_raw = denorm(pred_n)

        print(f"Окно (исходное): {np.round(win_raw, 4)}")
        print(f"Нормализованные входы: {np.round(win_norm, 4)}")
        print(f"Прогноз (нормализованный): {pred_n:.4f}")
        print(f"Прогноз (исходный): {pred_raw:.4f}")
        print(f"Истинное значение x({k}): {actual_val:.4f} | Ошибка: {abs(pred_raw - actual_val):.4f}")

    except ValueError:
        print("Неверный ввод.")