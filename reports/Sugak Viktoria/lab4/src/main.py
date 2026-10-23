import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 0.8, 0.20, 0.4, 0.50
P_VAR, N_VAR = 8, 4
SEED_BASE = 42
ETA = 0.05
EE = 1e-4
MAX_EPOCHS = 20000
K_TOTAL = 200
TRAIN_END = 140

np.random.seed(SEED_BASE)

def generate_series(a=A, b=B, c=C, d=D, k_max=K_TOTAL):
    k = np.arange(k_max)
    return a * np.cos(b * k) + c * np.sin(d * k)

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def denormalize(x_norm, x_min, x_max):
    return x_norm * (x_max - x_min) + x_min

def make_windows(series, p):
    X, y = [], []
    for i in range(p, len(series)):
        X.append(series[i - p:i])
        y.append(series[i])
    return np.array(X), np.array(y)

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -500, 500)))

def sigmoid_deriv(out):
    return out * (1.0 - out)

class MLP:
    def __init__(self, p, n, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W_h = np.random.uniform(-0.5, 0.5, (p, n))
        self.b_h = np.random.uniform(-0.5, 0.5, n)
        self.W_o = np.random.uniform(-0.5, 0.5, n)
        self.b_o = np.random.uniform(-0.5, 0.5)

    def forward(self, x):
        self.s_h = x @ self.W_h + self.b_h
        self.h = sigmoid(self.s_h)
        self.s_o = self.h @ self.W_o + self.b_o
        self.y = self.s_o
        return self.y

    def backward(self, x, target, eta):
        delta_o = self.y - target
        delta_h = delta_o * self.W_o * sigmoid_deriv(self.h)
        self.W_o -= eta * delta_o * self.h
        self.b_o -= eta * delta_o
        self.W_h -= eta * np.outer(x, delta_h)
        self.b_h -= eta * delta_h

    def train(self, X, y, eta=ETA, ee=EE, max_epochs=MAX_EPOCHS, verbose=False):
        n_samples = len(X)
        history = []
        for epoch in range(1, max_epochs + 1):
            idx = np.random.permutation(n_samples)
            for i in idx:
                self.forward(X[i])
                self.backward(X[i], y[i], eta)
            preds = np.array([self.forward(x) for x in X])
            es = np.mean((preds - y) ** 2)
            history.append(es)
            if verbose and epoch % 1000 == 0:
                print(f"  epoch {epoch}: Es = {es:.6e}")
            if es <= ee:
                return epoch, es, history
        return max_epochs, es, history

    def predict(self, X):
        return np.array([self.forward(x) for x in X])

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-12))

def recursive_forecast(model, series_norm, p, start_k, steps=30):
    window = list(series_norm[start_k - p:start_k])
    preds = []
    for _ in range(steps):
        x = np.array(window[-p:])
        y_hat = model.forward(x)
        preds.append(y_hat)
        window.append(y_hat)
    return np.array(preds)

def main():
    series = generate_series()
    print(f"Ряд сгенерирован. min={series.min():.4f}, max={series.max():.4f}")

    plt.figure(figsize=(12, 4))
    plt.plot(series, label='x(k)')
    plt.axvline(TRAIN_END - 0.5, color='r', linestyle='--', label='граница train/test')
    plt.title('Временной ряд (вариант 4)')
    plt.xlabel('k')
    plt.ylabel('x(k)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('01_series.png', dpi=120)
    plt.show()

    train_raw = series[:TRAIN_END]
    x_min, x_max = train_raw.min(), train_raw.max()
    print(f"\nНормализация: x_min={x_min:.6f}, x_max={x_max:.6f} (только train)")
    print("Обратный пересчёт: x = x_norm * (x_max - x_min) + x_min")
    print("min/max нельзя брать по всему ряду, иначе информация о тесте утекает в обучение.")

    series_norm = normalize(series, x_min, x_max)
    train_norm = series_norm[:TRAIN_END]

    X_train, y_train = make_windows(train_norm, P_VAR)
    X_all, y_all = make_windows(series_norm, P_VAR)
    mask_test = np.arange(P_VAR, K_TOTAL) >= TRAIN_END
    X_test = X_all[mask_test]
    y_test = y_all[mask_test]

    print(f"\nОбучающих примеров: {len(X_train)}")
    print(f"Тестовых примеров:  {len(X_test)}")

    print("\n===== Базовый эксперимент (p=8, N=4) =====")
    model = MLP(P_VAR, N_VAR, seed=SEED_BASE)
    epochs, es_final, hist = model.train(X_train, y_train, verbose=True)
    print(f"Эпох: {epochs}, итоговая Es = {es_final:.6e}")

    pred_train_n = model.predict(X_train)
    pred_test_n = model.predict(X_test)

    pred_train = denormalize(pred_train_n, x_min, x_max)
    pred_test = denormalize(pred_test_n, x_min, x_max)
    y_train_raw = denormalize(y_train, x_min, x_max)
    y_test_raw = denormalize(y_test, x_min, x_max)

    print("\nМетрики (исходная шкала):")
    print(f"  Train MAE  = {mae(y_train_raw, pred_train):.6f}")
    print(f"  Train RMSE = {rmse(y_train_raw, pred_train):.6f}")
    print(f"  Train sMAPE= {smape(y_train_raw, pred_train):.4f} %")
    print(f"  Test  MAE  = {mae(y_test_raw, pred_test):.6f}")
    print(f"  Test  RMSE = {rmse(y_test_raw, pred_test):.6f}")
    print(f"  Test  sMAPE= {smape(y_test_raw, pred_test):.4f} %")
    print("(sMAPE используется, т.к. ряд проходит через ноль)")

    print("\n===== Исследование размера окна p =====")
    p_list = [2, 4, P_VAR, 2 * P_VAR]
    results_p = {}
    for p in p_list:
        epochs_list, mae_list, fail_cnt = [], [], 0
        Xtr, ytr = make_windows(train_norm, p)
        Xte_all, yte_all = make_windows(series_norm, p)
        mask = np.arange(p, K_TOTAL) >= TRAIN_END
        Xte = Xte_all[mask]
        yte = yte_all[mask]
        yte_raw = denormalize(yte, x_min, x_max)

        for run in range(5):
            m = MLP(p, N_VAR, seed=SEED_BASE + run * 17)
            ep, es, _ = m.train(Xtr, ytr)
            pred = denormalize(m.predict(Xte), x_min, x_max)
            epochs_list.append(ep)
            mae_list.append(mae(yte_raw, pred))
            if ep >= MAX_EPOCHS:
                fail_cnt += 1
        results_p[p] = {
            'epochs_mean': np.mean(epochs_list),
            'epochs_std': np.std(epochs_list),
            'mae_mean': np.mean(mae_list),
            'mae_std': np.std(mae_list),
            'fails': fail_cnt
        }
        print(f"p={p}: epochs={np.mean(epochs_list):.0f}±{np.std(epochs_list):.0f}, "
              f"MAE={np.mean(mae_list):.5f}±{np.std(mae_list):.5f}, fails={fail_cnt}")

    print("\n===== Исследование числа нейронов N =====")
    n_list = [1, 2, N_VAR, 2 * N_VAR]
    results_n = {}
    Xtr, ytr = make_windows(train_norm, P_VAR)
    Xte_all, yte_all = make_windows(series_norm, P_VAR)
    mask = np.arange(P_VAR, K_TOTAL) >= TRAIN_END
    Xte = Xte_all[mask]
    yte = yte_all[mask]
    yte_raw = denormalize(yte, x_min, x_max)

    for n in n_list:
        epochs_list, mae_list, fail_cnt = [], [], 0
        for run in range(5):
            m = MLP(P_VAR, n, seed=SEED_BASE + run * 19)
            ep, es, _ = m.train(Xtr, ytr)
            pred = denormalize(m.predict(Xte), x_min, x_max)
            epochs_list.append(ep)
            mae_list.append(mae(yte_raw, pred))
            if ep >= MAX_EPOCHS:
                fail_cnt += 1
        results_n[n] = {
            'epochs_mean': np.mean(epochs_list),
            'epochs_std': np.std(epochs_list),
            'mae_mean': np.mean(mae_list),
            'mae_std': np.std(mae_list),
            'fails': fail_cnt
        }
        print(f"N={n}: epochs={np.mean(epochs_list):.0f}±{np.std(epochs_list):.0f}, "
              f"MAE={np.mean(mae_list):.5f}±{np.std(mae_list):.5f}, fails={fail_cnt}")

    best_model = model
    best_p, best_n = P_VAR, N_VAR

    print("\n===== Рекурсивный прогноз (30 шагов) =====")
    rec_norm = recursive_forecast(best_model, series_norm, best_p, TRAIN_END, steps=30)
    rec = denormalize(rec_norm, x_min, x_max)
    true_future = series[TRAIN_END:TRAIN_END + 30]
    abs_err = np.abs(rec - true_future)
    range_x = x_max - x_min
    step_5pct = np.where(abs_err > 0.05 * range_x)[0]
    first_bad = step_5pct[0] + 1 if len(step_5pct) else ">30"
    print(f"Ошибка > 5% размаха впервые на шаге: {first_bad}")

    full_pred_n = np.full(K_TOTAL, np.nan)
    for i, pred in enumerate(pred_train_n):
        full_pred_n[P_VAR + i] = pred
    for i, pred in enumerate(pred_test_n):
        full_pred_n[TRAIN_END + i] = pred
    full_pred = denormalize(full_pred_n, x_min, x_max)

    plt.figure(figsize=(12, 4))
    plt.plot(series, label='эталон', alpha=0.7)
    plt.plot(full_pred, label='одношаговый прогноз', alpha=0.8)
    plt.axvline(TRAIN_END - 0.5, color='r', linestyle='--')
    plt.title('Ряд + одношаговый прогноз')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('02_onestep.png', dpi=120)
    plt.show()

    m2 = MLP(2, N_VAR, seed=SEED_BASE)
    _, _, hist_p2 = m2.train(*make_windows(train_norm, 2))
    m16 = MLP(16, N_VAR, seed=SEED_BASE)
    _, _, hist_p16 = m16.train(*make_windows(train_norm, 16))

    plt.figure(figsize=(10, 5))
    plt.semilogy(hist, label=f'p={P_VAR}')
    plt.semilogy(hist_p2, label='p=2')
    plt.semilogy(hist_p16, label='p=16')
    plt.axhline(EE, color='r', linestyle='--', label='Ee')
    plt.title('Сходимость (Es)')
    plt.xlabel('эпоха')
    plt.ylabel('Es (log)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('03_convergence.png', dpi=120)
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_test_raw, pred_test, alpha=0.6)
    lims = [min(y_test_raw.min(), pred_test.min()),
            max(y_test_raw.max(), pred_test.max())]
    plt.plot(lims, lims, 'r--')
    plt.xlabel('эталон')
    plt.ylabel('прогноз')
    plt.title('Диаграмма рассеяния (тест)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('04_scatter.png', dpi=120)
    plt.show()

    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(true_future, label='эталон')
    ax[0].plot(rec, label='рекурсивный прогноз')
    ax[0].legend()
    ax[0].set_title('Рекурсивный прогноз (30 шагов)')
    ax[0].grid(True)
    ax[1].plot(abs_err, color='orange')
    ax[1].axhline(0.05 * range_x, color='r', linestyle='--', label='5% размаха')
    ax[1].set_xlabel('шаг')
    ax[1].set_ylabel('|ошибка|')
    ax[1].legend()
    ax[1].grid(True)
    plt.tight_layout()
    plt.savefig('05_recursive.png', dpi=120)
    plt.show()

    print("\n===== Демонстрация режима функционирования =====")
    for k in [20, 50, 100]:
        window = series[k - P_VAR:k]
        true_v = series[k]
        wn = normalize(window, x_min, x_max)
        pn = best_model.forward(wn)
        p = denormalize(pn, x_min, x_max)
        print(f"k={k}: pred={p:.5f}, true={true_v:.5f}, |err|={abs(p - true_v):.5f}")

    for k in [150, 170, 190]:
        window = series[k - P_VAR:k]
        true_v = series[k]
        wn = normalize(window, x_min, x_max)
        pn = best_model.forward(wn)
        p = denormalize(pn, x_min, x_max)
        print(f"k={k}: pred={p:.5f}, true={true_v:.5f}, |err|={abs(p - true_v):.5f}")

    print("\nПользовательские окна:")
    user_windows = [
        series[30:30 + P_VAR],
        np.array([0.5] * P_VAR),
        series[40:40 + P_VAR] + np.array([2.0] + [0] * (P_VAR - 1))
    ]
    for i, w in enumerate(user_windows):
        wn = normalize(w, x_min, x_max)
        pn = best_model.forward(wn)
        p = denormalize(pn, x_min, x_max)
        print(f"  окно {i + 1}: pred={p:.5f}")

    print("\nГотово. Графики сохранены.")

if __name__ == "__main__":
    main()