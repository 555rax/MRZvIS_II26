import numpy as np
import matplotlib.pyplot as plt


# ------------------------------
# 1. Генерация временного ряда и параметры
# ------------------------------
def series():
    k = np.arange(200, dtype=float)
    return 1.2 * np.cos(0.28 * k) + 0.6 * np.sin(0.65 * k)


def fit_scale(values):
    # Min-Max только по обучающей выборке (первые 140 точек)
    lo = float(values[:140].min())
    hi = float(values[:140].max())
    return lo, hi


def normalize(v, lo, hi):
    return (np.asarray(v) - lo) / (hi - lo)


def denormalize(v, lo, hi):
    return lo + (hi - lo) * np.asarray(v)


def windows(x, p):
    # Формирование скользящих окон
    xx = np.asarray([x[k - p:k] for k in range(p, len(x))], dtype=float)
    yy = np.asarray(x[p:], dtype=float)
    inds = np.arange(p, len(x))

    # Разделение на train (k=6..139) и test (k=140..199)
    train_mask = inds < 140
    test_mask = inds >= 140

    return (xx[train_mask], yy[train_mask], inds[train_mask],
            xx[test_mask], yy[test_mask], inds[test_mask])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    # Симметричная MAPE (sMAPE) для рядов, проходящих через ноль
    den = np.abs(y_true) + np.abs(y_pred)
    smape = 100 * np.mean(np.where(den > 1e-12, 2 * np.abs(err) / np.maximum(den, 1e-12), 0.0))

    return {"MAE": float(mae), "RMSE": float(rmse), "sMAPE": float(smape)}


# ------------------------------
# 2. Класс MLP для прогнозирования (p-N-1)
# ------------------------------
class ForecastMLP:
    def __init__(self, p=6, n=4, seed=6, lr=0.08):
        rng = np.random.RandomState(seed)
        # Инициализация весов U(-0.1, 0.1)
        self.W1 = rng.uniform(-0.1, 0.1, (p, n))
        self.b1 = rng.uniform(-0.1, 0.1, n)
        self.W2 = rng.uniform(-0.1, 0.1, n)
        self.b2 = float(rng.uniform(-0.1, 0.1))
        self.p, self.n, self.lr = p, n, lr

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        h = sigmoid(np.dot(x, self.W1) + self.b1)
        return np.dot(h, self.W2) + self.b2

    def step(self, x, y):
        h = sigmoid(np.dot(x, self.W1) + self.b1)
        pred = float(np.dot(h, self.W2) + self.b2)

        # Градиент для MSE: L = 1/2 (y - t)^2
        delta = pred - y
        dh = delta * self.W2 * h * (1 - h)

        # Обновление весов
        self.W2 -= self.lr * delta * h
        self.b2 -= self.lr * delta
        self.W1 -= self.lr * np.outer(x, dh)
        self.b1 -= self.lr * dh

    def fit(self, X, y, ee=1e-4, max_epochs=20000, seed=6):
        rng = np.random.RandomState(54321 + seed)
        order = np.arange(len(y))
        history = []

        for epoch in range(1, max_epochs + 1):
            rng.shuffle(order)
            for j in order:
                self.step(X[j], y[j])

            pred = self.predict(X)
            es = float(0.5 * np.mean((pred - y) ** 2))
            history.append(es)

            if es <= ee:
                break

        return {"epochs": epoch, "Es": es, "converged": es <= ee, "history": np.asarray(history)}


# ------------------------------
# 3. Функция эксперимента
# ------------------------------
def run_experiment(p, n, seed, raw, lo, hi, max_epochs=20000):
    scaled = normalize(raw, lo, hi)
    xtr, ytr, itr, xte, yte, ite = windows(scaled, p)

    model = ForecastMLP(p, n, seed=seed, lr=0.08)
    result = model.fit(xtr, ytr, ee=1e-4, max_epochs=max_epochs, seed=seed)

    train_pred = denormalize(model.predict(xtr), lo, hi)
    test_pred = denormalize(model.predict(xte), lo, hi)

    train_metrics = metrics(raw[itr], train_pred)
    test_metrics = metrics(raw[ite], test_pred)

    return model, result, train_metrics, test_metrics, ite, test_pred


# ------------------------------
# 4. Основной скрипт
# ------------------------------
def main():
    raw = series()
    lo, hi = fit_scale(raw)
    SEEDS = [6, 7, 11, 14, 18]

    # --- 3. Базовый эксперимент (p=6, N=4, seed=6) ---
    print("=" * 60)
    print("Базовый эксперимент (p=6, N=4, seed=6)")
    model_base, res_base, tr_base, te_base, ite_base, pred_base = run_experiment(6, 4, 6, raw, lo, hi)
    print(f"Эпохи: {res_base['epochs']}, Es: {res_base['Es']:.6f}")
    print(f"Test MAE: {te_base['MAE']:.4f}, Test RMSE: {te_base['RMSE']:.4f}, Test sMAPE: {te_base['sMAPE']:.2f}%")

    # График 1: Ряд и одношаговый прогноз
    plt.figure(figsize=(10, 4))
    plt.plot(raw, label='True series')
    plt.plot(ite_base, pred_base, label='Test one-step', alpha=0.8)
    plt.axvline(x=139.5, color='gray', linestyle='--', label='train / test split')
    plt.title('Variant 6: series and one-step predictions')
    plt.xlabel('k')
    plt.ylabel('x(k)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 4. Исследование длины окна p (N=4) ---
    print("\n" + "=" * 60)
    print("Исследование длины окна p (N=4)")
    p_list = [2, 4, 6, 12]
    p_results = {}
    p_histories = {}

    for p in p_list:
        maes = []
        epochs_list = []
        conv_count = 0
        for seed in SEEDS:
            _, res, _, te, _, _ = run_experiment(p, 4, seed, raw, lo, hi)
            maes.append(te['MAE'])
            epochs_list.append(res['epochs'])
            if res['converged']: conv_count += 1
            if seed == 6:
                p_histories[p] = res['history']
        p_results[p] = {"mae_mean": np.mean(maes), "mae_std": np.std(maes), "conv": conv_count}
        print(
            f"p={p:<2} | Эпохи (mean): {np.mean(epochs_list):.1f} | Test MAE: {p_results[p]['mae_mean']:.5f} ± {p_results[p]['mae_std']:.5f} | Сходимость: {conv_count}/5")

    # --- 5. Исследование скрытого слоя N (p=6) ---
    print("\n" + "=" * 60)
    print("Исследование скрытого слоя N (p=6)")
    n_list = [1, 2, 4, 8]
    n_results = {}

    for n in n_list:
        maes = []
        epochs_list = []
        conv_count = 0
        for seed in SEEDS:
            _, res, _, te, _, _ = run_experiment(6, n, seed, raw, lo, hi)
            maes.append(te['MAE'])
            epochs_list.append(res['epochs'])
            if res['converged']: conv_count += 1
        n_results[n] = {"mae_mean": np.mean(maes), "mae_std": np.std(maes), "conv": conv_count}
        print(
            f"N={n:<2} | Эпохи (mean): {np.mean(epochs_list):.1f} | Test MAE: {n_results[n]['mae_mean']:.5f} ± {n_results[n]['mae_std']:.5f} | Сходимость: {conv_count}/5")

    # График 2: Влияние p и N
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar([str(x) for x in p_list], [p_results[p]['mae_mean'] for p in p_list],
            yerr=[p_results[p]['mae_std'] for p in p_list], capsize=5)
    ax1.set_title("Window size p vs prediction accuracy")
    ax1.set_xlabel("Window size p")
    ax1.set_ylabel("Test MAE")

    ax2.bar([str(x) for x in n_list], [n_results[n]['mae_mean'] for n in n_list],
            yerr=[n_results[n]['mae_std'] for n in n_list], capsize=5)
    ax2.set_title("Hidden size N vs prediction accuracy")
    ax2.set_xlabel("Hidden size N")
    ax2.set_ylabel("Test MAE")

    plt.tight_layout()
    plt.show()

    # График 3: Динамика обучения
    plt.figure(figsize=(8, 4))
    for p in [2, 6, 12]:
        plt.plot(p_histories[p], label=f'p={p}')
    plt.axhline(y=1e-4, color='gray', linestyle='--', label='Ee=1e-4')
    plt.yscale('log')
    plt.xlim(0, 20000)
    plt.xlabel('Epoch')
    plt.ylabel('Mean 1/2 SSE')
    plt.title('Learning dynamics by window size')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 6. Рекурсивный прогноз (Лучший запуск: p=4, N=4, seed=7) ---
    print("\n" + "=" * 60)
    print("Рекурсивный прогноз 30 шагов (p=4, N=4, seed=7)")
    model_rec, _, _, _, _, _ = run_experiment(4, 4, 7, raw, lo, hi)

    # Рекурсивный прогноз
    scaled = normalize(raw, lo, hi)
    history = list(scaled[:140])
    rec_preds = []
    for _ in range(30):
        v_norm = float(model_rec.predict(np.asarray(history[-4:])))
        rec_preds.append(v_norm)
        history.append(v_norm)

    rec_preds_raw = denormalize(np.array(rec_preds), lo, hi)
    true_vals = raw[140:170]
    abs_errors = np.abs(rec_preds_raw - true_vals)

    # График 4: Рекурсивный прогноз
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(140, 170), true_vals, label='True values')
    plt.plot(np.arange(140, 170), rec_preds_raw, label='Recursive forecast')
    plt.title('Recursive 30-step forecast (p=4, N=4, seed=7)')
    plt.xlabel('k')
    plt.ylabel('x(k)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # График 5: Накопление ошибки
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, 31), abs_errors, marker='.', label='Absolute error')
    plt.axhline(y=0.05 * (hi - lo), color='gray', linestyle='--', label='5% of series range')
    plt.xlabel('Forecast step')
    plt.ylabel('Absolute error')
    plt.title('Error accumulation over steps')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 7. Дополнительное исследование: шум ---
    print("\n" + "=" * 60)
    print("Исследование устойчивости к шуму (sigma=0.03)")
    noise_results = {"no_noise": [], "noise": []}

    scaled = normalize(raw, lo, hi)
    xtr, ytr, itr, xte, yte, ite = windows(scaled, 6)

    for seed in SEEDS:
        # Без шума
        m_clean = ForecastMLP(6, 4, seed=seed)
        m_clean.fit(xtr, ytr, seed=seed)
        pred_clean = denormalize(m_clean.predict(xte), lo, hi)
        noise_results["no_noise"].append(metrics(raw[ite], pred_clean)['MAE'])

        # С шумом
        rng = np.random.RandomState(1000 + seed)
        noisy_xtr = xtr + rng.normal(loc=0.0, scale=0.03, size=xtr.shape)
        m_noisy = ForecastMLP(6, 4, seed=seed)
        m_noisy.fit(noisy_xtr, ytr, seed=seed)
        pred_noisy = denormalize(m_noisy.predict(xte), lo, hi)
        noise_results["noise"].append(metrics(raw[ite], pred_noisy)['MAE'])

    # График 6: Шум
    x_pos = np.arange(len(SEEDS))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x_pos - width / 2, noise_results["no_noise"], width, label='No noise')
    plt.bar(x_pos + width / 2, noise_results["noise"], width, label='Noise sigma=0.03')
    plt.xticks(x_pos, SEEDS)
    plt.xlabel('Seed')
    plt.ylabel('Test MAE')
    plt.title('Optional robustness test: noisy training inputs')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- 8. Режим функционирования (вывод таблицы) ---
    print("\n" + "=" * 60)
    print("Работа в режиме функционирования (Базовая модель p=6, N=4)")
    print(f"{'k / тип':<12} | {'y норм.':<8} | {'y исход.':<8} | {'Истинное':<8} | {'Абс. ошибка':<8}")
    print("-" * 60)

    def inference(model, raw_data, lo, hi, k):
        if not (6 <= k <= 199): raise ValueError("Неверный k")
        window_raw = raw_data[k - 6:k]
        actual = raw_data[k]
        xn = normalize(window_raw, lo, hi)
        pred_norm = float(model.predict(xn))
        pred = float(denormalize(pred_norm, lo, hi))
        return pred_norm, pred, actual, abs(pred - actual)

    for k in [20, 40, 100, 140, 160, 199]:
        pn, p, a, e = inference(model_base, raw, lo, hi, k)
        print(f"k={k:<10} | {pn:>8.4f} | {p:>8.4f} | {a:>8.4f} | {e:>8.4f}")


if __name__ == "__main__":
    main()