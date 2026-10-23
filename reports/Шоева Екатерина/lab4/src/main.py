import numpy as np
import matplotlib.pyplot as plt


A, B, C, D = 1.4, 0.38, 0.7, 0.85
P_VAR = 6
N_VAR = 5

N_POINTS = 200
TRAIN_END = 140

EE = 1e-4
MAX_EPOCHS = 20000
LEARNING_RATE = 0.1
SEEDS = [1, 2, 3, 4, 5]
BASE_SEED = 1

P_LIST = [2, 4, P_VAR, 2 * P_VAR]
N_LIST = [1, 2, N_VAR, 2 * N_VAR]

FORECAST_STEPS = 30
ERR_THRESHOLD_FRAC = 0.05

NOISE_STD_FRAC = 0.02
NOISE_EPOCHS = 3000

EPS = 1e-12


def generate_series(a=A, b=B, c=C, d=D, n=N_POINTS):
    k = np.arange(n)
    return a * np.cos(b * k) + c * np.sin(d * k)


def fit_minmax(train_part):
    return float(np.min(train_part)), float(np.max(train_part))


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def denormalize(x_norm, x_min, x_max):
    return x_norm * (x_max - x_min) + x_min


def make_windows(s, p, start, end):
    X, Y, K = [], [], []
    for t in range(start, end):
        X.append(s[t - p:t][::-1])
        Y.append(s[t])
        K.append(t)
    return np.array(X), np.array(Y), np.array(K)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLP:
    def __init__(self, p, N, lr=LEARNING_RATE, seed=0):
        self.p, self.N, self.lr = p, N, lr
        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.uniform(-0.5, 0.5, size=(p, N))
        self.b1 = self.rng.uniform(-0.5, 0.5, size=N)
        self.W2 = self.rng.uniform(-0.5, 0.5, size=N)
        self.b2 = float(self.rng.uniform(-0.5, 0.5))


    def forward(self, x):
        h = sigmoid(x @ self.W1 + self.b1)
        y = float(h @ self.W2 + self.b2)
        return h, y

    def predict(self, x):
        return self.forward(np.asarray(x, dtype=float))[1]

    def predict_batch(self, X):
        H = sigmoid(X @ self.W1 + self.b1)
        return H @ self.W2 + self.b2


    def train_step(self, x, d):
        h, y = self.forward(x)
        e = d - y
        delta_h = e * self.W2 * h * (1.0 - h)
        self.W2 += self.lr * e * h
        self.b2 += self.lr * e
        self.W1 += self.lr * np.outer(x, delta_h)
        self.b1 += self.lr * delta_h

    def mean_sq_error(self, X, Y):
        return float(np.mean((Y - self.predict_batch(X)) ** 2))


    def train(self, X, Y, ee=EE, max_epochs=MAX_EPOCHS):
        n = len(X)
        history = []
        Es = float("inf")
        for epoch in range(1, max_epochs + 1):
            for i in self.rng.permutation(n):
                self.train_step(X[i], Y[i])
            Es = self.mean_sq_error(X, Y)
            history.append(Es)
            if Es <= ee:
                return epoch, Es, history, True
        return max_epochs, Es, history, False


def mae(y, y_hat):
    return float(np.mean(np.abs(y - y_hat)))


def rmse(y, y_hat):
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def mape(y, y_hat):

    return float(np.mean(np.abs((y - y_hat) / (np.abs(y) + EPS))) * 100.0)


def smape(y, y_hat):

    denom = np.abs(y) + np.abs(y_hat) + EPS
    return float(np.mean(2.0 * np.abs(y - y_hat) / denom) * 100.0)


def all_metrics(y, y_hat):
    return {"MAE": mae(y, y_hat), "RMSE": rmse(y, y_hat), "sMAPE": smape(y, y_hat)}


class Dataset:


    def __init__(self, series, p):
        self.series = series
        self.p = p
        self.x_min, self.x_max = fit_minmax(series[:TRAIN_END])
        self.sn = normalize(series, self.x_min, self.x_max)
        self.Xtr, self.Ytr, self.Ktr = make_windows(self.sn, p, p, TRAIN_END)
        self.Xte, self.Yte, self.Kte = make_windows(self.sn, p, TRAIN_END, len(series))

    def denorm(self, v):
        return denormalize(v, self.x_min, self.x_max)


def run_config(series, p, N, seed, lr=LEARNING_RATE, ee=EE, max_epochs=MAX_EPOCHS):
    ds = Dataset(series, p)
    model = MLP(p, N, lr=lr, seed=seed)
    epochs, Es, history, converged = model.train(ds.Xtr, ds.Ytr, ee=ee, max_epochs=max_epochs)
    pred_tr = ds.denorm(model.predict_batch(ds.Xtr))
    pred_te = ds.denorm(model.predict_batch(ds.Xte))
    y_tr = ds.denorm(ds.Ytr)
    y_te = ds.denorm(ds.Yte)
    return {
        "p": p, "N": N, "seed": seed, "model": model, "ds": ds,
        "epochs": epochs, "Es": Es, "converged": converged, "history": history,
        "train": all_metrics(y_tr, pred_tr), "test": all_metrics(y_te, pred_te),
        "pred_tr": pred_tr, "pred_te": pred_te, "y_tr": y_tr, "y_te": y_te,
    }


def run_series(series, p, N, seeds=SEEDS, lr=LEARNING_RATE, verbose=True):
    results = []
    for s in seeds:
        r = run_config(series, p, N, s, lr=lr)
        results.append(r)
        if verbose:
            print(f"    p={p:>2} N={N:>2} seed={s}: эпох={r['epochs']:>5} "
                  f"Es={r['Es']:.3e} сошлось={r['converged']} "
                  f"testMAE={r['test']['MAE']:.4f}", flush=True)
    return results


def summarize(results):
    ep = np.array([r["epochs"] for r in results], dtype=float)
    tmae = np.array([r["test"]["MAE"] for r in results])
    trmae = np.array([r["train"]["MAE"] for r in results])
    return {
        "ep_mean": ep.mean(), "ep_std": ep.std(ddof=1), "ep_min": ep.min(), "ep_max": ep.max(),
        "te_mean": tmae.mean(), "te_std": tmae.std(ddof=1), "te_min": tmae.min(), "te_max": tmae.max(),
        "tr_mean": trmae.mean(),
        "n_fail": int(sum(not r["converged"] for r in results)),
        "n": len(results),
    }


def print_study_table(title, key, values, all_results):
    print(f"\n{title}")
    hdr = (f"{key:>4} | {'эпохи (ср.±ст.)':>18} | {'[min; max]':>14} | "
           f"{'train MAE':>9} | {'test MAE (ср.±ст.)':>20} | {'[min; max]':>18} | {'не сошлось':>10}")
    print(hdr)
    print("-" * len(hdr))
    for v, res in zip(values, all_results):
        s = summarize(res)
        print(f"{v:>4} | {s['ep_mean']:>10.1f}±{s['ep_std']:<7.1f} | "
              f"[{s['ep_min']:>5.0f};{s['ep_max']:>6.0f}] | {s['tr_mean']:>9.4f} | "
              f"{s['te_mean']:>11.4f}±{s['te_std']:<8.4f} | "
              f"[{s['te_min']:.4f};{s['te_max']:.4f}] | {s['n_fail']:>4}/{s['n']}")


def lr_sweep(series, lrs=(0.01, 0.03, 0.05, 0.1, 0.2, 0.3)):
    print("\nПодбор скорости обучения (базовая конфигурация p=%d, N=%d, 5 запусков):" % (P_VAR, N_VAR))
    print(f"{'eta':>6} | {'эпох (ср.)':>10} | {'сошлось':>8} | {'test MAE (ср.)':>14}")
    for lr in lrs:
        res = [run_config(series, P_VAR, N_VAR, s, lr=lr) for s in SEEDS]
        sm = summarize(res)
        print(f"{lr:>6} | {sm['ep_mean']:>10.1f} | {sm['n'] - sm['n_fail']:>4}/{sm['n']:<3} | {sm['te_mean']:>14.4f}")



def recursive_forecast(model, sn, p, start_k, steps):

    window = list(sn[start_k - p:start_k])
    preds = []
    for _ in range(steps):
        inp = np.array(window[::-1])
        y = model.predict(inp)
        preds.append(y)
        window = window[1:] + [y]
    return np.array(preds)


def predict_window(model, window_chrono, x_min, x_max):

    w = np.asarray(window_chrono, dtype=float)
    wn = normalize(w, x_min, x_max)
    y_norm = model.predict(wn[::-1])
    y_orig = denormalize(y_norm, x_min, x_max)
    return wn, y_norm, y_orig


def show_prediction(model, window_chrono, x_min, x_max, label, true_value=None, k=None):
    wn, y_norm, y_orig = predict_window(model, window_chrono, x_min, x_max)
    head = f"[{label}]" + (f" k={k}" if k is not None else "")
    print(head)
    print("  окно (исх. шкала):   ", np.array2string(np.asarray(window_chrono, dtype=float),
                                                       precision=4, separator=", "))
    print("  нормализованные входы:", np.array2string(wn, precision=4, separator=", "))
    out_of_range = np.any((wn < 0) | (wn > 1))
    if out_of_range:
        print("  (!) часть входов вне [0; 1] — значения вышли за диапазон обучающего участка")
    print(f"  прогноз (норм. шкала): {y_norm:.4f}")
    print(f"  прогноз (исх. шкала):  {y_orig:.4f}")
    if true_value is not None:
        print(f"  истинное x(k) = {true_value:.4f},  |ошибка| = {abs(true_value - y_orig):.4f}")
    print()


def show_from_series(model, series, p, x_min, x_max, k, label):
    show_prediction(model, series[k - p:k], x_min, x_max, label, true_value=series[k], k=k)


def demo_mode(model, series, p, x_min, x_max):
    print("\n" + "=" * 78)
    print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ: примеры работы обученной сети (p=%d)" % p)
    print("=" * 78)
    print("--- 3 окна из обучающего участка ---")
    for k in (20, 70, 120):
        show_from_series(model, series, p, x_min, x_max, k, "train")
    print("--- 3 окна из тестового участка ---")
    for k in (145, 170, 195):
        show_from_series(model, series, p, x_min, x_max, k, "test")
    print("--- пользовательские окна ---")
    custom = list(series[150 - p:150])
    custom[p // 2] += 3.0
    show_prediction(model, custom, x_min, x_max,
                    "пользовательское окно с выбросом (+3.0 в середине окна)",
                    true_value=series[150], k=150)
    show_prediction(model, [0.5, 0.9, 1.3, 1.1, 0.6, 0.1][:p] if p == 6 else list(series[:p]),
                    x_min, x_max, "пользовательское окно (произвольные значения)")


def interactive_mode(model, series, p, x_min, x_max):
    print("\n--- Интерактивный режим ---")
    print(f"Команды: 'k <номер>'  — взять окно из ряда (p <= k <= {len(series) - 1});")
    print(f"         'v v1 v2 ... v{p}' — ввести {p} значений в исходной шкале "
          f"в хронологическом порядке (старое -> новое);")
    print("         'q' — выход.")
    while True:
        raw = input(">>> ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            print("Выход из интерактивного режима.")
            break
        parts = raw.replace(",", " ").split()
        try:
            if parts[0].lower() == "k":
                k = int(parts[1])
                if not (p <= k < len(series)):
                    print(f"k должно быть в диапазоне [{p}; {len(series) - 1}].")
                    continue
                show_from_series(model, series, p, x_min, x_max, k, "из ряда")
            elif parts[0].lower() == "v":
                vals = [float(t) for t in parts[1:]]
                if len(vals) != p:
                    print(f"Нужно ровно {p} значений, получено {len(vals)}.")
                    continue
                show_prediction(model, vals, x_min, x_max, "пользовательское окно")
            else:
                print("Неизвестная команда. Примеры: 'k 150' или 'v 0.1 0.5 0.9 1.2 1.0 0.7'")
        except (ValueError, IndexError):
            print("Некорректный ввод. Примеры: 'k 150' или 'v 0.1 0.5 0.9 1.2 1.0 0.7'")


def manual_step_check(V=11, eta=0.1):
    X = np.array([0.2, 0.5, 0.8])
    d = 0.6
    W1 = np.array([[0.05 * V, -0.2],
                   [0.3, 0.05 * V],
                   [-0.1, 0.4]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([0.5, -0.2 * V])
    b2 = 0.2

    s = X @ W1 + b1
    h = sigmoid(s)
    y = float(h @ W2 + b2)
    e = d - y
    delta_h = e * W2 * h * (1 - h)
    W2_new = W2 + eta * e * h
    b2_new = b2 + eta * e
    W1_new = W1 + eta * np.outer(X, delta_h)
    b1_new = b1 + eta * delta_h

    np.set_printoptions(precision=6, suppress=True)
    print("\n" + "=" * 78)
    print(f"ПРОВЕРКА РУЧНОГО РАСЧЁТА (сеть 3-2-1, V={V}, eta={eta})")
    print("=" * 78)
    print("Forward:")
    print("  s (вход скрытых)  =", s)
    print("  h = sigma(s)      =", h)
    print(f"  y (линейный выход)= {y:.6f}")
    print(f"  ошибка e = d - y  = {e:.6f}   (E = 1/2 e^2 = {0.5 * e * e:.6f})")
    print("Backward:")
    print(f"  delta_out         = e = {e:.6f}")
    print("  delta_hidden      =", delta_h, "  (= e * W_out * h * (1-h))")
    print("Новые веса (eta = %s):" % eta)
    print("  W_out_new         =", W2_new, f"   b_out_new = {b2_new:.6f}")
    print("  W1_new (строки — входы, столбцы — скрытые нейроны):")
    print(W1_new)
    print("  b1_new            =", b1_new)


def plot_series(series, filename="lab4_series.png"):
    k = np.arange(len(series))
    plt.figure(figsize=(9, 4.5))
    plt.plot(k, series, color="tab:blue", label="x(k)")
    plt.axvline(TRAIN_END - 0.5, color="k", linestyle="--", label="граница train/test")
    plt.axvspan(TRAIN_END - 0.5, len(series), color="orange", alpha=0.12)
    plt.text(TRAIN_END / 2, series.max() * 0.95, "обучающий участок", ha="center")
    plt.text((TRAIN_END + len(series)) / 2, series.max() * 0.95, "тестовый", ha="center")
    plt.title("Временной ряд x(k) = a·cos(b·k) + c·sin(d·k), вариант 11")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.legend(loc="lower left"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"Сохранён график: {filename}")


def plot_series_forecast(base, series, filename="lab4_series_forecast.png"):
    ds = base["ds"]
    k = np.arange(len(series))
    plt.figure(figsize=(10, 4.8))
    plt.plot(k, series, color="tab:blue", linewidth=1.5, label="эталон x(k)")
    plt.plot(ds.Ktr, base["pred_tr"], "--", color="tab:green", linewidth=1.2,
             label="одношаговый прогноз (train)")
    plt.plot(ds.Kte, base["pred_te"], "--", color="tab:red", linewidth=1.4,
             label="одношаговый прогноз (test)")
    plt.axvline(TRAIN_END - 0.5, color="k", linestyle=":", label="граница train/test")
    plt.title(f"Одношаговый прогноз сети {base['p']}-{base['N']}-1 на обучающем и тестовом участках")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Сохранён график: {filename}")


def plot_convergence(curves, filename="lab4_convergence.png"):
    plt.figure(figsize=(8, 5))
    for label, hist in curves:
        plt.plot(range(1, len(hist) + 1), hist, label=label)
    plt.axhline(EE, color="gray", linestyle="--", linewidth=0.9, label=f"Ee = {EE:g}")
    plt.yscale("log")
    plt.title("Сходимость: средняя ошибка Es по эпохам\n(лог. шкала по Y; по X — тоже лог., т.к. p=2 идёт 20 000 эпох)")
    plt.xlabel("Эпоха"); plt.ylabel("Es (норм. шкала)")
    plt.xscale("log")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"Сохранён график: {filename}")


def plot_scatter(base, filename="lab4_scatter.png"):
    y, yh = base["y_te"], base["pred_te"]
    lo, hi = min(y.min(), yh.min()), max(y.max(), yh.max())
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y, yh, s=22, color="tab:red", alpha=0.8, label="тестовые точки")
    plt.plot([lo, hi], [lo, hi], "k--", label="y = x")
    plt.title("Тестовый участок: эталон – прогноз")
    plt.xlabel("эталон"); plt.ylabel("прогноз сети")
    plt.legend(); plt.grid(alpha=0.3); plt.axis("equal")
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"Сохранён график: {filename}")


def plot_recursive(ks, ref, rec, errs, thr, step_exceed, label, filename="lab4_recursive.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(ks, ref, "o-", color="tab:blue", markersize=3, label="эталон")
    axes[0].plot(ks, rec, "s--", color="tab:red", markersize=3, label="рекурсивный прогноз")
    axes[0].set_title(f"Рекурсивный прогноз на {len(ks)} шагов ({label})")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("x(k)")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    steps = np.arange(1, len(errs) + 1)
    axes[1].plot(steps, errs, "o-", color="tab:purple", markersize=3, label="|ошибка|")
    axes[1].axhline(thr, color="gray", linestyle="--", label=f"5 % размаха = {thr:.3f}")
    if step_exceed is not None:
        axes[1].axvline(step_exceed, color="red", linestyle=":", label=f"превышение на шаге {step_exceed}")
    axes[1].set_ylim(0, max(thr, errs.max()) * 1.45)
    axes[1].set_title("Рост абсолютной ошибки по шагам")
    axes[1].set_xlabel("номер шага"); axes[1].set_ylabel("|x - x^|")
    axes[1].legend(loc="upper left", fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"Сохранён график: {filename}")


def plot_studies(p_vals, p_res, n_vals, n_res, filename="lab4_studies.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, vals, res, xl in ((axes[0], p_vals, p_res, "размер окна p"),
                              (axes[1], n_vals, n_res, "число скрытых нейронов N")):
        m = [summarize(r)["te_mean"] for r in res]
        s = [summarize(r)["te_std"] for r in res]
        tr = [summarize(r)["tr_mean"] for r in res]
        ax.errorbar(range(len(vals)), m, yerr=s, fmt="o-", capsize=4, color="tab:red", label="test MAE (ср.±ст.)")
        ax.plot(range(len(vals)), tr, "s--", color="tab:green", label="train MAE (ср.)")
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(vals)
        if ax is axes[0]:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, max(max(m), max(tr)) * 1.6)
        ax.set_xlabel(xl); ax.set_ylabel("MAE (исходная шкала)")
        ax.grid(alpha=0.3, which="both"); ax.legend()
    axes[0].set_title("Влияние размера окна p (N = %d)" % N_VAR)
    axes[1].set_title("Влияние числа нейронов N (p = %d)" % P_VAR)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()
    print(f"Сохранён график: {filename}")

def noise_experiment(series, N_values=(N_VAR, 4 * N_VAR), p=P_VAR, seeds=SEEDS):
    print("\n" + "=" * 78)
    print("ЭКСПЕРИМЕНТ С ШУМОМ (обучение на зашумлённом train, тест на ЧИСТОМ ряде)")
    print("=" * 78)
    rng = np.random.default_rng(12345)
    x_min0, x_max0 = fit_minmax(series[:TRAIN_END])
    sigma = NOISE_STD_FRAC * (x_max0 - x_min0)
    noisy = series.copy()
    noisy[:TRAIN_END] += rng.normal(0.0, sigma, size=TRAIN_END)
    print(f"Гауссов шум: sigma = {sigma:.4f} ({NOISE_STD_FRAC * 100:.0f} % размаха train), "
          f"p={p}; для сопоставимости обе серии обучаются фиксированные {NOISE_EPOCHS} эпох (Ee отключён).")
    print(f"{'обучение':>10} | {'N':>3} | {'train MAE (свои цели)':>21} | "
          f"{'train MAE (чистый ряд)':>22} | {'test MAE (чистый)':>17}")
    table = {}
    for with_noise in (False, True):
        ser_train = noisy if with_noise else series
        x_min, x_max = fit_minmax(ser_train[:TRAIN_END])
        sn_tr = normalize(ser_train, x_min, x_max)
        sn_clean = normalize(series, x_min, x_max)
        Xtr, Ytr, _ = make_windows(sn_tr, p, p, TRAIN_END)
        Xtr_c, Ytr_c, _ = make_windows(sn_clean, p, p, TRAIN_END)
        Xte, Yte, _ = make_windows(sn_clean, p, TRAIN_END, len(series))
        for N in N_values:
            own, clean_tr, te = [], [], []
            for sd in seeds:
                m = MLP(p, N, seed=sd)
                m.train(Xtr, Ytr, ee=0.0, max_epochs=NOISE_EPOCHS)
                dn = lambda v: denormalize(v, x_min, x_max)
                own.append(mae(dn(Ytr), dn(m.predict_batch(Xtr))))
                clean_tr.append(mae(dn(Ytr_c), dn(m.predict_batch(Xtr_c))))
                te.append(mae(dn(Yte), dn(m.predict_batch(Xte))))
            table[(with_noise, N)] = (np.mean(own), np.mean(clean_tr), np.mean(te))
            tag = "с шумом" if with_noise else "без шума"
            print(f"{tag:>10} | {N:>3} | {np.mean(own):>21.4f} | {np.mean(clean_tr):>22.4f} | {np.mean(te):>17.4f}",
                  flush=True)
    for N in N_values:
        r = table[(True, N)][2] / table[(False, N)][2]
        print(f"N={N}: шум в обучающих данных увеличил test MAE в {r:.2f} раза "
              f"(gap test - train по чистому ряду при шуме: {table[(True, N)][2] - table[(True, N)][1]:+.4f})")
    print("Вывод по эксперименту смотрите по числам выше: сравните (а) рост test MAE из-за шума, "
          "(б) разницу между малым и большим N при одном и том же шуме и (в) gap test - train.")

def main():
    series = generate_series()
    print(f"Ряд сгенерирован: {len(series)} точек, min={series.min():.4f}, max={series.max():.4f}, "
          f"размах={series.max() - series.min():.4f}")
    crosses_zero = bool(np.any(np.sign(series[:-1]) * np.sign(series[1:]) < 0))
    print(f"Ряд пересекает ноль: {'да -> вместо MAPE используем sMAPE' if crosses_zero else 'нет'}")
    plot_series(series)

    x_min, x_max = fit_minmax(series[:TRAIN_END])
    sn = normalize(series, x_min, x_max)
    test_norm = sn[TRAIN_END:]
    n_out = int(np.sum((test_norm < 0) | (test_norm > 1)))
    print(f"\nНормализация по train: x_min={x_min:.4f}, x_max={x_max:.4f}")
    print("  x_norm = (x - x_min)/(x_max - x_min);   x = x_norm*(x_max - x_min) + x_min")
    print(f"  тестовые значения в норм. шкале: [{test_norm.min():.4f}; {test_norm.max():.4f}], "
          f"вне [0;1]: {n_out} из {len(test_norm)}")

    ds_var = Dataset(series, P_VAR)
    print(f"\nСкользящее окно p={P_VAR}: обучающих примеров = {len(ds_var.Xtr)} "
          f"(цели k={P_VAR}..{TRAIN_END - 1}), тестовых = {len(ds_var.Xte)} "
          f"(цели k={TRAIN_END}..{len(series) - 1})")

    lr_sweep(series)

    print("\n" + "=" * 78)
    print(f"БАЗОВЫЙ ЭКСПЕРИМЕНТ: p={P_VAR}, N={N_VAR}, eta={LEARNING_RATE}, seed={BASE_SEED}")
    print("=" * 78)
    base = run_config(series, P_VAR, N_VAR, BASE_SEED)
    print(f"Эпох: {base['epochs']}  (критерий Es<=Ee достигнут: {base['converged']})")
    print(f"Итоговая Es = {base['Es']:.3e}  (норм. шкала)")
    print(f"{'':<8}{'MAE':>10}{'RMSE':>10}{'sMAPE,%':>10}")
    for name in ("train", "test"):
        m = base[name]
        print(f"{name:<8}{m['MAE']:>10.4f}{m['RMSE']:>10.4f}{m['sMAPE']:>10.3f}")
    mp_te = mape(base["y_te"], base["pred_te"])
    rel = np.abs(base["y_te"] - base["pred_te"]) / (np.abs(base["y_te"]) + EPS) * 100.0
    j = int(np.argmax(rel))
    print(f"(для сравнения, MAPE на test = {mp_te:.1f} %, но отдельная точка может дать {rel[j]:.0f} % "
          f"при |x| = {abs(base['y_te'][j]):.4f} (≈0) — поэтому MAPE нестабильна, используем sMAPE)")

    print("\n" + "=" * 78)
    print(f"ИССЛЕДОВАНИЕ ОКНА (N={N_VAR}), p in {P_LIST}")
    print("=" * 78)
    p_results = []
    for p in P_LIST:
        print(f"  --- p = {p} ---", flush=True)
        p_results.append(run_series(series, p, N_VAR))
    print_study_table(f"Таблица: влияние окна p (N={N_VAR}, 5 запусков)", "p", P_LIST, p_results)

    print("\n" + "=" * 78)
    print(f"ИССЛЕДОВАНИЕ СКРЫТОГО СЛОЯ (p={P_VAR}), N in {N_LIST}")
    print("=" * 78)
    n_results = []
    for N in N_LIST:
        print(f"  --- N = {N} ---", flush=True)
        n_results.append(run_series(series, P_VAR, N))
    print_study_table(f"Таблица: влияние числа нейронов N (p={P_VAR}, 5 запусков)", "N", N_LIST, n_results)

    print()
    plot_series_forecast(base, series)
    curves = [(f"базовая p={P_VAR}, N={N_VAR}", base["history"]),
              (f"p=2, N={N_VAR}", p_results[0][0]["history"]),
              (f"p={2 * P_VAR}, N={N_VAR}", p_results[3][0]["history"])]
    plot_convergence(curves)
    plot_scatter(base)
    plot_studies(P_LIST, p_results, N_LIST, n_results)

    candidates = {}
    for res in p_results + n_results:
        key = (res[0]["p"], res[0]["N"])
        candidates[key] = res
    best_key = min(candidates, key=lambda kk: summarize(candidates[kk])["te_mean"])
    best_res = candidates[best_key]
    best_run = min(best_res, key=lambda r: r["test"]["MAE"])
    bp, bN = best_key
    print("\n" + "=" * 78)
    print(f"МНОГОШАГОВЫЙ ПРОГНОЗ: лучшая по среднему test MAE конфигурация p={bp}, N={bN} "
          f"(средний test MAE = {summarize(best_res)['te_mean']:.4f}; берём запуск seed={best_run['seed']})")
    print("=" * 78)
    ds_b = best_run["ds"]
    pred_n = recursive_forecast(best_run["model"], ds_b.sn, bp, TRAIN_END, FORECAST_STEPS)
    rec = ds_b.denorm(pred_n)
    ks = np.arange(TRAIN_END, TRAIN_END + FORECAST_STEPS)
    ref = series[ks]
    errs = np.abs(ref - rec)
    thr = ERR_THRESHOLD_FRAC * (series.max() - series.min())
    exceed = np.where(errs > thr)[0]
    step_exceed = int(exceed[0]) + 1 if len(exceed) else None
    one_step_err = np.abs(best_run["pred_te"][:FORECAST_STEPS] - ref)
    print(f"Порог 5 % размаха ряда = {thr:.4f}")
    print(f"Одношаговый прогноз на тех же {FORECAST_STEPS} точках: MAE = {one_step_err.mean():.4f}, "
          f"макс. |ошибка| = {one_step_err.max():.4f}")
    print(f"Рекурсивный прогноз: MAE = {errs.mean():.4f}, макс. |ошибка| = {errs.max():.4f}")
    if step_exceed is None:
        print(f"Ошибка НЕ превышает 5 % размаха ни на одном из {FORECAST_STEPS} шагов.")
    else:
        print(f"Ошибка впервые превышает 5 % размаха на шаге {step_exceed} (k={TRAIN_END + step_exceed - 1}).")
    print("Ошибка по шагам:", np.array2string(errs, precision=4, separator=", ", max_line_width=100))
    plot_recursive(ks, ref, rec, errs, thr, step_exceed, f"p={bp}, N={bN}")

    n_full = len(series) - TRAIN_END
    rec_full = ds_b.denorm(recursive_forecast(best_run["model"], ds_b.sn, bp, TRAIN_END, n_full))
    errs_full = np.abs(series[TRAIN_END:] - rec_full)
    ex_full = np.where(errs_full > thr)[0]
    msg = (f"впервые превышает 5 %% размаха на шаге %d" % (int(ex_full[0]) + 1)) if len(ex_full) \
        else "не превышает 5 % размаха"
    print(f"[Дополнительно] прогноз на весь тест ({n_full} шагов): макс. |ошибка| = {errs_full.max():.4f}, "
          f"ошибка {msg}; MAE по шагам 1-30 = {errs_full[:30].mean():.4f}, по шагам 31-{n_full} = "
          f"{errs_full[30:].mean():.4f}")

    noise_experiment(series)

    manual_step_check(V=11, eta=0.1)

    demo_mode(base["model"], series, P_VAR, base["ds"].x_min, base["ds"].x_max)

    choice = input("Запустить интерактивный режим (базовая сеть p=%d, N=%d)? (y/Enter — пропустить): "
                   % (P_VAR, N_VAR)).strip().lower()
    if choice == "y":
        interactive_mode(base["model"], series, P_VAR, base["ds"].x_min, base["ds"].x_max)
    else:
        print("Интерактивный режим пропущен.")


if __name__ == "__main__":
    main()

