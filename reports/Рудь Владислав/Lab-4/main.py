# -*- coding: utf-8 -*-
"""
Лабораторная работа №4. Вариант 15.
MLP p-N-1 для прогнозирования временного ряда.

Запуск:  python lab4.py
Результат: все графики в ./results/, полный лог в консоли.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


A, B, C, D   = 1.6, 0.26, 0.8, 0.63
P_VAR, N_VAR = 8, 5
K_TOTAL      = 200
K_TRAIN_LAST = 139


LR           = 0.1
TOL          = 1e-4
MAX_EPOCHS   = 20000
N_SEEDS      = 5

os.makedirs("results", exist_ok=True)



def generate_series(a, b, c, d, n=K_TOTAL):
    k = np.arange(n)
    return a * np.cos(b * k) + c * np.sin(d * k)



def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def denormalize(xn, xmin, xmax):
    return xn * (xmax - xmin) + xmin



def make_windows(x, p):

    n = len(x)
    X = np.empty((n - p, p))
    y = np.empty(n - p)
    for i in range(n - p):
        X[i] = x[i:i + p]
        y[i] = x[i + p]
    return X, y


def split_train_test(X, y, p):

    k_targets = np.arange(p, p + len(y))
    mask = k_targets <= K_TRAIN_LAST
    return X[mask], y[mask], X[~mask], y[~mask]



def sigmoid(z):
    z = np.clip(z, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-z))


def init_mlp(p, N, seed):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(0.0, 0.1, (N, p))     # (N, p)
    b1 = np.zeros(N)
    w2 = rng.normal(0.0, 0.1, N)          # (N,)
    b2 = 0.0
    return W1, b1, w2, b2


def forward_one(x, W1, b1, w2, b2):
    z = W1 @ x + b1
    h = sigmoid(z)
    y = float(w2 @ h + b2)
    return z, h, y


def backward_one(x, d, h, y, w2):

    dy  = y - d
    dw2 = dy * h
    db2 = dy
    dh  = dy * w2
    dz  = dh * h * (1.0 - h)
    dW1 = np.outer(dz, x)
    db1 = dz.copy()
    return dW1, db1, dw2, db2


def forward_batch(X, W1, b1, w2, b2):
    Z = X @ W1.T + b1
    H = sigmoid(Z)
    y = H @ w2 + b2
    return H, y


def train_mlp(X, y, N, lr=LR, max_epochs=MAX_EPOCHS, tol=TOL, seed=0):

    p = X.shape[1]
    W1, b1, w2, b2 = init_mlp(p, N, seed)
    n = len(y)
    rng = np.random.default_rng(seed + 9999)
    hist = []
    converged = False
    ep = 0
    for ep in range(max_epochs):
        order = rng.permutation(n)
        for i in order:
            z, h, yp = forward_one(X[i], W1, b1, w2, b2)
            gW1, gb1, gw2, gb2 = backward_one(X[i], y[i], h, yp, w2)
            W1 -= lr * gW1
            b1 -= lr * gb1
            w2 -= lr * gw2
            b2 -= lr * gb2
        _, ya = forward_batch(X, W1, b1, w2, b2)
        Es = float(np.mean((ya - y) ** 2))
        hist.append(Es)
        if Es <= tol:
            converged = True
            break
    return (W1, b1, w2, b2), hist, converged, ep + 1


def predict(X, W1, b1, w2, b2):
    _, y = forward_batch(X, W1, b1, w2, b2)
    return y



def mae(a, b):   return float(np.mean(np.abs(a - b)))
def rmse(a, b):  return float(np.sqrt(np.mean((a - b) ** 2)))


def smape(a, b):
    den = np.abs(a) + np.abs(b)
    return float(100.0 * np.mean(2.0 * np.abs(a - b) / np.maximum(den, 1e-12)))


def mape(a, b):
    return float(100.0 * np.mean(np.abs((a - b) / np.where(np.abs(a) > 1e-8, a, 1e-8))))


def metrics_report(y_true_n, y_pred_n, xmin, xmax):
    yt = denormalize(y_true_n, xmin, xmax)
    yp = denormalize(y_pred_n, xmin, xmax)
    return dict(MAE=mae(yt, yp), RMSE=rmse(yt, yp),
                sMAPE=smape(yt, yp), MAPE=mape(yt, yp))



def recursive_forecast(W1, b1, w2, b2, x_norm, p, k_start, n_steps):
    window = x_norm[k_start - p:k_start].copy()
    out = []
    for _ in range(n_steps):
        _, _, yp = forward_one(window, W1, b1, w2, b2)
        out.append(yp)
        window = np.roll(window, -1)
        window[-1] = yp
    return np.array(out)



def functioning(window_raw, W1, b1, w2, b2, xmin, xmax, x_series, k_target=None):
    win_n = normalize(np.asarray(window_raw, dtype=float), xmin, xmax)
    _, _, yp_n = forward_one(win_n, W1, b1, w2, b2)
    yp_o = denormalize(yp_n, xmin, xmax)
    print(f"  окно (исх.)    : {np.round(window_raw, 4)}")
    print(f"  окно (норм.)   : {np.round(win_n, 4)}")
    print(f"  прогноз (норм.): {yp_n:+.4f}")
    print(f"  прогноз (исх.) : {yp_o:+.4f}")
    if k_target is not None:
        true_o = float(x_series[k_target])
        print(f"  истинное x({k_target}) : {true_o:+.4f}")
        print(f"  абс. ошибка         : {abs(yp_o - true_o):.4f}")



if __name__ == "__main__":


    x = generate_series(A, B, C, D)
    k = np.arange(K_TOTAL)

    plt.figure(figsize=(10, 3.5))
    plt.plot(k[:K_TRAIN_LAST + 1], x[:K_TRAIN_LAST + 1], label="train (k=0..139)")
    plt.plot(k[K_TRAIN_LAST + 1:], x[K_TRAIN_LAST + 1:], label="test (k=140..199)")
    plt.axvline(K_TRAIN_LAST + 0.5, color="k", ls="--", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x(k)"); plt.legend(); plt.grid(alpha=0.3)
    plt.title("Ряд варианта 15: x(k)=1.6·cos(0.26k)+0.8·sin(0.63k)")
    plt.tight_layout(); plt.savefig("results/01_series.png", dpi=130); plt.close()


    xmin = float(x[:K_TRAIN_LAST + 1].min())
    xmax = float(x[:K_TRAIN_LAST + 1].max())
    x_norm = normalize(x, xmin, xmax)
    range_series = float(x.max() - x.min())

    print("=" * 70)
    print("НОРМАЛИЗАЦИЯ (по обучающему участку):")
    print(f"  xmin(train)  = {xmin:.4f}, xmax(train)  = {xmax:.4f}")
    print(f"  тест (исх.)  : min={x[K_TRAIN_LAST + 1:].min():.4f}, "
          f"max={x[K_TRAIN_LAST + 1:].max():.4f}")
    print(f"  тест (норм.) : min={x_norm[K_TRAIN_LAST + 1:].min():.4f}, "
          f"max={x_norm[K_TRAIN_LAST + 1:].max():.4f}")
    print(f"  обратный пересчёт: x = x_norm·(xmax−xmin) + xmin")
    print(f"  размах всего ряда: {range_series:.4f}")


    p = P_VAR
    X_all, y_all = make_windows(x_norm, p)
    X_train, y_train, X_test, y_test = split_train_test(X_all, y_all, p)
    print("=" * 70)
    print(f"СКОЛЬЗЯЩЕЕ ОКНО: p={p}")
    print(f"  всего примеров : {len(y_all)}")
    print(f"  train примеров : {len(y_train)}")
    print(f"  test  примеров : {len(y_test)}")


    print("=" * 70)
    print(f"БАЗОВЫЙ ЭКСПЕРИМЕНТ: p={P_VAR}, N={N_VAR}, lr={LR}, seed=0")
    model_base, hist_base, conv_base, eps_base = train_mlp(X_train, y_train, N_VAR, seed=0)
    W1b, b1b, w2b, b2b = model_base
    pred_tr = predict(X_train, W1b, b1b, w2b, b2b)
    pred_te = predict(X_test,  W1b, b1b, w2b, b2b)
    mtr = metrics_report(y_train, pred_tr, xmin, xmax)
    mte = metrics_report(y_test,  pred_te, xmin, xmax)
    print(f"  эпох          : {eps_base}")
    print(f"  сошлось (≤1e-4): {conv_base}")
    print(f"  Es_final       : {hist_base[-1]:.3e}")
    print(f"  TRAIN: MAE={mtr['MAE']:.4f}  RMSE={mtr['RMSE']:.4f}  sMAPE={mtr['sMAPE']:.2f}%")
    print(f"  TEST : MAE={mte['MAE']:.4f}  RMSE={mte['RMSE']:.4f}  sMAPE={mte['sMAPE']:.2f}%")
    print(f"  MAPE (справочно, нестабилен из-за нулей ряда) = {mte['MAPE']:.2f}%")


    def study_p(values, N_fixed, n_seeds=N_SEEDS):
        rows = []
        for p_val in values:
            Xa, ya = make_windows(x_norm, p_val)
            Xtr, ytr, Xte, yte = split_train_test(Xa, ya, p_val)
            eps_list, mae_list, fails = [], [], 0
            for s in range(n_seeds):
                m, _, cv, ee = train_mlp(Xtr, ytr, N_fixed, seed=s)
                pe = predict(Xte, *m)
                mae_list.append(mae(denormalize(yte, xmin, xmax),
                                    denormalize(pe, xmin, xmax)))
                eps_list.append(ee)
                if not cv:
                    fails += 1
            rows.append((p_val,
                         float(np.mean(eps_list)),
                         float(np.std(eps_list, ddof=1)),
                         float(np.mean(mae_list)),
                         float(np.std(mae_list, ddof=1)),
                         fails))
        return rows

    def study_N(values, p_fixed, n_seeds=N_SEEDS):
        Xa, ya = make_windows(x_norm, p_fixed)
        Xtr, ytr, Xte, yte = split_train_test(Xa, ya, p_fixed)
        rows = []
        for N_val in values:
            eps_list, mae_list, fails = [], [], 0
            for s in range(n_seeds):
                m, _, cv, ee = train_mlp(Xtr, ytr, N_val, seed=s)
                pe = predict(Xte, *m)
                mae_list.append(mae(denormalize(yte, xmin, xmax),
                                    denormalize(pe, xmin, xmax)))
                eps_list.append(ee)
                if not cv:
                    fails += 1
            rows.append((N_val,
                         float(np.mean(eps_list)),
                         float(np.std(eps_list, ddof=1)),
                         float(np.mean(mae_list)),
                         float(np.std(mae_list, ddof=1)),
                         fails))
        return rows

    print("=" * 70)
    print(f"ИССЛЕДОВАНИЕ p (N={N_VAR}, {N_SEEDS} seed-ов):")
    rows_p = study_p([2, 4, P_VAR, 2 * P_VAR], N_VAR)
    print(f"{'p':>3} | эпохи mean±std | test MAE mean±std | несошлось")
    for r in rows_p:
        print(f"{r[0]:>3} | {r[1]:7.0f} ± {r[2]:5.0f} | "
              f"{r[3]:.4f} ± {r[4]:.4f} | {r[5]}/{N_SEEDS}")

    print("=" * 70)
    print(f"ИССЛЕДОВАНИЕ N (p={P_VAR}, {N_SEEDS} seed-ов):")
    rows_N = study_N([1, 2, N_VAR, 2 * N_VAR], P_VAR)
    print(f"{'N':>3} | эпохи mean±std | test MAE mean±std | несошлось")
    for r in rows_N:
        print(f"{r[0]:>3} | {r[1]:7.0f} ± {r[2]:5.0f} | "
              f"{r[3]:.4f} ± {r[4]:.4f} | {r[5]}/{N_SEEDS}")


    candidates = []
    for val, ep_m, ep_s, mae_m, mae_s, fails in rows_p:
        candidates.append((val, N_VAR, mae_m, ep_m, fails))
    for val, ep_m, ep_s, mae_m, mae_s, fails in rows_N:
        if val == N_VAR:
            continue
        candidates.append((P_VAR, val, mae_m, ep_m, fails))

    best_p, best_N, best_mae, best_ep, best_fails = min(candidates, key=lambda t: t[2])
    print("=" * 70)
    print(f"ЛУЧШАЯ КОНФИГУРАЦИЯ по mean test MAE: p={best_p}, N={best_N}")
    print(f"  mean MAE={best_mae:.4f}, mean эпох={best_ep:.0f}, "
          f"несошлось {best_fails}/{N_SEEDS}")

    Xb_all, yb_all = make_windows(x_norm, best_p)
    Xb_tr, yb_tr, Xb_te, yb_te = split_train_test(Xb_all, yb_all, best_p)
    if best_p == P_VAR and best_N == N_VAR:
        print("  (лучшая совпадает с базовой — переиспользуем обученную модель)")
        model_best, hist_best, conv_best, eps_best = model_base, hist_base, conv_base, eps_base
    else:
        model_best, hist_best, conv_best, eps_best = train_mlp(Xb_tr, yb_tr, best_N, seed=0)
        print(f"  переобучено: эпох={eps_best}, Es_final={hist_best[-1]:.3e}, "
              f"сошлось={conv_best}")
    W1best, b1best, w2best, b2best = model_best


    N_STEPS = 30
    K_START = K_TRAIN_LAST + 1        # 140
    rec_norm = recursive_forecast(W1best, b1best, w2best, b2best,
                                  x_norm, best_p, K_START, N_STEPS)
    rec_orig = denormalize(rec_norm, xmin, xmax)
    true_orig = x[K_START:K_START + N_STEPS]
    abs_err = np.abs(rec_orig - true_orig)
    threshold = 0.05 * range_series
    step_exceed = (int(np.argmax(abs_err > threshold) + 1)
                   if np.any(abs_err > threshold) else None)

    print("=" * 70)
    print(f"РЕКУРСИВНЫЙ ПРОГНОЗ ({N_STEPS} шагов от k={K_START}):")
    print(f"  лучшая конфигурация: p={best_p}, N={best_N}")
    print(f"  порог 5% размаха ряда = {threshold:.4f}")
    print(f"  первый шаг, где |err| > порога: {step_exceed}")
    print(f"  MAE на 30 шагах = {mae(true_orig, rec_orig):.4f}")
    for s in (5, 10, 20, 30):
        print(f"  шаг {s:>2}: err = {abs_err[s - 1]:.4f}")


    y_pred_all = denormalize(np.concatenate([pred_tr, pred_te]), xmin, xmax)
    k_pred = np.arange(p, K_TOTAL)
    plt.figure(figsize=(10, 4))
    plt.plot(k, x, "k-", label="эталон")
    plt.plot(k_pred, y_pred_all, "r--", lw=1.2, label="одношаговый прогноз")
    plt.axvline(K_TRAIN_LAST + 0.5, color="gray", ls=":", label="граница train/test")
    plt.xlabel("k"); plt.ylabel("x(k)"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Ряд и одношаговый прогноз (p={P_VAR}, N={N_VAR})")
    plt.tight_layout(); plt.savefig("results/02_oneshot.png", dpi=130); plt.close()


    X2, y2 = make_windows(x_norm, 2)
    X2_tr, y2_tr, _, _ = split_train_test(X2, y2, 2)
    _, h_p2, _, _ = train_mlp(X2_tr, y2_tr, N_VAR, seed=0)

    X16, y16 = make_windows(x_norm, 16)
    X16_tr, y16_tr, _, _ = split_train_test(X16, y16, 16)
    _, h_p16, _, _ = train_mlp(X16_tr, y16_tr, N_VAR, seed=0)

    plt.figure(figsize=(8, 4))
    for h, lbl in [(hist_base, f"p={P_VAR} (base)"),
                   (h_p2,      "p=2"),
                   (h_p16,     "p=16")]:
        plt.semilogy(h, label=lbl)
    plt.xlabel("эпоха"); plt.ylabel("Es (log)")
    plt.legend(); plt.grid(alpha=0.3, which="both")
    plt.title("Сходимость Es")
    plt.tight_layout(); plt.savefig("results/03_convergence.png", dpi=130); plt.close()


    yt_o = denormalize(y_test, xmin, xmax)
    yp_o = denormalize(pred_te, xmin, xmax)
    plt.figure(figsize=(5, 5))
    plt.scatter(yt_o, yp_o, s=14, alpha=0.7)
    lo = min(yt_o.min(), yp_o.min()); hi = max(yt_o.max(), yp_o.max())
    plt.plot([lo, hi], [lo, hi], "k--", label="y = x")
    plt.xlabel("эталон"); plt.ylabel("прогноз"); plt.legend(); plt.grid(alpha=0.3)
    plt.title(f"Тест: прогноз vs эталон (p={P_VAR}, N={N_VAR})")
    plt.tight_layout(); plt.savefig("results/04_scatter.png", dpi=130); plt.close()


    steps = np.arange(1, N_STEPS + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(K_START + steps - 1, true_orig, "k-", label="эталон")
    ax[0].plot(K_START + steps - 1, rec_orig, "r--", label="рекурсивный")
    ax[0].set_xlabel("k"); ax[0].set_ylabel("x(k)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"Рекурсивный прогноз (p={best_p}, N={best_N})")
    ax[1].plot(steps, abs_err, "b-o", ms=3)
    ax[1].axhline(threshold, color="r", ls=":", label="5% размаха ряда")
    ax[1].set_xlabel("шаг"); ax[1].set_ylabel("|err|"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_title("Рост абсолютной ошибки")
    plt.tight_layout(); plt.savefig("results/05_recursive.png", dpi=130); plt.close()


    print("=" * 70)
    print(f"РЕЖИМ ФУНКЦИОНИРОВАНИЯ (база: p={P_VAR}, N={N_VAR}):")
    print("-- окна из обучающего участка --")
    for kk in (20, 60, 100):
        print(f"[k={kk}]")
        functioning(x[kk - p:kk], W1b, b1b, w2b, b2b, xmin, xmax, x, k_target=kk)
    print("-- окна из тестового участка --")
    for kk in (145, 170, 195):
        print(f"[k={kk}]")
        functioning(x[kk - p:kk], W1b, b1b, w2b, b2b, xmin, xmax, x, k_target=kk)
    print("-- пользовательские окна --")
    print("[пользовательское, без цели]")
    functioning([0.2, 0.3, -0.1, 0.4, 0.5, -0.2, 0.1, 0.3],
                W1b, b1b, w2b, b2b, xmin, xmax, x)
    print("[пользовательское с выбросом, без цели]")
    functioning([0.2, 0.3, -0.1, 0.4, 0.5, -0.2, 0.1, 5.0],
                W1b, b1b, w2b, b2b, xmin, xmax, x)

    print("=" * 70)
    print("Готово. Графики в ./results/")