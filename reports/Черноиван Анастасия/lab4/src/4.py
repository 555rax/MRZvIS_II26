import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A, B, C, D = 1.8, 0.45, 0.9, 1.0
P_VAR, N_VAR = 6, 5
K_TOTAL, K_SPLIT = 200, 140
EE = 1e-4
MAX_EPOCHS = 20000
SEEDS = [1, 2, 3, 4, 5]
HORIZON = 30
OUT = "out"

def generate_series(a=A, b=B, c=C, d=D, n=K_TOTAL):
    k = np.arange(n)
    return a * np.cos(b * k) + c * np.sin(d * k)

def fit_minmax(train):
    return float(np.min(train)), float(np.max(train))

def normalize(x, mn, mx):
    return (x - mn) / (mx - mn)

def denormalize(xn, mn, mx):
    return xn * (mx - mn) + mn

def make_windows(x, p, k_from, k_to):
    X, y = [], []
    for k in range(k_from, k_to):
        X.append([x[k - j] for j in range(1, p + 1)])
        y.append(x[k])
    return np.array(X), np.array(y)

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def init_weights(p, n_hidden, seed):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.uniform(-0.5, 0.5, (n_hidden, p)),
        "b1": rng.uniform(-0.5, 0.5, n_hidden),
        "v":  rng.uniform(-0.5, 0.5, n_hidden),
        "b2": float(rng.uniform(-0.5, 0.5)),
    }, rng

def forward(w, x):
    h = sigmoid(w["W1"] @ x + w["b1"])
    y = float(w["v"] @ h + w["b2"])
    return h, y

def backward_update(w, x, h, y, target, lr):
    delta = y - target
    delta_h = delta * w["v"] * h * (1.0 - h)
    w["v"] -= lr * delta * h
    w["b2"] -= lr * delta
    w["W1"] -= lr * np.outer(delta_h, x)
    w["b1"] -= lr * delta_h
    return 0.5 * delta * delta

def predict_all(w, X):
    return np.array([forward(w, x)[1] for x in X])

def train(Xtr, ytr, n_hidden, lr, seed, max_epochs=MAX_EPOCHS, ee=EE):
    w, rng = init_weights(Xtr.shape[1], n_hidden, seed)
    m = len(ytr)
    history = []
    for epoch in range(1, max_epochs + 1):
        total = 0.0
        for i in rng.permutation(m):
            h, y = forward(w, Xtr[i])
            total += backward_update(w, Xtr[i], h, y, ytr[i], lr)
        es = total / m
        history.append(es)
        if es <= ee:
            break
    return w, history, (history[-1] <= ee)

def mae(t, y):
    return float(np.mean(np.abs(t - y)))

def rmse(t, y):
    return float(np.sqrt(np.mean((t - y) ** 2)))

def mape(t, y):
    return float(np.mean(np.abs((t - y) / t)) * 100)

def smape(t, y):
    return float(np.mean(2 * np.abs(t - y) / (np.abs(t) + np.abs(y))) * 100)

def all_metrics(t, y):
    return {"MAE": mae(t, y), "RMSE": rmse(t, y), "sMAPE": smape(t, y)}

def recursive_forecast(w, xn, k_start, steps, p):
    window = [xn[k_start - j] for j in range(1, p + 1)]
    out = []
    for _ in range(steps):
        _, y = forward(w, np.array(window))
        out.append(y)
        window = [y] + window[:-1]
    return np.array(out)

def functioning_mode(w, mn, mx, p, window_raw=None, k=None, series=None, label=""):
    if window_raw is None:
        window_raw = [series[k - j] for j in range(1, p + 1)]
    true_val = series[k] if (k is not None and series is not None) else None
    window_raw = np.array(window_raw, dtype=float)
    xin = normalize(window_raw, mn, mx)
    _, yn = forward(w, xin)
    y = denormalize(yn, mn, mx)
    lines = [f"[{label}]" if label else ""]
    lines.append("  окно (x(k-1)..x(k-p)), исх. шкала : " + np.array2string(window_raw, precision=4))
    lines.append("  нормализованные входы            : " + np.array2string(xin, precision=4))
    lines.append(f"  прогноз (норм. шкала)            : {yn:.4f}")
    lines.append(f"  прогноз (исходная шкала)         : {y:.4f}")
    if true_val is not None:
        lines.append(f"  истинное x({k})                   : {true_val:.4f}")
        lines.append(f"  абсолютная ошибка                : {abs(true_val - y):.4f}")
    txt = "\n".join(l for l in lines if l != "")
    print(txt)
    return txt, y

def ask_windows(w, mn, mx, x, p=P_VAR):
    print(f"\nРежим функционирования. Введите k ({p}..{K_TOTAL - 1}) или {p} значений x(k-1)..x(k-{p}) через пробел; пустая строка - выход.")
    while True:
        try:
            s = input("> ").strip()
        except EOFError:
            break
        if not s:
            break
        try:
            vals = [float(v) for v in s.replace(",", " ").split()]
            if len(vals) == 1 and p <= vals[0] < K_TOTAL and vals[0] == int(vals[0]):
                functioning_mode(w, mn, mx, p, k=int(vals[0]), series=x)
            elif len(vals) == p:
                functioning_mode(w, mn, mx, p, window_raw=vals)
            else:
                print(f"Нужно одно k или {p} чисел")
        except ValueError:
            print("Не число, повторите ввод")

def run_config(x, xn, mn, mx, p, n_hidden, lr, seed, max_epochs=MAX_EPOCHS):
    Xtr, ytr = make_windows(xn, p, p, K_SPLIT)
    Xte, yte = make_windows(xn, p, K_SPLIT, K_TOTAL)
    w, hist, ok = train(Xtr, ytr, n_hidden, lr, seed, max_epochs)
    ptr = denormalize(predict_all(w, Xtr), mn, mx)
    pte = denormalize(predict_all(w, Xte), mn, mx)
    ttr = denormalize(ytr, mn, mx)
    tte = denormalize(yte, mn, mx)
    return {"w": w, "hist": hist, "ok": ok, "epochs": len(hist), "Es": hist[-1],
            "train": all_metrics(ttr, ptr), "test": all_metrics(tte, pte),
            "pred_train": ptr, "pred_test": pte, "true_train": ttr, "true_test": tte,
            "n_train": len(ytr), "n_test": len(yte), "p": p, "N": n_hidden, "seed": seed}

def series_study(x, xn, mn, mx, p_list, n_list, lr, label):
    rows = []
    for p, n in zip(p_list, n_list):
        runs = [run_config(x, xn, mn, mx, p, n, lr, s) for s in SEEDS]
        ep = np.array([r["epochs"] for r in runs], float)
        mt = np.array([r["test"]["MAE"] for r in runs])
        best = runs[int(np.argmin(mt))]
        rows.append({"p": p, "N": n, "ep_mean": ep.mean(), "ep_std": ep.std(),
                     "mae_mean": mt.mean(), "mae_std": mt.std(),
                     "fails": int(sum(not r["ok"] for r in runs)),
                     "Es_mean": float(np.mean([r["Es"] for r in runs])),
                     "mae_train_mean": float(np.mean([r["train"]["MAE"] for r in runs])),
                     "runs": runs, "best": best})
        print(f"  {label}: p={p:2d} N={n:2d} | эпохи {ep.mean():8.1f}±{ep.std():7.1f} | "
              f"test MAE {mt.mean():.5f}±{mt.std():.5f} | не достигли Ee: {rows[-1]['fails']}/5")
    return rows

def manual_step(V=8, lr=0.1):
    X = np.array([0.2, 0.5, 0.8]); d = 0.6
    W = np.array([[0.05 * V, -0.2], [0.3, 0.05 * V], [-0.1, 0.4]])
    b1 = np.array([0.1, -0.1])
    v = np.array([0.5, -0.2 * V]); b2 = 0.2
    s = X @ W + b1
    h = sigmoid(s)
    y = float(v @ h + b2)
    E = 0.5 * (y - d) ** 2
    delta = y - d
    dv = delta * h; db2 = delta
    dh = delta * v * h * (1 - h)
    dW = np.outer(X, dh); db1 = dh
    res = {"s": s, "h": h, "y": y, "E": E, "delta_out": delta, "dv": dv, "db2": db2,
           "delta_h": dh, "dW": dW, "db1": db1, "lr": lr,
           "v_new": v - lr * dv, "b2_new": b2 - lr * db2,
           "W_new": W - lr * dW, "b1_new": b1 - lr * db1}
    s2 = X @ res["W_new"] + res["b1_new"]
    y2 = float(res["v_new"] @ sigmoid(s2) + res["b2_new"])
    res["y_after"] = y2; res["E_after"] = 0.5 * (y2 - d) ** 2
    return res

def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    log = []

    def say(s=""):
        print(s); log.append(s)

    x = generate_series()
    mn, mx = fit_minmax(x[:K_SPLIT])
    xn = normalize(x, mn, mx)
    rng_all = float(x.max() - x.min())
    say(f"Ряд: x(k)={A}*cos({B}k)+{C}*sin({D}k); train min={mn:.4f}, max={mx:.4f}; размах ряда={rng_all:.4f}")
    say(f"Нормализованный тест: min={xn[K_SPLIT:].min():.4f}, max={xn[K_SPLIT:].max():.4f} "
        f"(вне [0;1]: {int(np.sum((xn[K_SPLIT:] < 0) | (xn[K_SPLIT:] > 1)))} точек)")
    say(f"Ряд проходит через ноль: {bool(np.any(x[:-1] * x[1:] < 0))} -> используем sMAPE")

    say("\n== Подбор скорости обучения (p=6, N=5, seed=1) ==")
    lr_rows = []
    for lr in (0.01, 0.05, 0.1, 0.3):
        r = run_config(x, xn, mn, mx, P_VAR, N_VAR, lr, 1, max_epochs=5000)
        lr_rows.append((lr, r["epochs"], r["Es"], r["test"]["MAE"], r["ok"]))
        say(f"  lr={lr:<5} эпох={r['epochs']:5d}  Es={r['Es']:.3e}  test MAE={r['test']['MAE']:.4f}  Ee достигнут: {r['ok']}")
    LR = 0.1
    say(f"Выбрано lr = {LR}")

    say("\n== Базовый эксперимент (p=6, N=5, seed=1) ==")
    base = run_config(x, xn, mn, mx, P_VAR, N_VAR, LR, 1)
    say(f"Примеров: train={base['n_train']}, test={base['n_test']}")
    say(f"Эпох: {base['epochs']}, Es={base['Es']:.3e}, Ee достигнут: {base['ok']}")
    for part in ("train", "test"):
        m = base[part]
        say(f"  {part:5s}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  sMAPE={m['sMAPE']:.3f} %")

    if "--func" in sys.argv:
        ask_windows(base["w"], mn, mx, x)
        return

    say("\n== Исследование окна (N=5) ==")
    p_list = [2, 4, P_VAR, 2 * P_VAR]
    win = series_study(x, xn, mn, mx, p_list, [N_VAR] * 4, LR, "окно")
    say("\n== Исследование скрытого слоя (p=6) ==")
    n_list = [1, 2, N_VAR, 2 * N_VAR]
    hid = series_study(x, xn, mn, mx, [P_VAR] * 4, n_list, LR, "скрытый")

    cands = [(r["mae_mean"], r) for r in win + hid]
    best_row = min(cands, key=lambda t: t[0])[1]
    best = best_row["best"]
    say(f"\nЛучшая конфигурация: p={best_row['p']}, N={best_row['N']} "
        f"(средний test MAE={best_row['mae_mean']:.5f}); для прогноза взят запуск seed={best['seed']}")

    pb = best["p"]
    rec_n = recursive_forecast(best["w"], xn, K_SPLIT, HORIZON, pb)
    rec = denormalize(rec_n, mn, mx)
    ref = x[K_SPLIT:K_SPLIT + HORIZON]
    err = np.abs(ref - rec)
    thr = 0.05 * rng_all
    over = np.where(err > thr)[0]
    first = int(over[0]) + 1 if len(over) else None
    one_step = best["pred_test"][:HORIZON]
    say(f"\n== Рекурсивный прогноз на {HORIZON} шагов (k=140..{139 + HORIZON}) ==")
    say(f"Порог 5% размаха = {thr:.4f}; первый шаг с превышением: {first}")
    say("Шаг: ошибка -> " + ", ".join(f"{i + 1}:{e:.4f}" for i, e in enumerate(err)))
    say(f"Средняя абс. ошибка: рекурсивный {err.mean():.4f} vs одношаговый {np.abs(ref - one_step).mean():.4f}")

    say("\n== Эксперимент с шумом (шум только в обучающем участке, sigma=0.2, обучение до 5000 эпох) ==")
    nrng = np.random.default_rng(123)
    x_noisy = x.copy()
    x_noisy[:K_SPLIT] += nrng.normal(0, 0.2, K_SPLIT)
    mn_n, mx_n = fit_minmax(x_noisy[:K_SPLIT])
    xn_noisy = normalize(x_noisy, mn_n, mx_n)
    noise_rows = []
    for n in (2, N_VAR, 20):
        r = run_config(x_noisy, xn_noisy, mn_n, mx_n, P_VAR, n, LR, 1, max_epochs=5000)
        Xte, _ = make_windows(xn_noisy, P_VAR, K_SPLIT, K_TOTAL)
        pte = denormalize(predict_all(r["w"], Xte), mn_n, mx_n)
        clean_mae = mae(x[K_SPLIT:], pte)
        noise_rows.append({"N": n, "epochs": r["epochs"], "Es": r["Es"], "train_MAE": r["train"]["MAE"],
                           "test_MAE_clean": clean_mae})
        say(f"  N={n:2d}: эпох={r['epochs']}, Es={r['Es']:.3e}, MAE(train, зашумл.)={r['train']['MAE']:.4f}, "
            f"MAE(test, чистый ряд)={clean_mae:.4f}")

    say("\n== Режим функционирования (базовая сеть p=6, N=5) ==")
    demo_txt = []
    w = base["w"]
    for k in (20, 60, 120):
        t, _ = functioning_mode(w, mn, mx, P_VAR, k=k, series=x, label=f"обучающий участок, k={k}")
        demo_txt.append(t)
    for k in (145, 170, 199):
        t, _ = functioning_mode(w, mn, mx, P_VAR, k=k, series=x, label=f"тестовый участок, k={k}")
        demo_txt.append(t)
    user_win = [x[130 - j] for j in range(1, P_VAR + 1)]
    t, _ = functioning_mode(w, mn, mx, P_VAR, window_raw=user_win, k=130, series=x, label="пользовательское окно = окно k=130 (контроль)")
    demo_txt.append(t)
    outl = list(user_win); outl[2] = 9.0
    t, _ = functioning_mode(w, mn, mx, P_VAR, window_raw=outl, label="пользовательское окно с выбросом (3-е значение = 9.0), без истинного")
    demo_txt.append(t)
    log.extend(demo_txt)

    mres = manual_step()
    say("\n== Ручной расчёт 1 шага (3-2-1, V=8, lr=0.1) ==")
    for key in ("s", "h", "y", "E", "delta_out", "dv", "db2", "delta_h", "dW", "db1",
                "v_new", "b2_new", "W_new", "b1_new", "y_after", "E_after"):
        say(f"  {key}: {np.array2string(np.array(mres[key]), precision=5)}")

    kk = np.arange(K_TOTAL)
    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(kk, x, color="#1f4e79"); ax.axvline(K_SPLIT - 0.5, color="r", ls="--", label="граница train/test")
    ax.set_xlabel("k"); ax.set_ylabel("x(k)"); ax.set_title("Ряд x(k) = 1.8cos(0.45k)+0.9sin(k)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_series.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(kk, x, color="gray", lw=1.5, label="эталон")
    ax.plot(np.arange(P_VAR, K_SPLIT), base["pred_train"], "g.-", ms=3, lw=0.8, label="прогноз (train)")
    ax.plot(np.arange(K_SPLIT, K_TOTAL), base["pred_test"], "r.-", ms=3, lw=0.8, label="прогноз (test)")
    ax.axvline(K_SPLIT - 0.5, color="k", ls="--"); ax.set_xlabel("k"); ax.set_ylabel("x(k)")
    ax.set_title("Одношаговый прогноз сети 6-5-1"); ax.legend(ncol=3); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_onestep.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.8))
    r_p2 = win[0]["runs"][0]; r_p12 = win[3]["runs"][0]
    for hist, lab in ((base["hist"], f"p=6, N=5 ({base['epochs']} эп.)"),
                      (r_p2["hist"], f"p=2, N=5 ({r_p2['epochs']} эп.)"),
                      (r_p12["hist"], f"p=12, N=5 ({r_p12['epochs']} эп.)")):
        ax.loglog(np.arange(1, len(hist) + 1), hist, label=lab)
    ax.axhline(EE, color="k", ls=":", label="Ee = 1e-4")
    ax.set_xlabel("эпоха (лог.)"); ax.set_ylabel("Es (лог.)"); ax.set_title("Сходимость обучения (seed=1)")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_convergence.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.scatter(base["true_test"], base["pred_test"], s=14, color="#c0392b")
    lo, hi = min(base["true_test"].min(), base["pred_test"].min()), max(base["true_test"].max(), base["pred_test"].max())
    ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
    ax.set_xlabel("эталон"); ax.set_ylabel("прогноз"); ax.set_title("Тест: эталон – прогноз"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_scatter.png", dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    steps = np.arange(1, HORIZON + 1)
    axs[0].plot(steps, ref, "k.-", label="эталон"); axs[0].plot(steps, rec, "r.-", label="рекурсивный прогноз")
    axs[0].set_xlabel("шаг"); axs[0].set_ylabel("x"); axs[0].set_title(f"Рекурсивный прогноз (p={pb}, N={best['N']})")
    axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].plot(steps, err, "b.-", label="|ошибка|"); axs[1].axhline(thr, color="r", ls="--", label="5% размаха")
    axs[1].set_xlabel("шаг"); axs[1].set_ylabel("абс. ошибка"); axs[1].set_title("Рост ошибки по шагам")
    axs[1].legend(); axs[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_recursive.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(kk[:K_SPLIT], x_noisy[:K_SPLIT], ".", color="gray", ms=3, label="зашумлённый train")
    ax.plot(kk, x, color="#1f4e79", lw=1, label="чистый ряд"); ax.axvline(K_SPLIT - 0.5, color="r", ls="--")
    ax.set_xlabel("k"); ax.set_title("Эксперимент с шумом (sigma = 0.2 на обучающем участке)"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_noise.png", dpi=150); plt.close(fig)

    def strip(rows):
        return [{k: v for k, v in r.items() if k not in ("runs", "best")} for r in rows]
    summary = {
        "mn": mn, "mx": mx, "range": rng_all, "lr": LR, "lr_scan": lr_rows,
        "base": {k: base[k] for k in ("epochs", "Es", "ok", "train", "test", "n_train", "n_test")},
        "window": strip(win), "hidden": strip(hid),
        "window_runs": [[{"seed": r["seed"], "epochs": r["epochs"], "ok": bool(r["ok"]), "mae": r["test"]["MAE"]} for r in row["runs"]] for row in win],
        "hidden_runs": [[{"seed": r["seed"], "epochs": r["epochs"], "ok": bool(r["ok"]), "mae": r["test"]["MAE"]} for r in row["runs"]] for row in hid],
        "best": {"p": best_row["p"], "N": best_row["N"], "seed": best["seed"]},
        "recursive": {"first_exceed": first, "thr": thr, "err": err.tolist(),
                      "mean_rec": float(err.mean()), "mean_one": float(np.abs(ref - one_step).mean())},
        "noise": noise_rows, "demo": demo_txt,
        "xn_test_min": float(xn[K_SPLIT:].min()), "xn_test_max": float(xn[K_SPLIT:].max()),
        "manual": {k: (np.array(v).tolist()) for k, v in mres.items()},
    }
    with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=float)
    with open(f"{OUT}/log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))

    ask_windows(w, mn, mx, x)

if __name__ == "__main__":
    main()