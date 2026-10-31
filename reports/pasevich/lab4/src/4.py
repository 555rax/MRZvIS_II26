import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool

A, B, C, D = 0.9, 0.33, 0.45, 0.75
P_VAR, N_VAR = 7, 4

N_TOTAL = 200
N_TRAIN = 140
EE = 1e-4
MAX_EPOCHS = 20000
ETA = 0.1
BASE_SEED = 42
SEEDS = [1, 2, 3, 4, 5]


def generate_series(n=N_TOTAL):
    k = np.arange(n)
    return A * np.cos(B * k) + C * np.sin(D * k)


def fit_minmax(train_part):
    return float(np.min(train_part)), float(np.max(train_part))


def normalize(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


def denormalize(xn, xmin, xmax):
    return xn * (xmax - xmin) + xmin


def make_windows(series_n, p, first_target, last_target):
    X, d = [], []
    for k in range(first_target, last_target + 1):
        X.append([series_n[k - j] for j in range(1, p + 1)])
        d.append(series_n[k])
    return np.array(X), np.array(d)


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def init_weights(p, n_hidden, seed):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.uniform(-0.5, 0.5, (n_hidden, p)),
        "b1": rng.uniform(-0.5, 0.5, n_hidden),
        "W2": rng.uniform(-0.5, 0.5, n_hidden),
        "b2": float(rng.uniform(-0.5, 0.5)),
    }


def forward(w, x):
    h = sigmoid(w["W1"] @ x + w["b1"])
    y = float(w["W2"] @ h + w["b2"])
    return h, y


def predict_batch(w, X):
    H = sigmoid(X @ w["W1"].T + w["b1"])
    return H @ w["W2"] + w["b2"]


def backward(w, x, h, y, d):
    delta_out = y - d
    g_W2 = delta_out * h
    g_b2 = delta_out
    delta_h = delta_out * w["W2"] * h * (1.0 - h)
    g_W1 = np.outer(delta_h, x)
    g_b1 = delta_h
    return g_W1, g_b1, g_W2, g_b2


def mse(w, X, d):
    return float(np.mean((predict_batch(w, X) - d) ** 2))


def train(Xtr, dtr, p, n_hidden, seed, eta=ETA, ee=EE, max_epochs=MAX_EPOCHS):
    w = init_weights(p, n_hidden, seed)
    rng = np.random.default_rng(seed + 1000)
    M = len(dtr)
    history = []
    epochs = 0
    reached = False
    for ep in range(1, max_epochs + 1):
        for i in rng.permutation(M):
            x = Xtr[i]
            h, y = forward(w, x)
            gW1, gb1, gW2, gb2 = backward(w, x, h, y, dtr[i])
            w["W1"] -= eta * gW1
            w["b1"] -= eta * gb1
            w["W2"] -= eta * gW2
            w["b2"] -= eta * gb2
        es = mse(w, Xtr, dtr)
        history.append(es)
        epochs = ep
        if es <= ee:
            reached = True
            break
    return w, history, epochs, reached


def mae(y, yh):
    return float(np.mean(np.abs(y - yh)))


def rmse(y, yh):
    return float(np.sqrt(np.mean((y - yh) ** 2)))


def mape(y, yh):
    return float(np.mean(np.abs((y - yh) / y)) * 100)


def smape(y, yh):
    denom = np.abs(y) + np.abs(yh)
    eps = 1e-12
    return float(np.mean(2 * np.abs(y - yh) / (denom + eps)) * 100)


def all_metrics(y, yh):
    return {"MAE": mae(y, yh), "RMSE": rmse(y, yh), "sMAPE": smape(y, yh)}


def prepare(series, p):
    xmin, xmax = fit_minmax(series[:N_TRAIN])
    sn = normalize(series, xmin, xmax)
    Xtr, dtr = make_windows(sn, p, p, N_TRAIN - 1)
    Xte, dte = make_windows(sn, p, N_TRAIN, N_TOTAL - 1)
    return sn, xmin, xmax, Xtr, dtr, Xte, dte


def run_config(args):
    p, n_hidden, seed, eta, noise_sigma = args
    series = generate_series()
    sn, xmin, xmax, Xtr, dtr, Xte, dte = prepare(series, p)

    if noise_sigma > 0:
        rng = np.random.default_rng(seed + 5000)
        sn_noisy = sn.copy()
        sn_noisy[:N_TRAIN] += rng.normal(0, noise_sigma, N_TRAIN)
        Xtr, dtr = make_windows(sn_noisy, p, p, N_TRAIN - 1)

    w, hist, ep, ok = train(Xtr, dtr, p, n_hidden, seed, eta=eta)

    Xtr_clean, dtr_clean = make_windows(sn, p, p, N_TRAIN - 1)
    ytr = denormalize(dtr_clean, xmin, xmax)
    ptr = denormalize(predict_batch(w, Xtr_clean), xmin, xmax)

    yte = denormalize(dte, xmin, xmax)
    pte = denormalize(predict_batch(w, Xte), xmin, xmax)

    return {
        "p": p, "N": n_hidden, "seed": seed, "eta": eta, "epochs": ep, "reached": ok,
        "Es": hist[-1], "train": all_metrics(ytr, ptr), "test": all_metrics(yte, pte),
        "history": hist,
        "w": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in w.items()},
    }


def summarize(runs):
    ep = np.array([r["epochs"] for r in runs], float)
    te = np.array([r["test"]["MAE"] for r in runs])
    return {
        "epochs_mean": float(ep.mean()), "epochs_std": float(ep.std(ddof=0)),
        "epochs_min": float(ep.min()), "epochs_max": float(ep.max()),
        "mae_mean": float(te.mean()), "mae_std": float(te.std(ddof=0)),
        "mae_min": float(te.min()), "mae_max": float(te.max()),
        "failed": int(sum(not r["reached"] for r in runs)),
    }


def recursive_forecast(w, series_n, p, start, steps):
    window = [series_n[start - j] for j in range(1, p + 1)]
    preds = []
    wn = {k: np.array(v) for k, v in w.items()}
    for _ in range(steps):
        _, y = forward(wn, np.array(window))
        preds.append(y)
        window = [y] + window[:-1]
    return np.array(preds)


def functioning_mode(w, p, xmin, xmax, series, window_raw=None, k=None):
    wn = {kk: np.array(v) for kk, v in w.items()}
    truth = None
    if k is not None:
        window_raw = [series[k - j] for j in range(1, p + 1)]
        truth = series[k]
    window_raw = np.array(window_raw, float)
    xin = normalize(window_raw, xmin, xmax)
    _, yn = forward(wn, xin)
    y = denormalize(yn, xmin, xmax)
    return {
        "window_raw": window_raw.tolist(),
        "inputs_norm": xin.tolist(),
        "pred_norm": float(yn),
        "pred": float(y),
        "truth": None if truth is None else float(truth),
        "abs_err": None if truth is None else float(abs(truth - y)),
        "k": k,
    }


def format_func(res):
    s = []
    if res["k"] is not None:
        s.append(f"  window from series for k = {res['k']}")
    s.append("  input (original scale, order x_{k-1}..x_{k-p}): " +
             ", ".join(f"{v:.4f}" for v in res["window_raw"]))
    s.append("  normalized inputs: " +
             ", ".join(f"{v:.4f}" for v in res["inputs_norm"]))
    s.append(f"  forecast (normalized scale): {res['pred_norm']:.4f}")
    s.append(f"  forecast (original scale):  {res['pred']:.4f}")
    if res["truth"] is not None:
        s.append(f"  true x(k) = {res['truth']:.4f},  |error| = {res['abs_err']:.4f}")
    return "\n".join(s)


def main():
    series = generate_series()
    k_all = np.arange(N_TOTAL)
    xmin, xmax = fit_minmax(series[:N_TRAIN])

    out = {
        "xmin": xmin, "xmax": xmax,
        "range_full": float(series.max() - series.min()),
        "full_min": float(series.min()), "full_max": float(series.max()),
    }

    sn_all = normalize(series, xmin, xmax)
    out["test_norm_min"] = float(sn_all[N_TRAIN:].min())
    out["test_norm_max"] = float(sn_all[N_TRAIN:].max())
    out["train_norm_min"] = float(sn_all[:N_TRAIN].min())
    out["train_norm_max"] = float(sn_all[:N_TRAIN].max())
    out["min_abs_series"] = float(np.min(np.abs(series)))

    plt.figure(figsize=(10, 4))
    plt.plot(k_all, series, "b-", lw=1.4, label="x(k)")
    plt.axvline(N_TRAIN - 0.5, color="r", ls="--", label="train/test boundary")
    plt.axvspan(0, N_TRAIN - 0.5, color="green", alpha=0.07)
    plt.axvspan(N_TRAIN - 0.5, N_TOTAL, color="orange", alpha=0.10)
    plt.xlabel("k")
    plt.ylabel("x(k)")
    plt.title("Series x_k = 0.9*cos(0.33k) + 0.45*sin(0.75k)  (variant 9)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig0_series.png", dpi=150)
    plt.close()

    eta_res = []
    with Pool() as pool:
        jobs = [(P_VAR, N_VAR, BASE_SEED, e, 0.0)
                for e in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3)]
        for r in pool.map(run_config, jobs):
            eta_res.append({"eta": r["eta"], "epochs": r["epochs"],
                            "reached": r["reached"], "Es": r["Es"],
                            "test_MAE": r["test"]["MAE"]})
    out["eta_sweep"] = eta_res
    print("eta-sweep:", eta_res, flush=True)

    base = run_config((P_VAR, N_VAR, BASE_SEED, ETA, 0.0))
    out["base"] = {kk: base[kk] for kk in
                   ("p", "N", "seed", "eta", "epochs", "reached", "Es", "train", "test")}

    sn, xmin_b, xmax_b, Xtr_b, dtr_b, Xte_b, dte_b = prepare(series, P_VAR)
    out["n_train_examples_p"] = int(len(dtr_b))
    out["n_test_examples_p"] = int(len(dte_b))
    print("base:", out["base"], flush=True)

    p_list = [2, 4, P_VAR, 2 * P_VAR]
    n_list = [1, 2, N_VAR, 2 * N_VAR]

    jobs_p = [(p, N_VAR, s, ETA, 0.0) for p in p_list for s in SEEDS]
    jobs_n = [(P_VAR, n, s, ETA, 0.0) for n in n_list for s in SEEDS]

    with Pool() as pool:
        res_all = pool.map(run_config, jobs_p + jobs_n)
    res_p = res_all[:len(jobs_p)]
    res_n = res_all[len(jobs_p):]

    out["window_study"], out["hidden_study"] = {}, {}
    for i, p in enumerate(p_list):
        rs = res_p[i * 5:(i + 1) * 5]
        out["window_study"][str(p)] = {
            "summary": summarize(rs),
            "runs": [{"seed": r["seed"], "epochs": r["epochs"],
                      "reached": r["reached"], "Es": r["Es"],
                      "train_MAE": r["train"]["MAE"],
                      "test_MAE": r["test"]["MAE"]} for r in rs],
            "train_mae_mean": float(np.mean([r["train"]["MAE"] for r in rs])),
        }
    for i, n in enumerate(n_list):
        rs = res_n[i * 5:(i + 1) * 5]
        out["hidden_study"][str(n)] = {
            "summary": summarize(rs),
            "runs": [{"seed": r["seed"], "epochs": r["epochs"],
                      "reached": r["reached"], "Es": r["Es"],
                      "train_MAE": r["train"]["MAE"],
                      "test_MAE": r["test"]["MAE"]} for r in rs],
            "train_mae_mean": float(np.mean([r["train"]["MAE"] for r in rs])),
        }

    print(json.dumps({
        "window": {k: v["summary"] for k, v in out["window_study"].items()},
        "hidden": {k: v["summary"] for k, v in out["hidden_study"].items()},
    }, indent=1), flush=True)

    cand = res_p + res_n
    mean_by_cfg = {}
    for r in cand:
        mean_by_cfg.setdefault((r["p"], r["N"]), []).append(r)
    best_cfg = min(mean_by_cfg,
                   key=lambda c: np.mean([r["test"]["MAE"] for r in mean_by_cfg[c]]))
    best_run = min(mean_by_cfg[best_cfg], key=lambda r: r["test"]["MAE"])
    out["best_cfg"] = {
        "p": best_cfg[0], "N": best_cfg[1], "seed": best_run["seed"],
        "epochs": best_run["epochs"], "reached": best_run["reached"],
        "test": best_run["test"], "train": best_run["train"],
        "Es": best_run["Es"],
    }

    pb, nb = best_cfg
    snb, xminb, xmaxb, Xtrb, dtrb, Xteb, dteb = prepare(series, pb)
    wb = best_run["w"]

    steps = 30
    rec_n = recursive_forecast(wb, snb, pb, N_TRAIN, steps)
    rec = denormalize(rec_n, xminb, xmaxb)
    ref = series[N_TRAIN:N_TRAIN + steps]
    err = np.abs(ref - rec)
    thr = 0.05 * (series.max() - series.min())
    over = np.where(err > thr)[0]

    out["recursive"] = {
        "p": pb, "N": nb, "threshold": float(thr),
        "first_exceed_step": int(over[0] + 1) if len(over) else None,
        "err": err.tolist(),
        "max_err": float(err.max()), "mean_err": float(err.mean()),
        "err_step1": float(err[0]), "err_step5": float(err[4]),
        "err_step10": float(err[9]), "err_step20": float(err[19]),
        "err_step30": float(err[29]),
        "mae_30": float(err.mean()),
    }
    one_step_n = predict_batch({kk: np.array(v) for kk, v in wb.items()},
                               Xteb[:steps])
    one_step = denormalize(one_step_n, xminb, xmaxb)
    out["recursive"]["onestep_mae_30"] = float(np.mean(np.abs(ref - one_step)))
    out["recursive"]["onestep_max_30"] = float(np.max(np.abs(ref - one_step)))

    sigma = 0.05
    with Pool() as pool:
        noisy = pool.map(run_config,
                         [(P_VAR, N_VAR, s, ETA, sigma) for s in SEEDS])
    out["noise"] = {
        "sigma_norm": sigma,
        "epochs": [r["epochs"] for r in noisy],
        "reached": [r["reached"] for r in noisy],
        "Es": [r["Es"] for r in noisy],
        "train_MAE_mean": float(np.mean([r["train"]["MAE"] for r in noisy])),
        "test_MAE_mean": float(np.mean([r["test"]["MAE"] for r in noisy])),
        "test_MAE_std": float(np.std([r["test"]["MAE"] for r in noisy])),
        "clean_test_MAE_mean": out["window_study"][str(P_VAR)]["summary"]["mae_mean"],
    }

    wbase = {kk: np.array(v) for kk, v in base["w"].items()}
    snp, xmin_p, xmax_p, Xtr7, dtr7, Xte7, dte7 = prepare(series, P_VAR)
    pr_tr = denormalize(predict_batch(wbase, Xtr7), xmin_p, xmax_p)
    pr_te = denormalize(predict_batch(wbase, Xte7), xmin_p, xmax_p)

    plt.figure(figsize=(11, 4.5))
    plt.plot(k_all, series, "b-", lw=1.4, label="reference x(k)")
    plt.plot(np.arange(P_VAR, N_TRAIN), pr_tr, "g.", ms=4,
             label="network forecast (train)")
    plt.plot(np.arange(N_TRAIN, N_TOTAL), pr_te, "r.", ms=4,
             label="network forecast (test)")
    plt.axvline(N_TRAIN - 0.5, color="k", ls="--", label="train/test boundary")
    plt.xlabel("k"); plt.ylabel("x(k)")
    plt.title("One-step forecast, p=7, N=4 (variant 9)")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("fig1_onestep.png", dpi=150); plt.close()

    h2 = [r for r in res_p if r["p"] == 2][0]["history"]
    h14 = [r for r in res_p if r["p"] == 2 * P_VAR][0]["history"]

    plt.figure(figsize=(8, 4.5))
    plt.semilogy(np.arange(1, len(base["history"]) + 1), base["history"],
                 label=f"base p=7, N=4 ({len(base['history'])} epochs)")
    plt.semilogy(np.arange(1, len(h2) + 1), h2, label=f"p=2 ({len(h2)} epochs)")
    plt.semilogy(np.arange(1, len(h14) + 1), h14, label=f"p=14 ({len(h14)} epochs)")
    plt.axhline(EE, color="k", ls=":", label="Ee = 1e-4")
    plt.xscale("log")
    plt.xlabel("epoch (log scale)")
    plt.ylabel("Es (normalized scale)")
    plt.title("Training convergence")
    plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    plt.savefig("fig2_convergence.png", dpi=150); plt.close()

    yte7 = series[N_TRAIN:]
    plt.figure(figsize=(5.2, 5))
    plt.scatter(yte7, pr_te, s=14, c="r")
    lo = min(yte7.min(), pr_te.min()) - 0.1
    hi = max(yte7.max(), pr_te.max()) + 0.1
    plt.plot([lo, hi], [lo, hi], "k--", label="y = x")
    plt.xlabel("reference"); plt.ylabel("forecast")
    plt.title("Test segment: reference vs forecast")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("fig3_scatter.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    steps_ax = np.arange(1, steps + 1)
    ax[0].plot(steps_ax, ref, "b-o", ms=3, label="reference")
    ax[0].plot(steps_ax, rec, "r-s", ms=3, label="recursive forecast")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("x")
    ax[0].set_title(f"Recursive forecast (p={pb}, N={nb})")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(steps_ax, err, "m-o", ms=3, label="|error|")
    ax[1].axhline(thr, color="k", ls="--",
                  label=f"5% of range = {thr:.3f}")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("absolute error")
    ax[1].set_title("Error growth by step")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig4_recursive.png", dpi=150); plt.close()

    demo = []
    lines = [
        "OPERATING MODE",
        "(base network p=7, N=4, seed=42, variant 9)",
        f"xmin = {xmin:.6f}, xmax = {xmax:.6f}",
        "",
    ]
    cases = [("train", 20), ("train", 60), ("train", 120),
             ("test", 145), ("test", 170), ("test", 195)]
    for tag, k in cases:
        r = functioning_mode(base["w"], P_VAR, xmin, xmax, series, k=k)
        demo.append(dict(tag=tag, **r))
        lines.append(f"[{tag}]")
        lines.append(format_func(r))
        lines.append("")

    custom1 = [series[150 - j] for j in range(1, P_VAR + 1)]
    r = functioning_mode(base["w"], P_VAR, xmin, xmax, series,
                         window_raw=custom1)
    r["truth"] = float(series[150])
    r["abs_err"] = float(abs(r["truth"] - r["pred"]))
    r["k"] = None
    demo.append(dict(tag="user (from series, k=150)", **r))
    lines.append("[user window = series values k=143..149]")
    lines.append(format_func(r)); lines.append("")

    custom2 = list(custom1)
    custom2[3] += 3.0
    r = functioning_mode(base["w"], P_VAR, xmin, xmax, series,
                         window_raw=custom2)
    r["truth"] = float(series[150])
    r["abs_err"] = float(abs(r["truth"] - r["pred"]))
    r["k"] = None
    demo.append(dict(tag="user with outlier (+3.0 in x_{k-4})", **r))
    lines.append("[user window with outlier: x_{k-4} increased by 3.0]")
    lines.append(format_func(r)); lines.append("")

    open("functioning_demo.txt", "w", encoding="utf-8").write("\n".join(lines))
    out["demo"] = demo

    out["base_history_len"] = len(base["history"])
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    with open("base_weights.json", "w", encoding="utf-8") as f:
        json.dump(base["w"], f)

    print("\n".join(lines))

    if "--interactive" in sys.argv:
        print("\nInteractive mode.")
        print("Enter 'k=<number>' or 7 values separated by spaces; q to quit")
        while True:
            s = input("> ").strip()
            if s.lower() == "q":
                break
            if s.startswith("k="):
                res = functioning_mode(base["w"], P_VAR, xmin, xmax, series,
                                       k=int(s[2:]))
            else:
                res = functioning_mode(base["w"], P_VAR, xmin, xmax, series,
                                       window_raw=[float(v) for v in s.split()])
            print(format_func(res))


if __name__ == "__main__":
    main()