import numpy as np
import matplotlib.pyplot as plt

A, B, C, D = 1.8, 0.45, 0.9, 1.0
P, N = 6, 5
K_TOTAL = 200
K_TRAIN = 140         
Lr = 0.1
Ee = 1e-4
MAX_EPOCHS = 20000
SEED = 42


def generate_series(a, b, c, d, n = K_TOTAL):
    k = np.arange(n)
    return a * np.cos(b * k) + c * np.sin(d * k)

series = generate_series(A, B, C, D)

plt.figure(figsize=(11, 4))
plt.plot(series, label="x(k)")
plt.axvline(K_TRAIN - 0.5, color="r", linestyle="--", label="граница train/test")
plt.xlabel("k"); plt.ylabel("x(k)"); plt.title("Ряд x(k)")
plt.legend(); plt.grid(alpha=0.3)
plt.show()


def fit_minmax(x_train):
    return x_train.min(), x_train.max()

def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)

def denormalize(xn, x_min, x_max):
    return xn * (x_max - x_min) + x_min

x_min, x_max = fit_minmax(series[:K_TRAIN])
series_n = normalize(series, x_min, x_max)
print(f"x_min = {x_min:.4f}, x_max = {x_max:.4f}")
print(f"train в [{series_n[:K_TRAIN].min():.3f}; {series_n[:K_TRAIN].max():.3f}]")
print(f"test в [{series_n[K_TRAIN:].min():.3f}; {series_n[K_TRAIN:].max():.3f}]")


def make_windows(x, p, t_start, t_end):
    X = np.array([x[t - p:t] for t in range(t_start, t_end)])
    y = np.array([x[t] for t in range(t_start, t_end)])
    return X, y

Xtr, ytr = make_windows(series_n, P, P, K_TRAIN)
Xte, yte = make_windows(series_n, P, K_TRAIN, K_TOTAL)
print("train:", Xtr.shape, ytr.shape, "-> примеров:", len(Xtr))
print("test:", Xte.shape, yte.shape, "-> примеров:", len(Xte))


def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))

def init_weights(p, n, rng, scale=0.3):
    return {"W1": rng.uniform(-scale, scale, (n, p)),
            "b1": rng.uniform(-scale, scale, n),
            "W2": rng.uniform(-scale, scale, n),
            "b2": float(rng.uniform(-scale, scale))}

def forward(w, x):
    h = sigmoid(x @ w["W1"].T + w["b1"])
    y = h @ w["W2"] + w["b2"]
    return h, y

def backward(w, x, h, y, d):
    delta_out = 2.0 * (y - d)                      
    dW2 = delta_out * h
    db2 = delta_out
    delta_h = delta_out * w["W2"] * h * (1.0 - h)  
    dW1 = np.outer(delta_h, x)
    db1 = delta_h
    return dW1, db1, dW2, db2

def mse(w, X, d):
    _, y = forward(w, X)
    return float(np.mean((y - d) ** 2))

def train(Xtr, ytr, p, n, seed, lr = Lr, ee = Ee, max_epochs = MAX_EPOCHS):
    rng = np.random.default_rng(seed)
    w = init_weights(p, n, rng)
    history = []
    for epoch in range(1, max_epochs + 1):
        for i in rng.permutation(len(Xtr)):        
            x, d = Xtr[i], ytr[i]
            h, y = forward(w, x)
            dW1, db1, dW2, db2 = backward(w, x, h, y, d)
            w["W1"] -= lr * dW1
            w["b1"] -= lr * db1
            w["W2"] -= lr * dW2
            w["b2"] -= lr * db2                    
        Es = mse(w, Xtr, ytr)
        history.append(Es)
        if Es <= ee:
            return w, epoch, history, True
    return w, max_epochs, history, False


def mae(y, yh):   return float(np.mean(np.abs(y - yh)))
def rmse(y, yh):  return float(np.sqrt(np.mean((y - yh) ** 2)))
def mape(y, yh):  return float(100 * np.mean(np.abs((y - yh) / y)))
def smape(y, yh): return float(100 * np.mean(2 * np.abs(y - yh) / (np.abs(y) + np.abs(yh))))

def predict_orig(w, Xn, x_min, x_max):
    _, yn = forward(w, Xn)
    return denormalize(yn, x_min, x_max)

def evaluate(w, Xn, y_true, x_min, x_max):
    yh = predict_orig(w, Xn, x_min, x_max)
    return {"MAE": mae(y_true, yh), "RMSE": rmse(y_true, yh),
            "sMAPE": smape(y_true, yh), "MAPE": mape(y_true, yh)}


print(" lr   | эпохи (ср ± σ) | test MAE | не сошлись")
for lr in [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]:
    eps, maes, fails = [], [], 0
    for s in [1, 2, 3, 4, 5]:
        w, ep, hist, ok = train(Xtr, ytr, P, N, s, lr = lr)
        eps.append(ep)
        maes.append(evaluate(w, Xte, series[K_TRAIN:], x_min, x_max)["MAE"])
        fails += (not ok)
    print(f"{lr:5.2f} | {np.mean(eps):6.1f} ± {np.std(eps):5.1f} | {np.mean(maes):.4f}   | {fails}")


w_base, ep_base, hist_base, ok_base = train(Xtr, ytr, P, N, SEED)
print(f"Эпох: {ep_base}, Es = {hist_base[-1]:.3e}, критерий достигнут: {ok_base}")
for name, X, y_true in [("train", Xtr, series[P:K_TRAIN]),
                        ("test ", Xte, series[K_TRAIN:])]:
    m = evaluate(w_base, X, y_true, x_min, x_max)
    print(f"{name}: MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}  "
          f"sMAPE={m['sMAPE']:.2f}%  (MAPE={m['MAPE']:.1f}% — раздувается из-за нулей)")


SEEDS = [1, 2, 3, 4, 5]

def run_config(p, n, seeds=SEEDS):
    Xtr_, ytr_ = make_windows(series_n, p, p, K_TRAIN)
    Xte_, _ = make_windows(series_n, p, K_TRAIN, K_TOTAL)
    runs = []
    for s in seeds:
        w, ep, hist, ok = train(Xtr_, ytr_, p, n, s)
        m = evaluate(w, Xte_, series[K_TRAIN:], x_min, x_max)
        runs.append({"p": p, "n": n, "seed": s, "w": w, "epochs": ep,
                     "hist": hist, "ok": ok, "test_mae": m["MAE"]})
    return runs

def print_table(title, name, table):
    print(title)
    print(f"{name:>4} | эпохи: ср. ± σ  | test MAE: ср. ± σ | не достигли Es≤Ee (из 5)")
    for key, runs in table.items():
        ep = np.array([r["epochs"] for r in runs])
        ma = np.array([r["test_mae"] for r in runs])
        fails = sum(not r["ok"] for r in runs)
        print(f"{key:>4} | {ep.mean():8.1f} ± {ep.std():6.1f} | {ma.mean():.4f} ± {ma.std():.4f}  | {fails}")
    print()

window_runs = {p: run_config(p, N) for p in [2, 4, P, 2 * P]}
neuron_runs = {n: (window_runs[P] if n == N else run_config(P, n))
               for n in [1, 2, N, 2 * N]}

print_table(f"Исследование окна (N = {N})", "p", window_runs)
print_table(f"Исследование скрытого слоя (p = {P})", "N", neuron_runs)

configs = {}
for p_, runs in window_runs.items():
    configs[(p_, N)] = runs
for n_, runs in neuron_runs.items():
    configs[(P, n_)] = runs
best_key = min(configs, key=lambda k: np.mean([r["test_mae"] for r in configs[k]]))
best_run = min(configs[best_key], key=lambda r: r["test_mae"])
print(f"Лучшая конфигурация по среднему test MAE: p = {best_key[0]}, N = {best_key[1]}")
print(f"Берём её лучший запуск: seed = {best_run['seed']}, test MAE = {best_run['test_mae']:.4f}")


def recursive_forecast(w, series_n, p, start, steps):
    window = list(series_n[start - p:start])
    out = []
    for _ in range(steps):
        _, y = forward(w, np.array(window))
        out.append(float(y))
        window = window[1:] + [float(y)]           
    return np.array(out)

STEPS = 30
p_best = best_run["p"]
rec_pred = denormalize(recursive_forecast(best_run["w"], series_n, p_best, K_TRAIN, STEPS), x_min, x_max)
rec_true = series[K_TRAIN:K_TRAIN + STEPS]
rec_err = np.abs(rec_true - rec_pred)
threshold = 0.05 * (series.max() - series.min())
bad = np.where(rec_err > threshold)[0]
print(f"Порог 5% размаха: {threshold:.4f}")
if len(bad):
    print(f"Ошибка впервые превышает порог на шаге {bad[0] + 1} (k = {K_TRAIN + bad[0]})")
else:
    print("За 30 шагов ошибка порог не превысила")


Xtr, ytr = make_windows(series_n, P, P, K_TRAIN)
Xte, yte = make_windows(series_n, P, K_TRAIN, K_TOTAL)
pred_tr = predict_orig(w_base, Xtr, x_min, x_max)
pred_te = predict_orig(w_base, Xte, x_min, x_max)

plt.figure(figsize=(12, 4))
plt.plot(series, "k", label="эталон")
plt.plot(range(P, K_TRAIN), pred_tr, "b--", label="прогноз (train)")
plt.plot(range(K_TRAIN, K_TOTAL), pred_te, "r--", label="прогноз (test)")
plt.axvline(K_TRAIN - 0.5, color="gray", linestyle=":")
plt.xlabel("k"); plt.ylabel("x(k)"); plt.title("Одношаговый прогноз, базовая сеть")
plt.legend(); plt.grid(alpha=0.3); plt.show()

plt.figure(figsize=(8, 5))
plt.semilogy(hist_base, label=f"базовая: p={P}, N={N}")
for p_ in (2, 2 * P):
    plt.semilogy(window_runs[p_][0]["hist"], label=f"p={p_}, N={N} (seed={SEEDS[0]})")
plt.axhline(Ee, color="r", linestyle=":", label="Ee = 1e-4")
plt.xlabel("эпоха"); plt.ylabel("Es (лог. шкала)"); plt.title("Сходимость")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.show()

plt.figure(figsize=(5.5, 5.5))
plt.scatter(series[K_TRAIN:], pred_te, s=18)
lims = [series.min() - 0.2, series.max() + 0.2]
plt.plot(lims, lims, "r--", label="y = x")
plt.xlabel("эталон"); plt.ylabel("прогноз"); plt.title("Тест: эталон vs прогноз")
plt.legend(); plt.grid(alpha=0.3); plt.show()

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
steps_axis = np.arange(1, STEPS + 1)
ax[0].plot(steps_axis, rec_true, "k", label="эталон")
ax[0].plot(steps_axis, rec_pred, "r--", label="рекурсивный прогноз")
ax[0].set_xlabel("шаг"); ax[0].set_ylabel("x"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(steps_axis, rec_err, "o-")
ax[1].axhline(threshold, color="r", linestyle="--", label="5% размаха")
ax[1].set_xlabel("шаг"); ax[1].set_ylabel("|ошибка|"); ax[1].legend(); ax[1].grid(alpha=0.3)
plt.show()


def run_mode(w, p, series, x_min, x_max, k=None, window=None, title=""):
    if window is None:
        window = series[k - p:k]
    window = np.asarray(window, dtype=float)
    assert len(window) == p, f"нужно ровно {p} значений"
    xn = normalize(window, x_min, x_max)
    _, yn = forward(w, xn)
    y = denormalize(yn, x_min, x_max)
    print(f"--- {title}")
    print("окно (исходная шкала):", np.round(window, 4))
    print("нормализованные входы:", np.round(xn, 4))
    print(f"прогноз (норм.):   {float(yn):.4f}")
    print(f"прогноз (исходн.): {float(y):.4f}")
    if k is not None:
        print(f"истинное x({k}) = {series[k]:.4f}, абс. ошибка = {abs(series[k] - y):.4f}")

for k in [20, 70, 120]:
    run_mode(w_base, P, series, x_min, x_max, k=k, title=f"train, k={k}")
for k in [145, 170, 195]:
    run_mode(w_base, P, series, x_min, x_max, k=k, title=f"test, k={k}")

outlier = series[94:100].copy()
outlier[3] = 8.0                                   
run_mode(w_base, P, series, x_min, x_max, k=100, window=outlier, title="окно k=100 с выбросом 8.0")
run_mode(w_base, P, series, x_min, x_max, window=[1.0, 1.5, 2.0, 1.2, 0.3, -0.5], title="пользовательское окно")