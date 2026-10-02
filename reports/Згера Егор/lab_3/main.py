# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

c0, c1 = 1, 8

X = np.array([[c0, c0], [c0, c1], [c1, c0], [c1, c1]], dtype=float)
y_raw = np.array([[c0], [c1], [c1], [c0]], dtype=float)
y_bin = np.array([[0.], [1.], [1.], [0.]])

LR = 0.1
MAX_EPOCHS = 20000
EE_BCE = 0.35
SEEDS = [0, 1, 2, 3, 4]

CFG_NAMES = {"A": "Конфигурация А (Сигмоида)", "V": "Конфигурация В (ReLU)"}
CFG_COLORS = {"A": "tab:blue", "V": "tab:orange"}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def relu(x):
    return np.maximum(0.0, x)

def hidden_activation(z, config):
    return sigmoid(z) if config == "A" else relu(z)

def hidden_derivative(z, h, config):
    if config == "A":
        return h * (1.0 - h)
    return (z > 0).astype(float)

def init_weights(seed, config="A", n_in=2, n_hidden=2, n_out=1):
    rng = np.random.RandomState(seed)
    if config == "V":
        W1 = rng.uniform(-1.0, 1.0, (n_in, n_hidden))
        b1 = rng.uniform(-5.0, 5.0, (1, n_hidden))
        W2 = rng.uniform(-1.0, 1.0, (n_hidden, n_out))
    else:
        W1 = rng.uniform(-0.5, 0.5, (n_in, n_hidden))
        b1 = np.zeros((1, n_hidden))
        W2 = rng.uniform(-0.5, 0.5, (n_hidden, n_out))
    b2 = np.zeros((1, n_out))
    return W1, b1, W2, b2


def forward(Xb, W1, b1, W2, b2, config):
    z1 = Xb.dot(W1) + b1
    h_out = hidden_activation(z1, config)
    o_out = sigmoid(h_out.dot(W2) + b2)
    return z1, h_out, o_out

def bce_loss(o_out, y, eps=1e-9):
    o_clip = np.clip(o_out, eps, 1 - eps)
    return -np.mean(y * np.log(o_clip) + (1 - y) * np.log(1 - o_clip))

def train(seed, config, lr=LR, max_epochs=MAX_EPOCHS, Ee=EE_BCE):
    W1, b1, W2, b2 = init_weights(seed, config)
    errors = []
    converged_epoch = None
    n = X.shape[0]

    for epoch in range(max_epochs):
        z1, h_out, o_out = forward(X, W1, b1, W2, b2, config)

        Es = bce_loss(o_out, y_bin)
        errors.append(Es)

        if Es <= Ee:
            converged_epoch = epoch
            break

        d_output = (o_out - y_bin) / n
        d_hidden = d_output.dot(W2.T) * hidden_derivative(z1, h_out, config)

        W2 -= h_out.T.dot(d_output) * lr
        b2 -= np.sum(d_output, axis=0, keepdims=True) * lr
        W1 -= X.T.dot(d_hidden) * lr
        b1 -= np.sum(d_hidden, axis=0, keepdims=True) * lr

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, errors=errors,
                converged_epoch=converged_epoch, config=config)

def nearest_class(value, c0, c1):
    return c0 if abs(value - c0) <= abs(value - c1) else c1


def rescale_output(o_out, c0, c1):
    return c0 + o_out * (c1 - c0)

def degenerate_neurons(result):
    z1, _, _ = forward(X, result["W1"], result["b1"], result["W2"], result["b2"], result["config"])
    dead = int(np.sum(np.all(z1 <= 0, axis=0)))
    linear = int(np.sum(np.all(z1 > 0, axis=0)))
    return dead, linear

def evaluate_run(result):
    _, _, o_out = forward(X, result["W1"], result["b1"], result["W2"], result["b2"], result["config"])
    pred_scale = rescale_output(o_out.flatten(), c0, c1)
    true_scale = y_raw.flatten()
    pred_class = np.array([nearest_class(v, c0, c1) for v in pred_scale])
    accuracy = np.mean(pred_class == true_scale)
    mae = np.mean(np.abs(pred_scale - true_scale))
    Es_final = bce_loss(o_out, y_bin)
    return accuracy, mae, Es_final

def run_experiments():
    all_results = {"A": [], "V": []}
    for config in ["A", "V"]:
        print(f"\n{CFG_NAMES[config].upper()}: BCE, Ee = {EE_BCE}")
        for seed in SEEDS:
            res = train(seed, config)
            acc, mae, Es_final = evaluate_run(res)
            res.update(seed=seed, accuracy=acc, mae=mae, Es_final=Es_final)
            all_results[config].append(res)
            status = f"эпоха {res['converged_epoch']}" if res["converged_epoch"] is not None \
                else f"НЕ СОШЛАСЬ за {MAX_EPOCHS} эпох"
            extra = ""
            if config == "V":
                dead, lin = degenerate_neurons(res)
                extra = f", мёртвых={dead}, линейных={lin}"
            print(f"seed={seed}: {status}, Es_final={Es_final:.4f}, "
                  f"accuracy={acc:.2f}, MAE={mae:.2f}{extra}")
    return all_results

def comparison_table(all_results):
    header = f"{'Метрика':<34}{'А (Сигмоида)':>16}{'В (ReLU)':>16}"
    print("\nТАБЛИЦА СРАВНЕНИЯ (статистика по 5 запускам)")
    print(header)
    print("-" * len(header))

    def stats(results):
        conv = [r for r in results if r["converged_epoch"] is not None]
        ep = [r["converged_epoch"] for r in conv]
        return {
            "Сошлось запусков": f"{len(conv)}/{len(results)}",
            "Эпох, min": f"{min(ep)}" if ep else "—",
            "Эпох, max": f"{max(ep)}" if ep else "—",
            "Эпох, среднее": f"{np.mean(ep):.1f}" if ep else "—",
            "Эпох, std": f"{np.std(ep):.1f}" if ep else "—",
            "Итоговая Es (BCE), среднее": f"{np.mean([r['Es_final'] for r in results]):.4f}",
            "Accuracy, среднее": f"{np.mean([r['accuracy'] for r in results]):.2f}",
            "MAE в шкале [c0;c1], среднее": f"{np.mean([r['mae'] for r in results]):.3f}",
            "MAE в шкале [c0;c1], std": f"{np.std([r['mae'] for r in results]):.3f}",
        }

    sA, sV = stats(all_results["A"]), stats(all_results["V"])
    for key in sA:
        print(f"{key:<34}{sA[key]:>16}{sV[key]:>16}")


def plot_convergence(all_results, rep_seed=0):
    fig, ax = plt.subplots(figsize=(8, 5))
    for config in ["A", "V"]:
        rep = next(r for r in all_results[config] if r["seed"] == rep_seed)
        ax.plot(rep["errors"], color=CFG_COLORS[config], label=CFG_NAMES[config])
    ax.axhline(EE_BCE, color="gray", linestyle="--", alpha=0.7, label=f"Ee={EE_BCE}")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Es (BCE)")
    ax.set_title(f"Сходимость: Сигмоида vs ReLU (seed={rep_seed})")
    ax.legend()
    fig.tight_layout()
    plt.savefig("plot_1_convergence.png", dpi=150)
    plt.show()

def plot_epochs_bar(all_results):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SEEDS))
    width = 0.35

    for i, config in enumerate(["A", "V"]):
        epochs = [r["converged_epoch"] if r["converged_epoch"] is not None else MAX_EPOCHS
                  for r in all_results[config]]
        bars = ax.bar(x + (i - 0.5) * width, epochs, width,
                      label=CFG_NAMES[config], color=CFG_COLORS[config])
        for bar, r in zip(bars, all_results[config]):
            if r["converged_epoch"] is None:
                bar.set_hatch("//")
                bar.set_edgecolor("red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in SEEDS])
    ax.set_ylabel("Эпох до сходимости (MAX_EPOCHS, если не сошлось)")
    ax.set_title("Разброс числа эпох по 5 запускам (штриховка = не сошлось)")
    ax.legend()
    fig.tight_layout()
    plt.savefig("plot_2_epochs_bar.png", dpi=150)
    plt.show()

def plot_decision_surfaces(all_results, rep_seed=0):
    a_range = np.linspace(-10, 10, 200)
    b_range = np.linspace(-10, 10, 200)
    AA, BB = np.meshgrid(a_range, b_range)
    grid = np.column_stack([AA.ravel(), BB.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, config in zip(axes, ["A", "V"]):
        rep = next(r for r in all_results[config] if r["seed"] == rep_seed)
        _, _, o = forward(grid, rep["W1"], rep["b1"], rep["W2"], rep["b2"], config)
        ZZ = o.reshape(AA.shape)
        cs = ax.contourf(AA, BB, ZZ, levels=20, cmap="coolwarm", vmin=0, vmax=1)
        fig.colorbar(cs, ax=ax, label="ŷ (нормализованный выход, 0..1)")
        if ZZ.min() < 0.5 < ZZ.max():
            ax.contour(AA, BB, ZZ, levels=[0.5], colors="black", linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y_bin.flatten(), cmap="coolwarm",
                   edgecolors="black", s=120, linewidths=1.5)
        ax.set_title(CFG_NAMES[config])
        ax.set_xlabel("A")
        ax.set_ylabel("B")

    fig.suptitle(f"Разделяющая поверхность (A,B ∈ [-10;10], seed={rep_seed})")
    fig.tight_layout()
    plt.savefig("plot_3_decision_surfaces.png", dpi=150)
    plt.show()

def plot_mae_comparison(all_results):
    maes = [np.mean([r["mae"] for r in all_results[c]]) for c in ["A", "V"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([CFG_NAMES["A"], CFG_NAMES["V"]], maes,
                  color=[CFG_COLORS["A"], CFG_COLORS["V"]])
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")
    ax.set_ylabel(f"Средняя MAE в шкале [{c0}; {c1}] (по 5 запускам)")
    ax.set_title("Точность восстановления исходной шкалы")
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    plt.savefig("plot_4_mae_comparison.png", dpi=150)
    plt.show()

def predict(a, b, result):
    _, _, o_out = forward(np.array([[a, b]], dtype=float),
                          result["W1"], result["b1"], result["W2"], result["b2"], result["config"])
    y_hat_norm = o_out.item()
    y_hat_scaled = rescale_output(y_hat_norm, c0, c1)
    y_class = nearest_class(y_hat_scaled, c0, c1)
    return y_hat_norm, y_hat_scaled, y_class

def demo_predictions(result):
    print(f"\nРЕЖИМ ФУНКЦИОНИРОВАНИЯ СЕТИ ({CFG_NAMES[result['config']]}, seed={result['seed']})")

    print("\n--- Все 4 примера обучающей таблицы ---")
    for a, b in X:
        n_, s_, c_ = predict(a, b, result)
        print(f"Вход: ({a:.0f}, {b:.0f}) -> ŷ_norm={n_:.4f}, ŷ_scaled={s_:.2f}, класс≈{c_}")

    print("\n--- Дополнительные пары (вне обучающей выборки) ---")
    for a, b in [(3, 3), (2, 7), (5, 5), (-4, 6)]:
        n_, s_, c_ = predict(a, b, result)
        print(f"Вход: ({a}, {b}) -> ŷ_norm={n_:.4f}, ŷ_scaled={s_:.2f}, класс≈{c_}")

def interactive_mode(result):
    print("\n--- Интерактивный режим (введите 'q' для выхода) ---")
    while True:
        raw = input("Введите A,B через пробел или запятую в диапазоне [-10;10]: ").strip()
        if raw.lower() == "q":
            break
        try:
            a_str, b_str = raw.replace(",", " ").split()
            a, b = float(a_str), float(b_str)
        except ValueError:
            print("Не удалось разобрать ввод, попробуйте снова (пример: 3 7).")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Значения должны быть в диапазоне [-10;10].")
            continue
        n_, s_, c_ = predict(a, b, result)
        print(f"ŷ_norm={n_:.4f} | ŷ в шкале [{c0};{c1}]={s_:.2f} | ближе к классу {c_}")

if __name__ == "__main__":
    all_results = run_experiments()
    comparison_table(all_results)

    plot_convergence(all_results, rep_seed=0)
    plot_epochs_bar(all_results)
    plot_decision_surfaces(all_results, rep_seed=0)
    plot_mae_comparison(all_results)

    rep_V = next((r for r in all_results["V"] if r["converged_epoch"] is not None),
                 all_results["V"][0])
    demo_predictions(rep_V)

    # interactive_mode(rep_V)