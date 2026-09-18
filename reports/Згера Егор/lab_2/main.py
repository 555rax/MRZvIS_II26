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
EE_MSE = 0.01
EE_BCE = 0.05
SEEDS = [0, 1, 2, 3, 4]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative_from_output(o):
    return o * (1.0 - o)


def init_weights(seed, n_in=2, n_hidden=2, n_out=1):
    rng = np.random.RandomState(seed)
    W1 = rng.uniform(-0.5, 0.5, (n_in, n_hidden))
    b1 = np.zeros((1, n_hidden))
    W2 = rng.uniform(-0.5, 0.5, (n_hidden, n_out))
    b2 = np.zeros((1, n_out))
    return W1, b1, W2, b2


def forward(Xb, W1, b1, W2, b2):
    h_out = sigmoid(Xb.dot(W1) + b1)
    o_out = sigmoid(h_out.dot(W2) + b2)
    return h_out, o_out


def train_mse(seed, lr=LR, max_epochs=MAX_EPOCHS, Ee=EE_MSE):
    W1, b1, W2, b2 = init_weights(seed)
    errors = []
    converged_epoch = None

    for epoch in range(max_epochs):
        h_out, o_out = forward(X, W1, b1, W2, b2)

        error = y_raw - o_out
        Es = np.mean(error ** 2)
        errors.append(Es)

        if Es <= Ee:
            converged_epoch = epoch
            break

        d_output = error * sigmoid_derivative_from_output(o_out)
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative_from_output(h_out)

        W2 += h_out.T.dot(d_output) * lr
        b2 += np.sum(d_output, axis=0, keepdims=True) * lr
        W1 += X.T.dot(d_hidden) * lr
        b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, errors=errors, converged_epoch=converged_epoch)


def train_bce(seed, lr=LR, max_epochs=MAX_EPOCHS, Ee=EE_BCE, eps=1e-9):
    W1, b1, W2, b2 = init_weights(seed)
    errors = []
    converged_epoch = None
    n = X.shape[0]

    for epoch in range(max_epochs):
        h_out, o_out = forward(X, W1, b1, W2, b2)

        o_clip = np.clip(o_out, eps, 1 - eps)
        Es = -np.mean(y_bin * np.log(o_clip) + (1 - y_bin) * np.log(1 - o_clip))
        errors.append(Es)

        if Es <= Ee:
            converged_epoch = epoch
            break

        # dL/dz для BCE+сигмоида на выходе упрощается до (o_out - target)
        d_output = (o_out - y_bin) / n
        d_hidden = d_output.dot(W2.T) * sigmoid_derivative_from_output(h_out)

        W2 -= h_out.T.dot(d_output) * lr
        b2 -= np.sum(d_output, axis=0, keepdims=True) * lr
        W1 -= X.T.dot(d_hidden) * lr
        b1 -= np.sum(d_hidden, axis=0, keepdims=True) * lr

    return dict(W1=W1, b1=b1, W2=W2, b2=b2, errors=errors, converged_epoch=converged_epoch)


def nearest_class(value, c0, c1):
    return c0 if abs(value - c0) <= abs(value - c1) else c1


def rescale_bce_output(o_out, c0, c1):
    return c0 + o_out * (c1 - c0)


def evaluate_run(result, config):
    h_out, o_out = forward(X, result["W1"], result["b1"], result["W2"], result["b2"])

    if config == "A":
        pred_scale = o_out.flatten()
    else:
        pred_scale = rescale_bce_output(o_out.flatten(), c0, c1)

    true_scale = y_raw.flatten()
    pred_class = np.array([nearest_class(v, c0, c1) for v in pred_scale])
    accuracy = np.mean(pred_class == true_scale)
    mae = np.mean(np.abs(pred_scale - true_scale))
    return accuracy, mae, pred_scale


def run_experiments():
    results_A, results_B = [], []

    print(f"КОНФИГУРАЦИЯ А (MSE, без нормализации, Ee = {EE_MSE})")
    for seed in SEEDS:
        res = train_mse(seed)
        acc, mae, _ = evaluate_run(res, "A")
        res.update(seed=seed, accuracy=acc, mae=mae)
        results_A.append(res)
        status = f"эпоха {res['converged_epoch']}" if res["converged_epoch"] is not None \
            else f"НЕ СОШЛАСЬ за {MAX_EPOCHS} эпох"
        print(f"seed={seed}: {status}, Es_final={res['errors'][-1]:.4f}, accuracy={acc:.2f}, MAE={mae:.2f}")

    print(f"\nКОНФИГУРАЦИЯ Б (BCE, нормализованные метки 0/1, Ee = {EE_BCE})")
    for seed in SEEDS:
        res = train_bce(seed)
        acc, mae, _ = evaluate_run(res, "B")
        res.update(seed=seed, accuracy=acc, mae=mae)
        results_B.append(res)
        status = f"эпоха {res['converged_epoch']}" if res["converged_epoch"] is not None \
            else f"НЕ СОШЛАСЬ за {MAX_EPOCHS} эпох"
        print(f"seed={seed}: {status}, Es_final={res['errors'][-1]:.4f}, accuracy={acc:.2f}, MAE={mae:.2f}")

    return results_A, results_B


def convergence_summary(results, label):
    converged = [r for r in results if r["converged_epoch"] is not None]
    epochs = [r["converged_epoch"] for r in converged]
    print(f"\n{label}: сошлось {len(converged)} из {len(results)}")
    if epochs:
        print(f"Эпох до сходимости: min={min(epochs)}, max={max(epochs)}, "
              f"среднее={np.mean(epochs):.1f}, разброс={max(epochs) - min(epochs)}")
    else:
        print("Ни один запуск не достиг критерия остановки.")


def plot_convergence(results_A, results_B, rep_seed=0):
    rep_A = next(r for r in results_A if r["seed"] == rep_seed)
    rep_B = next(r for r in results_B if r["seed"] == rep_seed)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rep_A["errors"], color="tab:blue", label="Конфигурация А (MSE)")
    ax1.axhline(EE_MSE, color="tab:blue", linestyle="--", alpha=0.5, label=f"Ee(MSE)={EE_MSE}")
    ax1.set_xlabel("Эпоха")
    ax1.set_ylabel("Es (MSE)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(rep_B["errors"], color="tab:orange", label="Конфигурация Б (BCE)")
    ax2.axhline(EE_BCE, color="tab:orange", linestyle="--", alpha=0.5, label=f"Ee(BCE)={EE_BCE}")
    ax2.set_ylabel("Es (BCE)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.title("Сходимость: Конфигурация А (MSE) и Конфигурация Б (BCE)")
    fig.tight_layout()
    plt.savefig("plot_1_convergence.png", dpi=150)
    plt.show()


def plot_epochs_bar(results_A, results_B):
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SEEDS))
    width = 0.35

    epochs_A = [r["converged_epoch"] if r["converged_epoch"] is not None else MAX_EPOCHS for r in results_A]
    epochs_B = [r["converged_epoch"] if r["converged_epoch"] is not None else MAX_EPOCHS for r in results_B]
    conv_A = [r["converged_epoch"] is not None for r in results_A]
    conv_B = [r["converged_epoch"] is not None for r in results_B]

    bars_A = ax.bar(x - width / 2, epochs_A, width, label="Конфигурация А (MSE)", color="tab:blue")
    bars_B = ax.bar(x + width / 2, epochs_B, width, label="Конфигурация Б (BCE)", color="tab:orange")

    for bar, ok in zip(bars_A, conv_A):
        if not ok:
            bar.set_hatch("//")
            bar.set_edgecolor("red")
    for bar, ok in zip(bars_B, conv_B):
        if not ok:
            bar.set_hatch("//")
            bar.set_edgecolor("red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in SEEDS])
    ax.set_ylabel("Эпох до сходимости (или MAX_EPOCHS, если не сошлось)")
    ax.set_title("Устойчивость сходимости по 5 запускам (штриховка = не сошлось)")
    ax.legend()
    fig.tight_layout()
    plt.savefig("plot_2_epochs_bar.png", dpi=150)
    plt.show()


def plot_decision_surfaces(results_A, results_B, rep_seed=0):
    rep_A = next(r for r in results_A if r["seed"] == rep_seed)
    rep_B = next(r for r in results_B if r["seed"] == rep_seed)

    a_range = np.linspace(-10, 10, 200)
    b_range = np.linspace(-10, 10, 200)
    AA, BB = np.meshgrid(a_range, b_range)
    grid = np.column_stack([AA.ravel(), BB.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    _, o_A = forward(grid, rep_A["W1"], rep_A["b1"], rep_A["W2"], rep_A["b2"])
    ZZ_A = o_A.reshape(AA.shape)
    cs0 = axes[0].contourf(AA, BB, ZZ_A, levels=20, cmap="coolwarm")
    fig.colorbar(cs0, ax=axes[0], label="Выход сети (0;1), цели не нормализовались")
    axes[0].scatter(X[:, 0], X[:, 1], c=y_raw.flatten(), cmap="coolwarm", edgecolors="black", s=120, linewidths=1.5)
    axes[0].set_title("Конфигурация А (MSE, без нормализации)")
    axes[0].set_xlabel("A")
    axes[0].set_ylabel("B")

    # ŷ нормализованный: порог 0.5 = вероятность класса c1
    _, o_B = forward(grid, rep_B["W1"], rep_B["b1"], rep_B["W2"], rep_B["b2"])
    ZZ_B = o_B.reshape(AA.shape)
    cs1 = axes[1].contourf(AA, BB, ZZ_B, levels=20, cmap="coolwarm")
    fig.colorbar(cs1, ax=axes[1], label="ŷ (нормализованный выход, 0..1)")
    axes[1].contour(AA, BB, ZZ_B, levels=[0.5], colors="black", linewidths=2)
    axes[1].scatter(X[:, 0], X[:, 1], c=y_bin.flatten(), cmap="coolwarm", edgecolors="black", s=120, linewidths=1.5)
    axes[1].set_title("Конфигурация Б (BCE, нормализованные метки)")
    axes[1].set_xlabel("A")
    axes[1].set_ylabel("B")

    fig.suptitle("Разделяющая поверхность скрытого слоя (A,B ∈ [-10;10])")
    fig.tight_layout()
    plt.savefig("plot_3_decision_surfaces.png", dpi=150)
    plt.show()


def plot_mae_comparison(results_A, results_B):
    mae_A = np.mean([r["mae"] for r in results_A])
    mae_B = np.mean([r["mae"] for r in results_B])

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["Конфигурация А (MSE)", "Конфигурация Б (BCE)"], [mae_A, mae_B], color=["tab:blue", "tab:orange"])
    for bar, val in zip(bars, [mae_A, mae_B]):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.2f}", ha="center", va="bottom")
    ax.set_ylabel(f"Средняя MAE в шкале [{c0}; {c1}] (по 5 запускам)")
    ax.set_title("Точность восстановления исходной шкалы")
    fig.tight_layout()
    plt.savefig("plot_4_mae_comparison.png", dpi=150)
    plt.show()


def predict_B(a, b, result):
    h_out, o_out = forward(np.array([[a, b]], dtype=float), result["W1"], result["b1"], result["W2"], result["b2"])
    y_hat_norm = o_out.item()
    y_hat_scaled = rescale_bce_output(y_hat_norm, c0, c1)
    y_class = nearest_class(y_hat_scaled, c0, c1)
    return y_hat_norm, y_hat_scaled, y_class


def demo_predictions(result):
    print("\nРЕЖИМ ФУНКЦИОНИРОВАНИЯ СЕТИ (конфигурация Б)")

    print("\n--- Все 4 примера обучающей таблицы ---")
    for a, b in X:
        y_hat_norm, y_hat_scaled, y_class = predict_B(a, b, result)
        print(f"Вход: ({a:.0f}, {b:.0f}) -> ŷ_norm={y_hat_norm:.4f}, ŷ_scaled={y_hat_scaled:.2f}, класс≈{y_class}")

    print("\n--- Дополнительные пары (вне обучающей выборки) ---")
    extra_pairs = [(3, 3), (2, 7), (5, 5), (-4, 6)]
    for a, b in extra_pairs:
        y_hat_norm, y_hat_scaled, y_class = predict_B(a, b, result)
        print(f"Вход: ({a}, {b}) -> ŷ_norm={y_hat_norm:.4f}, ŷ_scaled={y_hat_scaled:.2f}, класс≈{y_class}")


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
        y_hat_norm, y_hat_scaled, y_class = predict_B(a, b, result)
        print(f"ŷ_norm={y_hat_norm:.4f} | ŷ в шкале [{c0};{c1}]={y_hat_scaled:.2f} | ближе к классу {y_class}")


if __name__ == "__main__":
    results_A, results_B = run_experiments()

    convergence_summary(results_A, "Конфигурация А (MSE)")
    convergence_summary(results_B, "Конфигурация Б (BCE)")

    plot_convergence(results_A, results_B, rep_seed=0)
    plot_epochs_bar(results_A, results_B)
    plot_decision_surfaces(results_A, results_B, rep_seed=0)
    plot_mae_comparison(results_A, results_B)

    rep_B = next(r for r in results_B if r["seed"] == 0)
    demo_predictions(rep_B)

    # interactive_mode(rep_B)