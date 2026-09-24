import numpy as np
import matplotlib.pyplot as plt

c0 = 3.0
c1 = -8.0

X = np.array([
    [3.0, 3.0],
    [3.0, -8.0],
    [-8.0, 3.0],
    [-8.0, -8.0]
])

y_raw = np.array([c0, c1, c1, c0]).reshape(-1, 1)

y_bin = np.where(y_raw == c0, 1.0, 0.0).reshape(-1, 1)

y_min, y_max = min(c0, c1), max(c0, c1)


def scale_y(y):
    return (y - y_min) / (y_max - y_min)


def descale_y(y_scaled):
    return y_scaled * (y_max - y_min) + y_min


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def dsigmoid(s):
    return s * (1.0 - s)


class MLP_2_2_1:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.W1 = np.random.uniform(-0.5, 0.5, (3, 2))
        self.W2 = np.random.uniform(-0.5, 0.5, (3, 1))

    def forward(self, x):
        # x: (n,2)
        x_b = np.hstack([x, np.ones((x.shape[0], 1))])  # (n,3)
        z1 = x_b.dot(self.W1)                           # (n,2)
        a1 = sigmoid(z1)                                # (n,2)
        a1_b = np.hstack([a1, np.ones((a1.shape[0], 1))])  # (n,3)
        z2 = a1_b.dot(self.W2)                          # (n,1)
        a2 = sigmoid(z2)                                # (n,1)
        return a2, a1, x_b, a1_b

    def train(self, X, y_target, lr, max_epochs, Ee, loss_type="mse", shuffle=True, seed=None):

        if seed is not None:
            np.random.seed(seed)

        n = X.shape[0]
        Es_history = []
        epoch_reached = None

        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            idx = np.random.permutation(n) if shuffle else np.arange(n)

            for i in idx:
                xi = X[i:i + 1]
                yi = y_target[i:i + 1]

                out, a1, x_b, a1_b = self.forward(xi)

                if loss_type == "mse":
                    err = yi - out
                    Es += 0.5 * np.sum(err ** 2)
                    delta2 = err * dsigmoid(out)
                elif loss_type == "bce":
                    eps = 1e-12
                    out_clipped = np.clip(out, eps, 1.0 - eps)
                    Es += -np.sum(yi * np.log(out_clipped) + (1.0 - yi) * np.log(1.0 - out_clipped))
                    # для связки sigmoid + BCE: dL/dz2 = out - y
                    delta2 = (out - yi)
                else:
                    raise ValueError("Unknown loss_type")

                # градиент по W2
                gradW2 = a1_b.T.dot(delta2)

                # градиент по W1
                W2_no_bias = self.W2[:-1, :]
                delta1 = (delta2.dot(W2_no_bias.T)) * dsigmoid(a1)
                gradW1 = x_b.T.dot(delta1)

                self.W2 -= lr * gradW2
                self.W1 -= lr * gradW1

            Es_history.append(Es)

            if Es <= Ee and epoch_reached is None:
                epoch_reached = epoch
                break

        return epoch_reached, Es_history


def accuracy_from_preds(preds_raw, targets_raw):
    preds = preds_raw.ravel()
    targets = targets_raw.ravel()
    pick = np.where(np.abs(preds - c0) <= np.abs(preds - c1), c0, c1)
    return np.mean(pick == targets)


def mae_from_preds(preds_raw, targets_raw):
    return np.mean(np.abs(preds_raw.ravel() - targets_raw.ravel()))


X_min, X_max = -10.0, 10.0


def scale_X(X):
    return (X - X_min) / (X_max - X_min) * 2.0 - 1.0  # в [-1;1]


X_norm = scale_X(X)


def run_experiments():
    lr = 0.1
    max_epochs = 20000
    Ee_mse = 1e-3
    Ee_bce = 1e-3

    seeds = [1, 7, 42, 123, 999]

    results_A = []  # MSE
    results_B = []  # BCE

    y_mse_target = scale_y(y_raw)

    y_bce_target = y_bin

    for seed in seeds:
        mlp_A = MLP_2_2_1(seed=seed)
        epoch_A, Es_hist_A = mlp_A.train(
            X_norm, y_mse_target,
            lr=lr, max_epochs=max_epochs, Ee=Ee_mse,
            loss_type="mse", shuffle=True, seed=seed
        )
        out_A, _, _, _ = mlp_A.forward(X_norm)
        preds_A_raw = descale_y(out_A)
        acc_A = accuracy_from_preds(preds_A_raw, y_raw)
        mae_A = mae_from_preds(preds_A_raw, y_raw)

        results_A.append({
            "seed": seed,
            "epoch": epoch_A if epoch_A is not None else max_epochs,
            "Es_final": Es_hist_A[-1],
            "acc": acc_A,
            "mae": mae_A,
            "Es_hist": Es_hist_A,
            "model": mlp_A
        })

        mlp_B = MLP_2_2_1(seed=seed + 100)
        epoch_B, Es_hist_B = mlp_B.train(
            X_norm, y_bce_target,
            lr=lr, max_epochs=max_epochs, Ee=Ee_bce,
            loss_type="bce", shuffle=True, seed=seed + 100
        )
        out_B, _, _, _ = mlp_B.forward(X_norm)
        preds_B_raw = descale_y(out_B)
        acc_B = accuracy_from_preds(preds_B_raw, y_raw)
        mae_B = mae_from_preds(preds_B_raw, y_raw)

        results_B.append({
            "seed": seed,
            "epoch": epoch_B if epoch_B is not None else max_epochs,
            "Es_final": Es_hist_B[-1],
            "acc": acc_B,
            "mae": mae_B,
            "Es_hist": Es_hist_B,
            "model": mlp_B
        })

        print(f"Seed={seed} | A(MSE): epoch={results_A[-1]['epoch']} Es={results_A[-1]['Es_final']:.4e} "
              f"acc={acc_A:.2f} mae={mae_A:.3f} | "
              f"B(BCE): epoch={results_B[-1]['epoch']} Es={results_B[-1]['Es_final']:.4e} "
              f"acc={acc_B:.2f} mae={mae_B:.3f}")

    return results_A, results_B


def plot_convergence(results_A, results_B):
    Es_A = results_A[0]["Es_hist"]
    Es_B = results_B[0]["Es_hist"]

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(Es_A) + 1), Es_A, label="MSE (A)", color="blue")
    plt.plot(range(1, len(Es_B) + 1), Es_B, label="BCE (B)", color="red")
    plt.axhline(y=1e-3, color="green", linestyle="--", label="Ee")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.title("Сходимость: MSE vs BCE")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_epoch_scatter(results_A, results_B):
    seeds = [r["seed"] for r in results_A]
    epochs_A = [r["epoch"] for r in results_A]
    epochs_B = [r["epoch"] for r in results_B]

    x = np.arange(len(seeds))

    plt.figure(figsize=(8, 6))
    width = 0.35
    plt.bar(x - width / 2, epochs_A, width, label="MSE (A)", color="blue")
    plt.bar(x + width / 2, epochs_B, width, label="BCE (B)", color="red")
    plt.xticks(x, [str(s) for s in seeds])
    plt.xlabel("Seed")
    plt.ylabel("Число эпох до сходимости (или max_epochs)")
    plt.title("Разброс числа эпох по запускам")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def plot_decision_surface(model_A, model_B):
    grid_size = 100
    A_vals = np.linspace(-10, 10, grid_size)
    B_vals = np.linspace(-10, 10, grid_size)
    AA, BB = np.meshgrid(A_vals, B_vals)
    grid = np.stack([AA.ravel(), BB.ravel()], axis=1)  # (N,2)
    grid_norm = scale_X(grid)

    out_A, _, _, _ = model_A.forward(grid_norm)
    out_B, _, _, _ = model_B.forward(grid_norm)

    # пересчёт в исходную шкалу
    Z_A = descale_y(out_A).reshape(AA.shape)
    Z_B = descale_y(out_B).reshape(AA.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # A: MSE
    cs1 = axes[0].contourf(AA, BB, Z_A, levels=20, cmap="coolwarm")
    axes[0].scatter(X[:, 0], X[:, 1],
                    c=np.where(y_raw.ravel() == c0, "yellow", "black"),
                    edgecolors="k", s=80, label="Обучающие точки")
    axes[0].set_title("Разделяющая поверхность (MSE)")
    axes[0].set_xlabel("A")
    axes[0].set_ylabel("B")
    axes[0].legend()
    fig.colorbar(cs1, ax=axes[0])

    # B: BCE
    cs2 = axes[1].contourf(AA, BB, Z_B, levels=20, cmap="coolwarm")
    axes[1].scatter(X[:, 0], X[:, 1],
                    c=np.where(y_raw.ravel() == c0, "yellow", "black"),
                    edgecolors="k", s=80, label="Обучающие точки")
    axes[1].set_title("Разделяющая поверхность (BCE)")
    axes[1].set_xlabel("A")
    axes[1].set_ylabel("B")
    axes[1].legend()
    fig.colorbar(cs2, ax=axes[1])

    plt.tight_layout()
    plt.show()


def plot_mae_bar(results_A, results_B):
    mae_A = np.mean([r["mae"] for r in results_A])
    mae_B = np.mean([r["mae"] for r in results_B])

    plt.figure(figsize=(6, 5))
    plt.bar(["MSE (A)", "BCE (B)"], [mae_A, mae_B], color=["blue", "red"])
    plt.ylabel("Средняя абсолютная ошибка (MAE) в шкале [c0;c1]")
    plt.title("Сравнение точности восстановления шкалы")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


def interactive_mode(model_B):
    print("Интерактивный режим. Введите A B (в диапазоне [-10;10]) или 'exit'.")

    while True:
        s = input("A B: ")
        if s.strip().lower() == "exit":
            break
        try:
            a, b = map(float, s.split())
        except Exception:
            print("Ошибка ввода. Формат: A B или exit.")
            continue

        x = np.array([[a, b]])
        x_n = scale_X(x)

        out, _, _, _ = model_B.forward(x_n)
        y_norm = out[0, 0]  # ŷ в (0;1)
        y_raw_pred = descale_y(out)[0, 0]

        cls = c0 if abs(y_raw_pred - c0) < abs(y_raw_pred - c1) else c1

        print(f"ŷ (норм.) = {y_norm:.4f}")
        print(f"ŷ (в шкале [c0;c1]) = {y_raw_pred:.4f}")
        print(f"Ближайший класс: {cls}")
        print("-" * 40)


if __name__ == "__main__":
    results_A, results_B = run_experiments()

    plot_convergence(results_A, results_B)
    plot_epoch_scatter(results_A, results_B)
    model_A = results_A[0]["model"]
    model_B = results_B[0]["model"]
    plot_decision_surface(model_A, model_B)
    plot_mae_bar(results_A, results_B)

    interactive_mode(model_B)
