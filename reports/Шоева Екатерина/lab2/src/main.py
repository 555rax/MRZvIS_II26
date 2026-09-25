import numpy as np
import matplotlib.pyplot as plt

C0 = 5
C1 = -2

RAW_DATA = [
    (C0, C0, C0),
    (C0, C1, C1),
    (C1, C0, C1),
    (C1, C1, C0),
]

LEARNING_RATE = 0.5
EE_MSE = 0.01
EE_BCE = 0.05
MAX_EPOCHS = 10000
SEEDS = [1, 2, 3, 4, 5]
REPRESENTATIVE_SEED = 1
EPS = 1e-9



def encode_value(x):

    return 1.0 if x == C0 else 0.0


def decode_value_continuous(y):

    return C1 + y * (C0 - C1)


def decode_class(y):

    return C0 if y >= 0.5 else C1


def encode_arbitrary(x):

    return 1.0 if abs(x - C0) < abs(x - C1) else 0.0


def encode_continuous(x):

    return (x - C1) / (C0 - C1)


def build_training_set():
    X, Y = [], []
    for a, b, target in RAW_DATA:
        X.append([encode_value(a), encode_value(b)])
        Y.append([encode_value(target)])
    return np.array(X), np.array(Y)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(y):
    return y * (1.0 - y)


class MLP221:
    def __init__(self, loss_type="mse", lr=LEARNING_RATE, seed=0):
        assert loss_type in ("mse", "bce")
        self.loss_type = loss_type
        self.lr = lr
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.5, 0.5, size=(2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, size=(2,))
        self.W2 = rng.uniform(-0.5, 0.5, size=(2, 1))
        self.b2 = rng.uniform(-0.5, 0.5, size=(1,))

    def forward(self, x):
        hidden_in = x @ self.W1 + self.b1
        hidden_out = sigmoid(hidden_in)
        out_in = hidden_out @ self.W2 + self.b2
        out = sigmoid(out_in)
        return hidden_out, out

    def loss_and_delta_out(self, target, out):

        if self.loss_type == "mse":
            error = target - out
            loss = float(np.sum(error ** 2))
            delta_out = error * sigmoid_derivative(out)
        else:
            y = np.clip(out, EPS, 1 - EPS)
            loss = float(-np.sum(target * np.log(y) + (1 - target) * np.log(1 - y)))

            delta_out = target - out
        return loss, delta_out

    def train_step(self, x, target):
        hidden_out, out = self.forward(x)
        loss, delta_out = self.loss_and_delta_out(target, out)

        error_hidden = delta_out @ self.W2.T
        delta_hidden = error_hidden * sigmoid_derivative(hidden_out)

        self.W2 += self.lr * np.outer(hidden_out, delta_out)
        self.b2 += self.lr * delta_out
        self.W1 += self.lr * np.outer(x, delta_hidden)
        self.b1 += self.lr * delta_hidden

        return loss

    def predict(self, x):
        _, out = self.forward(np.array(x, dtype=float))
        return float(out[0])

    def train(self, X, Y, ee, max_epochs=MAX_EPOCHS):
        history = []
        Es = float("inf")
        for epoch in range(1, max_epochs + 1):
            Es = 0.0
            for i in range(len(X)):
                Es += self.train_step(X[i], Y[i])
            history.append(Es)
            if Es <= ee:
                return epoch, Es, history, True
        return max_epochs, Es, history, False


def accuracy(model, X, Y):
    correct = 0
    for i in range(len(X)):
        pred_class = 1.0 if model.predict(X[i]) >= 0.5 else 0.0
        if pred_class == Y[i][0]:
            correct += 1
    return correct / len(X)


def mae_real_scale(model, X):

    errors = []
    for x, (a, b, target) in zip(X, RAW_DATA):
        y_hat = model.predict(x)
        pred_real = decode_value_continuous(y_hat)
        errors.append(abs(pred_real - target))
    return float(np.mean(errors))


def run_series(loss_type, ee, X, Y, seeds=SEEDS):
    results = []
    for seed in seeds:
        model = MLP221(loss_type=loss_type, seed=seed)
        epochs, Es, history, converged = model.train(X, Y, ee=ee)
        results.append({
            "seed": seed,
            "epochs": epochs,
            "Es": Es,
            "converged": converged,
            "accuracy": accuracy(model, X, Y),
            "mae": mae_real_scale(model, X),
            "model": model,
            "history": history,
        })
    return results


def print_series_table(title, results):
    print(f"\n{title}")
    print(f"{'seed':>6}{'epochs':>10}{'converged':>12}{'Es':>14}{'accuracy':>10}{'MAE':>10}")
    for r in results:
        print(f"{r['seed']:>6}{r['epochs']:>10}{str(r['converged']):>12}"
              f"{r['Es']:>14.6f}{r['accuracy']:>10.2f}{r['mae']:>10.4f}")



def plot_convergence(hist_a, hist_b, filename="conv_plot.png"):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(hist_a) + 1), hist_a, label="Конфигурация А (MSE)", color="tab:blue")
    plt.plot(range(1, len(hist_b) + 1), hist_b, label="Конфигурация Б (BCE)", color="tab:orange")
    plt.axhline(EE_MSE, color="tab:blue", linestyle="--", linewidth=0.8, label=f"Ee (MSE) = {EE_MSE}")
    plt.axhline(EE_BCE, color="tab:orange", linestyle="--", linewidth=0.8, label=f"Ee (BCE) = {EE_BCE}")
    plt.title("Сходимость: суммарная ошибка Es по эпохам (MSE vs BCE)")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранён график сходимости: {filename}")


def plot_epochs_bar(results_a, results_b, filename="epochs_bar.png"):
    seeds = [r["seed"] for r in results_a]
    n = len(seeds)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, r in enumerate(results_a):
        color = "tab:blue" if r["converged"] else "none"
        edge = "tab:blue"
        hatch = None if r["converged"] else "//"
        ax.bar(x[i] - width / 2, r["epochs"], width, color=color, edgecolor=edge, hatch=hatch)
    for i, r in enumerate(results_b):
        color = "tab:orange" if r["converged"] else "none"
        edge = "tab:orange"
        hatch = None if r["converged"] else "//"
        ax.bar(x[i] + width / 2, r["epochs"], width, color=color, edgecolor=edge, hatch=hatch)


    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="tab:blue", label="MSE, сошлось"),
        Patch(facecolor="none", edgecolor="tab:blue", hatch="//", label="MSE, не сошлось"),
        Patch(facecolor="tab:orange", label="BCE, сошлось"),
        Patch(facecolor="none", edgecolor="tab:orange", hatch="//", label="BCE, не сошлось"),
    ]
    ax.legend(handles=legend_elems)
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed={s}" for s in seeds])
    ax.set_xlabel("Запуск (seed)")
    ax.set_ylabel("Число эпох до сходимости")
    ax.set_title("Устойчивость сходимости: 5 запусков, MSE vs BCE")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранена диаграмма разброса эпох: {filename}")


def plot_decision_surface(model_a, model_b, filename="surface_plot.png"):
    grid = np.linspace(-10, 10, 200)
    AA, BB = np.meshgrid(grid, grid)

    def surface(model):
        Z = np.zeros_like(AA)
        for i in range(AA.shape[0]):
            for j in range(AA.shape[1]):

                x = [encode_continuous(AA[i, j]), encode_continuous(BB[i, j])]
                y_hat = model.predict(x)
                Z[i, j] = decode_value_continuous(y_hat)
        return Z

    Za = surface(model_a)
    Zb = surface(model_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, Z, title in zip(axes, [Za, Zb], ["Разделяющая поверхность (MSE)", "Разделяющая поверхность (BCE)"]):
        im = ax.contourf(AA, BB, Z, levels=25, cmap="coolwarm_r")
        ax.contour(AA, BB, Z, levels=25, colors="k", linewidths=0.3, alpha=0.5)
        fig.colorbar(im, ax=ax, label="Выход сети (шкала [c1; c0])")
        pts_a = [p[0] for p in RAW_DATA]
        pts_b = [p[1] for p in RAW_DATA]
        ax.scatter(pts_a, pts_b, s=110, c="crimson", edgecolors="k", zorder=5, label="Точки XOR")
        ax.set_xlabel("Вход A")
        ax.set_ylabel("Вход B")
        ax.set_title(title)
        ax.legend(loc="upper right")
    plt.suptitle("Визуализация разделяющей поверхности для обеих конфигураций")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранена визуализация разделяющей поверхности: {filename}")


def plot_mae_bar(mae_a, mae_b, filename="mae_bar.png"):
    plt.figure(figsize=(5, 4.5))
    bars = plt.bar(["MSE (конфиг. А)", "BCE (конфиг. Б)"], [mae_a, mae_b],
                    color=["tab:blue", "tab:orange"])
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h, f"{h:.4f}", ha="center", va="bottom")
    plt.ylabel("Средняя абсолютная ошибка (шкала [c1; c0])")
    plt.title("Точность восстановления исходной шкалы")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Сохранена диаграмма MAE: {filename}")

def run_and_print(model, a, b, label):
    x = [encode_arbitrary(a), encode_arbitrary(b)]
    y_hat = model.predict(x)
    y_real = decode_value_continuous(y_hat)
    cls = decode_class(y_hat)
    print(f"  {label:<28} A={a:>5} B={b:>5} | вход={x} | "
          f"y_hat={y_hat:.4f} | y_real={y_real:>7.3f} | ближе к классу {'c0' if cls == C0 else 'c1'} ({cls})")


def demo_examples(model, model_name):
    print(f"\n--- Демонстрация работы сети «{model_name}» ---")
    print("На 4 примерах обучающей выборки:")
    for a, b, target in RAW_DATA:
        run_and_print(model, a, b, f"target={target}")
    print("На дополнительных парах (вне обучающей выборки):")
    for a, b in [(3, 7), (-5, -8), (0, 0)]:
        run_and_print(model, a, b, "вне выборки")


def interactive_mode(model, model_name):
    print(f"\n--- Режим работы обученной сети ({model_name}) ---")
    print("Введите пару чисел A, B из диапазона [-10, 10] (или 'q' для выхода).")
    while True:
        raw = input("Введите A B через пробел: ").strip()
        if raw.lower() in ("q", "quit", "exit", ""):
            print("Завершение режима ввода.")
            break
        try:
            parts = raw.replace(",", " ").split()
            a, b = float(parts[0]), float(parts[1])
        except (ValueError, IndexError):
            print("Некорректный ввод. Пример: 3.5 -7")
            continue
        if not (-10 <= a <= 10 and -10 <= b <= 10):
            print("Значения должны быть в диапазоне [-10, 10].")
            continue
        run_and_print(model, a, b, "ввод пользователя")



def main():
    X, Y = build_training_set()
    print("Обучающая выборка (закодированная):")
    for x, y, (a, b, t) in zip(X, Y, RAW_DATA):
        print(f"  A={a:>3} B={b:>3} -> вход={x} target={y} (реальный={t})")


    print("\n" + "=" * 70)
    print("Серия из 5 запусков — Конфигурация А (MSE)")
    print("=" * 70)
    results_a = run_series("mse", EE_MSE, X, Y)
    print_series_table("Результаты (MSE):", results_a)

    print("\n" + "=" * 70)
    print("Серия из 5 запусков — Конфигурация Б (BCE)")
    print("=" * 70)
    results_b = run_series("bce", EE_BCE, X, Y)
    print_series_table("Результаты (BCE):", results_b)

    conv_a = sum(r["converged"] for r in results_a)
    conv_b = sum(r["converged"] for r in results_b)
    epochs_a_conv = [r["epochs"] for r in results_a if r["converged"]]
    epochs_b_conv = [r["epochs"] for r in results_b if r["converged"]]

    print("\n" + "=" * 70)
    print("Сводка по устойчивости сходимости")
    print("=" * 70)
    print(f"MSE: сошлось {conv_a}/5 запусков; "
          f"эпох (сошедшиеся): {epochs_a_conv} "
          f"(среднее={np.mean(epochs_a_conv) if epochs_a_conv else float('nan'):.1f})")
    print(f"BCE: сошлось {conv_b}/5 запусков; "
          f"эпох (сошедшиеся): {epochs_b_conv} "
          f"(среднее={np.mean(epochs_b_conv) if epochs_b_conv else float('nan'):.1f})")


    def pick_representative(results):
        converged = [r for r in results if r["converged"]]
        if converged:
            return min(converged, key=lambda r: r["epochs"])
        return min(results, key=lambda r: r["Es"])

    res_a_rep = pick_representative(results_a)
    res_b_rep = pick_representative(results_b)
    model_a, hist_a = res_a_rep["model"], res_a_rep["history"]
    model_b, hist_b = res_b_rep["model"], res_b_rep["history"]

    print("\n" + "=" * 70)
    print(f"Итоговая таблица (представительные запуски: "
          f"MSE seed={res_a_rep['seed']}, BCE seed={res_b_rep['seed']})")
    print("=" * 70)
    print(f"{'Критерий':<35}{'Конфиг. А (MSE)':>18}{'Конфиг. Б (BCE)':>18}")
    print(f"{'Эпох до сходимости':<35}{res_a_rep['epochs']:>18}{res_b_rep['epochs']:>18}")
    print(f"{'Критерий Es<=Ee достигнут':<35}{str(res_a_rep['converged']):>18}{str(res_b_rep['converged']):>18}")
    print(f"{'Итоговая Es':<35}{res_a_rep['Es']:>18.6f}{res_b_rep['Es']:>18.6f}")
    print(f"{'Accuracy':<35}{res_a_rep['accuracy']:>18.2f}{res_b_rep['accuracy']:>18.2f}")
    print(f"{'MAE (шкала c0/c1)':<35}{res_a_rep['mae']:>18.4f}{res_b_rep['mae']:>18.4f}")


    plot_convergence(hist_a, hist_b)
    plot_epochs_bar(results_a, results_b)
    plot_decision_surface(model_a, model_b)
    plot_mae_bar(res_a_rep["mae"], res_b_rep["mae"])


    demo_examples(model_a, "Конфигурация А (MSE)")
    demo_examples(model_b, "Конфигурация Б (BCE)")


    print("\nВыберите сеть для интерактивного режима:")
    print("  1 - Конфигурация А (MSE)")
    print("  2 - Конфигурация Б (BCE)")
    choice = input("Ваш выбор (1/2, Enter для пропуска): ").strip()
    if choice == "1":
        interactive_mode(model_a, "Конфигурация А (MSE)")
    elif choice == "2":
        interactive_mode(model_b, "Конфигурация Б (BCE)")
    else:
        print("Интерактивный режим пропущен.")


if __name__ == "__main__":
    main()
