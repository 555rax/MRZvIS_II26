import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

C0 = -3.0
C1 = -8.0
LEARNING_RATE = 0.1
MAX_EPOCHS = 20_000
SEEDS = (0, 1, 2, 3, 4)
LIMITS = {"A": 0.01, "B": 0.05}
X_REAL = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]])
Y_REAL = np.array([[C0], [C1], [C1], [C0]])
Y_BIN = (Y_REAL - C0) / (C1 - C0)
COLORS = {"A": "#2369a1", "B": "#d87519"}


def encode_inputs(values):
    return 2.0 * (np.asarray(values, dtype=float) - C0) / (C1 - C0) - 1.0


def decode_output(output):
    return C0 + (C1 - C0) * output


def nearest_class(values):
    values = np.asarray(values)
    return np.where(np.abs(values - C0) <= np.abs(values - C1), C0, C1)


def sigmoid(x):
    return np.exp(-np.logaddexp(0.0, -x))


class MLP:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.zeros((1, 2))
        self.W2 = rng.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        hidden = sigmoid(x @ self.W1 + self.b1)
        logits = hidden @ self.W2 + self.b2
        return hidden, logits, sigmoid(logits)

    def predict(self, x):
        return self.forward(x)[2]

    def update(self, x, y, mode, lr):
        hidden, _, output = self.forward(x)
        if mode == "A":
            delta = 2.0 * (output - y) * output * (1.0 - output)
        else:
            delta = output - y
        hidden_delta = (delta @ self.W2.T) * hidden * (1.0 - hidden)
        self.W2 -= lr * (hidden.T @ delta)
        self.b2 -= lr * delta
        self.W1 -= lr * (x.T @ hidden_delta)
        self.b1 -= lr * hidden_delta


def total_error(model, x, targets, mode):
    _, logits, output = model.forward(x)
    if mode == "A":
        return float(np.sum((output - targets) ** 2))
    return float(np.sum(np.logaddexp(0.0, logits) - targets * logits))


def assess(model, mode):
    output = model.predict(encode_inputs(X_REAL))
    values = output if mode == "A" else decode_output(output)
    return {
        "accuracy": float(np.mean(nearest_class(values) == Y_REAL) * 100.0),
        "mae": float(np.mean(np.abs(values - Y_REAL))),
        "values": values.ravel(),
    }


def train(seed, mode, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS):
    model = MLP(seed)
    x = encode_inputs(X_REAL)
    targets = Y_REAL if mode == "A" else Y_BIN
    history = [total_error(model, x, targets, mode)]
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        for i in range(len(x)):
            model.update(x[i:i + 1], targets[i:i + 1], mode, lr)
        history.append(total_error(model, x, targets, mode))
        if history[-1] <= LIMITS[mode]:
            break
    return {
        "model": model, "mode": mode, "seed": seed, "epochs": epoch,
        "converged": history[-1] <= LIMITS[mode], "loss": history[-1],
        "history": np.asarray(history), **assess(model, mode),
    }


def print_results(runs):
    print("\nКонф. Seed    Эпохи            Es   Accuracy         MAE  Порог")
    for r in runs:
        status = "достигнут" if r["converged"] else "не достигнут"
        print(f"{r['mode']:>5} {r['seed']:4d} {r['epochs']:8d} "
              f"{r['loss']:13.8f} {r['accuracy']:9.2f}% "
              f"{r['mae']:11.8f}  {status}")
    for mode in ("A", "B"):
        selected = [r for r in runs if r["mode"] == mode]
        epochs = [r["epochs"] for r in selected if r["converged"]]
        print(f"\n{mode}: сошлось {len(epochs)}/{len(selected)} запусков.")
        if epochs:
            print(f"Эпохи успешных запусков: min={min(epochs)}, "
                  f"max={max(epochs)}, mean={np.mean(epochs):.1f}, "
                  f"std={np.std(epochs):.1f}, размах={max(epochs) - min(epochs)}.")
        else:
            print("Число эпох до сходимости не определено: порог не достигнут.")


def save_results(runs, directory):
    lines = ["configuration,seed,epochs,converged,Es,accuracy_percent,MAE"]
    for r in runs:
        lines.append(f"{r['mode']},{r['seed']},{r['epochs']},"
                     f"{r['converged']},{r['loss']:.12f},"
                     f"{r['accuracy']:.2f},{r['mae']:.12f}")
    (directory / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def finish_plot(fig, directory, name):
    fig.savefig(directory / name, dpi=160, bbox_inches="tight")


def draw_plots(runs, directory, seed=0):
    picked = {mode: next(r for r in runs if r["mode"] == mode and r["seed"] == seed)
              for mode in ("A", "B")}
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for mode, title in (("A", "А: сумма квадратов"), ("B", "Б: сумма BCE")):
        ax.plot(picked[mode]["history"], label=title, color=COLORS[mode])
        ax.axhline(LIMITS[mode], color=COLORS[mode], linestyle="--",
                   label=f"Ee {mode} = {LIMITS[mode]}")
    ax.set(yscale="log", xlabel="Завершённые эпохи (0 — до обучения)",
           ylabel="Суммарная ошибка Es (логарифмическая шкала)",
           title=f"Сходимость двух конфигураций, seed={seed}")
    ax.grid(alpha=0.2)
    ax.legend()
    finish_plot(fig, directory, "01_convergence.png")

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    positions = np.arange(len(SEEDS))
    for mode, shift in (("A", -0.2), ("B", 0.2)):
        subset = [r for r in runs if r["mode"] == mode]
        bars = ax.bar(positions + shift, [r["epochs"] for r in subset],
                      width=0.4, color=COLORS[mode], label=f"Конфигурация {mode}")
        for bar, r in zip(bars, subset):
            if not r["converged"]:
                bar.set_hatch("///")
                bar.set_edgecolor("black")
    ax.set(xticks=positions, xticklabels=[str(s) for s in SEEDS], xlabel="Seed",
           ylabel="Эпохи до порога или до лимита",
           title="Устойчивость сходимости в пяти запусках")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [Patch(facecolor="white", hatch="///", edgecolor="black")],
              labels + ["Порог не достигнут"])
    finish_plot(fig, directory, "02_epochs.png")

    axis = np.linspace(-10, 10, 250)
    grid_a, grid_b = np.meshgrid(axis, axis)
    grid = encode_inputs(np.column_stack([grid_a.ravel(), grid_b.ravel()]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    point_cmap = ListedColormap(["#ffe15b", "#992878"])
    for ax, mode in zip(axes, ("A", "B")):
        output = picked[mode]["model"].predict(grid).reshape(grid_a.shape)
        values = output if mode == "A" else decode_output(output)
        limits = (0.0, 1.0) if mode == "A" else (C1, C0)
        surface = ax.pcolormesh(grid_a, grid_b, values, shading="auto",
                                cmap="viridis", vmin=limits[0], vmax=limits[1])
        fig.colorbar(surface, ax=ax, label="Выход сигмоиды" if mode == "A"
                     else "Выход после пересчёта: −3 − 5p")
        boundary = (C0 + C1) / 2.0
        if values.min() < boundary < values.max():
            ax.contour(grid_a, grid_b, values, levels=[boundary],
                       colors="white", linewidths=1.5)
        ax.scatter(X_REAL[:, 0], X_REAL[:, 1], c=Y_BIN.ravel(),
                   cmap=point_cmap, vmin=0, vmax=1, s=95, edgecolors="black")
        ax.set(xlim=(-10, 10), ylim=(-10, 10), xlabel="Вход A", ylabel="Вход B",
               title=f"Конфигурация {mode}, seed={seed}", aspect="equal")
        ax.legend(handles=[Patch(facecolor="#ffe15b", label="Истинный c0 = −3"),
                           Patch(facecolor="#992878", label="Истинный c1 = −8")],
                  loc="upper right", fontsize=8)
    fig.suptitle("Карта выхода сети; белая линия — граница классов ŷ = −5,5")
    finish_plot(fig, directory, "03_surfaces.png")

    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    means = [np.mean([r["mae"] for r in runs if r["mode"] == mode])
             for mode in ("A", "B")]
    bars = ax.bar(["А: исходные цели + MSE", "Б: метки 0/1 + BCE"], means,
                  color=[COLORS["A"], COLORS["B"]])
    ax.bar_label(bars, fmt="%.5f", padding=4)
    ax.set(ylabel="MAE в исходной шкале, среднее по пяти запускам",
           xlabel="Конфигурация", title="Точность восстановления числовых классов",
           ylim=(0, max(means) * 1.18))
    finish_plot(fig, directory, "04_mae.png")


def predict_pair(model, a, b):
    p = model.predict(encode_inputs([[a, b]])).item()
    value = decode_output(p)
    label = nearest_class(value).item()
    return p, value, label


def show_examples(model):
    pairs = [tuple(row) for row in X_REAL] + [(-2, -7), (-7, -2), (-9, -9)]
    print("\nРежим функционирования конфигурации Б:")
    print("       A        B       p         ŷ    Класс")
    for a, b in pairs:
        p, value, label = predict_pair(model, a, b)
        print(f"{a:8.1f} {b:8.1f} {p:7.4f} {value:9.4f} {label:8.0f}")


def interactive_mode(model):
    print("\nВведите A B из [-10; 10]; q — выход.")
    while True:
        try:
            line = input("A B: ").strip()
            if line.lower() in {"q", "exit", "выход"}:
                break
            try:
                a, b = map(float, line.replace(",", " ").split())
            except ValueError:
                print("Нужны два числа через пробел, например: -3 -8.")
                continue
            if not (np.isfinite(a) and np.isfinite(b)
                    and -10 <= a <= 10 and -10 <= b <= 10):
                print("Введите конечные числа в диапазоне [-10; 10].")
                continue
            p, value, label = predict_pair(model, a, b)
            class_name = "c0" if label == C0 else "c1"
            print(f"p = {p:.4f}; ŷ = {value:.4f}; {class_name} = {label:g}")
        except (EOFError, KeyboardInterrupt):
            break
    print("Работа завершена.")


def main():
    parser = argparse.ArgumentParser(description="ЛР №2, вариант 10: MSE и BCE")
    parser.add_argument("--no-input", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("lab2_results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"c0={C0:g}, c1={C1:g}; lr={LEARNING_RATE}; "
          f"лимит={MAX_EPOCHS}; Ee(A)={LIMITS['A']}; Ee(Б)={LIMITS['B']}")
    print("А: Es = сумма квадратов; Б: Es = сумма BCE. Это разные шкалы потерь.")
    print("Для А Es > 146: сигмоида не может воспроизвести отрицательные цели.")
    runs = []
    for mode in ("A", "B"):
        for seed in SEEDS:
            result = train(seed, mode)
            runs.append(result)
            print(f"Завершено: конфигурация {mode}, seed={seed}, "
                  f"эпох={result['epochs']}, Es={result['loss']:.8f}", flush=True)
    print_results(runs)
    save_results(runs, args.output_dir)
    draw_plots(runs, args.output_dir, seed=0)
    representative = next(r for r in runs if r["mode"] == "B" and r["seed"] == 0)
    if not representative["converged"]:
        print("Внимание: представительный запуск Б не достиг порога.")
    show_examples(representative["model"])
    print(f"\nТаблица и четыре графика сохранены: {args.output_dir.resolve()}")
    if args.no_show:
        plt.close("all")
    else:
        plt.show()
    if not args.no_input:
        interactive_mode(representative["model"])


if __name__ == "__main__":
    main()
