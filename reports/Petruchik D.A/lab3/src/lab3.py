import sys
import numpy as np
import matplotlib.pyplot as plt

c0, c1 = -3.0, -8.0

X = np.array([
    [-3, -3],
    [-3, -8],
    [-8, -3],
    [-8, -8]
], dtype=float)

Y_raw = np.array([
    [-3],
    [-8],
    [-8],
    [-3]
], dtype=float)

def normalize(x, lo, hi):
    return (x - lo) / (hi - lo)


def denormalize(y, lo, hi):
    return lo + y * (hi - lo)


Xn = normalize(X, c0, c1)
Yn = normalize(Y_raw, c0, c1)

EPS = 1e-12


def sigmoid(s):
    s = np.clip(s, -500, 500)
    return 1.0 / (1.0 + np.exp(-s))


def dsigmoid(y):
    return y * (1.0 - y)


def relu(s):
    return np.maximum(0.0, s)


def drelu(s):
    return (s > 0).astype(float)

class MLP221:

    def __init__(self, hidden="sigmoid", lr=0.5, seed=7):
        rng = np.random.RandomState(seed)
        self.W1 = rng.uniform(-0.1, 0.1, (2, 2))
        self.T1 = rng.uniform(-0.1, 0.1, (1, 2))
        self.W2 = rng.uniform(-0.1, 0.1, (2, 1))
        self.T2 = rng.uniform(-0.1, 0.1, (1, 1))

        self.lr = lr
        self.hidden = hidden

        self.rng = np.random.RandomState(seed + 1000)

    def forward(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        s1 = x @ self.W1 - self.T1

        if self.hidden == "relu":
            y1 = relu(s1)
        else:
            y1 = sigmoid(s1)
        y2 = sigmoid(y1 @ self.W2 - self.T2)

        return s1, y1, y2

    def error(self, y, e):
        y = np.clip(y, EPS, 1.0 - EPS)
        return -np.sum(
            e * np.log(y) +
            (1.0 - e) * np.log(1.0 - y)
        )

    def train_step(self, x, e):
        x = x.reshape(1, -1)
        e = e.reshape(1, -1)

        s1, y1, y2 = self.forward(x)
        delta2 = y2 - e
        back = delta2 @ self.W2.T
        if self.hidden == "relu":
            d1 = drelu(s1)
        else:
            d1 = dsigmoid(y1)

        delta1 = back * d1

        self.W2 -= self.lr * (y1.T @ delta2)
        self.T2 += self.lr * delta2
        self.W1 -= self.lr * (x.T @ delta1)
        self.T1 += self.lr * delta1

        return self.error(y2, e)

    def predict(self, x):
        return self.forward(x)[2][0, 0]

    def fit(self, X, Y, Ee, max_epochs=20000):
        hist = []

        for epoch in range(1, max_epochs + 1):
            idx = self.rng.permutation(len(X))

            Es = sum(
                self.train_step(X[i], Y[i])
                for i in idx
            )

            hist.append(Es)

            if Es <= Ee:
                return epoch, Es, True, hist

        return max_epochs, Es, False, hist

def accuracy(m, Xn, Y):
    predictions = [
        int(m.predict(x) >= 0.5)
        for x in Xn
    ]

    targets = [
        int(y[0] >= 0.5)
        for y in Y
    ]

    return np.mean(
        np.array(predictions) == np.array(targets)
    )


def mae_orig(m, Xn, Yraw):
    predictions = np.array([
        denormalize(m.predict(x), c0, c1)
        for x in Xn
    ])

    return float(
        np.mean(np.abs(predictions - Yraw.ravel()))
    )


def final_Es(m, Xn, Y):
    return sum(
        m.error(m.forward(x)[2], y.reshape(1, -1))
        for x, y in zip(Xn, Y)
    )

def infer(m, A, B, name):
    xn = normalize(
        np.array([A, B], dtype=float),
        c0,
        c1
    )

    yn = m.predict(xn)
    yh = denormalize(yn, c0, c1)

    cls = c0 if abs(yh - c0) <= abs(yh - c1) else c1

    print(
        f"[{name}] A={A:g}, B={B:g} -> "
        f"y_norm={yn:.4f}, "
        f"y_hat={yh:.4f}, "
        f"класс {cls:g}"
    )

CONFIGS = {
    "A (Sigmoid)": {
        "hidden": "sigmoid",
        "Ee": 0.05
    },

    "B (ReLU)": {
        "hidden": "relu",
        "Ee": 0.05
    }
}

SEEDS = [1, 2, 3, 4, 5]
MAX_EPOCHS = 20000

colors = {
    "A (Sigmoid)": "tab:blue",
    "B (ReLU)": "tab:red"
}


def run(cfg, seed):
    m = MLP221(
        hidden=cfg["hidden"],
        seed=seed
    )

    ep, Es, ok, hist = m.fit(
        Xn,
        Yn,
        cfg["Ee"],
        MAX_EPOCHS
    )

    result = {
        "seed": seed,
        "epochs": ep,
        "Es": final_Es(m, Xn, Yn),
        "ok": ok,
        "acc": accuracy(m, Xn, Yn),
        "mae": mae_orig(m, Xn, Y_raw),
        "hist": hist
    }

    return m, result

if __name__ == "__main__":

    res = {}
    models = {}

    for name, cfg in CONFIGS.items():
        res[name] = []
        models[name] = []

        for seed in SEEDS:
            m, r = run(cfg, seed)

            res[name].append(r)
            models[name].append(m)

        print(
            f"\nКонфигурация {name}, Ee={cfg['Ee']}"
        )

        print(
            "seed | эпохи | Es | accuracy | "
            "MAE[c0;c1] | достигнут Ee"
        )

        for r in res[name]:
            print(
                f"{r['seed']:>4} | "
                f"{r['epochs']:>5} | "
                f"{r['Es']:.5f} | "
                f"{r['acc']:.2f} | "
                f"{r['mae']:.4f} | "
                f"{r['ok']}"
            )

    print("\nСводная таблица (5 запусков)")

    print(
        f"{'Конфигурация':<14}|"
        f"{'Сошлось':>8}|"
        f"{'Эпохи min':>10}|"
        f"{'max':>7}|"
        f"{'mean':>8}|"
        f"{'std':>7}|"
        f"{'Es mean':>9}|"
        f"{'acc mean':>9}|"
        f"{'MAE mean':>9}|"
        f"{'MAE std':>8}"
    )

    for name in CONFIGS:
        R = res[name]
        ep = [r["epochs"] for r in R]

        print(
            f"{name:<14}|"
            f"{sum(r['ok'] for r in R):>6}/5|"
            f"{min(ep):>10}|"
            f"{max(ep):>7}|"
            f"{np.mean(ep):>8.0f}|"
            f"{np.std(ep):>7.0f}|"
            f"{np.mean([r['Es'] for r in R]):>9.4f}|"
            f"{np.mean([r['acc'] for r in R]):>9.2f}|"
            f"{np.mean([r['mae'] for r in R]):>9.4f}|"
            f"{np.std([r['mae'] for r in R]):>8.4f}"
        )

    rep = {}

    for name in CONFIGS:
        successful = [
            i for i, r in enumerate(res[name])
            if r["ok"]
        ]

        candidates = successful or list(range(len(SEEDS)))

        rep[name] = sorted(
            candidates,
            key=lambda k: res[name][k]["epochs"]
        )[len(candidates) // 2]
    plt.figure(figsize=(8, 5))

    for name in CONFIGS:
        r = res[name][rep[name]]

        plt.plot(
            range(1, len(r["hist"]) + 1),
            r["hist"],
            color=colors[name],
            label=f"{name}, seed={r['seed']}"
        )

    plt.axhline(
        0.05,
        color="k",
        linestyle=":",
        label="Ee = 0.05"
    )

    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("Суммарная ошибка Es")
    plt.title("Сходимость: Sigmoid vs ReLU (BCE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("1_convergence.png", dpi=150)
    plt.show()
    plt.close()
    plt.figure(figsize=(8, 5))

    w = 0.38
    xs = np.arange(len(SEEDS))

    for k, name in enumerate(CONFIGS):
        for j, r in enumerate(res[name]):
            plt.bar(
                xs[j] + (k - 0.5) * w,
                r["epochs"],
                w,
                color=colors[name],
                edgecolor="k",
                hatch=None if r["ok"] else "//",
                alpha=1 if r["ok"] else 0.5,
                label=name if j == 0 else None
            )

    plt.bar(
        0, 0,
        color="w",
        edgecolor="k",
        hatch="//",
        label="Не достигнут Ee"
    )

    plt.xticks(xs, [f"seed {s}" for s in SEEDS])
    plt.yscale("log")
    plt.xlabel("Запуск")
    plt.ylabel("Количество эпох")
    plt.title("Устойчивость сходимости (5 запусков)")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("2_epochs.png", dpi=150)
    plt.show()
    plt.close()

    g = np.linspace(-10, 10, 200)
    GA, GB = np.meshgrid(g, g)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for ax, name in zip(axs, CONFIGS):
        m = models[name][rep[name]]

        Z = np.array([
            [
                denormalize(
                    m.predict(
                        normalize(
                            np.array([a, b]),
                            c0,
                            c1
                        )
                    ),
                    c0,
                    c1
                )
                for a in g
            ]
            for b in g
        ])

        cs = ax.contourf(
            GA, GB, Z,
            levels=30,
            cmap="viridis"
        )

        ax.contour(
            GA, GB, Z,
            levels=[(c0 + c1) / 2],
            colors="w",
            linestyles="--"
        )

        for xi, yi in zip(X, Y_raw.ravel()):
            color = (
                [1, 0.85, 0]
                if yi == c1
                else [1, 0.2, 0.2]
            )

            ax.scatter(
                *xi,
                c=[color],
                s=140,
                edgecolors="k"
            )

        ax.scatter(
            [], [],
            c="gold",
            edgecolors="k",
            label=f"Класс {c1:g}"
        )

        ax.scatter(
            [], [],
            c="tab:red",
            edgecolors="k",
            label=f"Класс {c0:g}"
        )

        ax.set_xlabel("A")
        ax.set_ylabel("B")

        ax.set_title(
            f"Выход сети [{c0:g}; {c1:g}] — {name}"
        )

        ax.legend(loc="lower right")

        fig.colorbar(
            cs,
            ax=ax,
            label="Выход в исходной шкале"
        )

    plt.tight_layout()
    plt.savefig("3_surface.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(6, 5))

    means = [
        np.mean([r["mae"] for r in res[name]])
        for name in CONFIGS
    ]

    stds = [
        np.std([r["mae"] for r in res[name]])
        for name in CONFIGS
    ]

    bars = plt.bar(
        list(CONFIGS),
        means,
        yerr=stds,
        capsize=5,
        color=[colors[name] for name in CONFIGS]
    )

    for bar, value in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom"
        )

    plt.ylabel("MAE в шкале [c0; c1]")
    plt.title("MAE: среднее ± стандартное отклонение")
    plt.tight_layout()
    plt.savefig("4_mae.png", dpi=150)
    plt.show()
    plt.close()

    tests = (
        [tuple(x) for x in X]
        + [(0, 0), (-5, 7), (9.5, -3)]
    )

    for name in CONFIGS:
        m = models[name][rep[name]]

        print(f"\nФункционирование: {name}")

        for a, b in tests:
            infer(m, a, b, name)

    if sys.stdin.isatty():
        m = models["B (ReLU)"][rep["B (ReLU)"]]

        while True:
            raw = input(
                "Введите A B из [-10; 10] "
                "(пустая строка — выход): "
            ).strip()

            if not raw:
                break

            try:
                a, b = map(float, raw.split())

                if not (-10 <= a <= 10 and -10 <= b <= 10):
                    raise ValueError

            except ValueError:
                print(
                    "Ошибка: введите два числа из [-10; 10]."
                )
                continue

            infer(m, a, b, "ReLU")