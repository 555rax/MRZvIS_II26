import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

C0, C1 = -8.0, 3.0
X_raw = np.array([[-8, -8], [-8, 3], [3, -8], [3, 3]], dtype=float)
Y_raw = np.array([-8.0, 3.0, 3.0, -8.0])


def normalize(t):
    return (t - C0) / (C1 - C0)


def denormalize(o):
    return C0 + (C1 - C0) * o


X = normalize(X_raw)
Y = normalize(Y_raw)

LR = {"sigmoid": 5.0, "relu": 0.2}
MAX_EPOCHS = 5000
EE = 0.05
SEEDS = [0, 1, 2, 3, 4]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLP:

    def __init__(self, act="sigmoid", seed=None):
        rng = np.random.RandomState(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.W2 = rng.uniform(-0.5, 0.5, 2)
        self.b2 = rng.uniform(-0.5, 0.5)
        self.act = act
        self.lr = LR[act]

    def _f(self, z):
        return sigmoid(z) if self.act == "sigmoid" else np.maximum(0.0, z)

    def _df(self, z, h):
        return h * (1 - h) if self.act == "sigmoid" else (z > 0).astype(float)

    def forward(self, x):
        z1 = self.W1 @ x + self.b1
        h = self._f(z1)
        o = sigmoid(self.W2 @ h + self.b2)
        return z1, h, o

    def train_epoch(self, X, Y):
        total = 0.0
        eps = 1e-9
        for x, t in zip(X, Y):
            z1, h, o = self.forward(x)
            oc = np.clip(o, eps, 1 - eps)
            total += -(t * np.log(oc) + (1 - t) * np.log(1 - oc))
            d_out = t - o
            d_hid = d_out * self.W2 * self._df(z1, h)
            self.W2 += self.lr * d_out * h
            self.b2 += self.lr * d_out
            self.W1 += self.lr * np.outer(d_hid, x)
            self.b1 += self.lr * d_hid
        return total

    def predict(self, x):
        return self.forward(x)[2]

    def dead_neurons(self):
        dead = np.ones(2, dtype=bool)
        for x in X:
            dead &= (self.forward(x)[0] <= 0)
        return int(dead.sum())


def train_network(act, seed):
    net = MLP(act, seed)
    history, conv = [], None
    for ep in range(1, MAX_EPOCHS + 1):
        Es = net.train_epoch(X, Y)
        history.append(Es)
        if Es <= EE:
            conv = ep
            break
    return net, conv, history


def evaluate(net):
    correct, errs = 0, []
    for i in range(len(X)):
        y = denormalize(net.predict(X[i]))
        cls = C0 if abs(y - C0) < abs(y - C1) else C1
        correct += (cls == Y_raw[i])
        errs.append(abs(y - Y_raw[i]))
    return correct / len(X), float(np.mean(errs))


def run_series(act):
    res = []
    for s in SEEDS:
        net, conv, hist = train_network(act, s)
        acc, mae = evaluate(net)
        res.append(dict(seed=s, converged=conv is not None,
                        epochs=conv if conv else len(hist),
                        Es=hist[-1], acc=acc, mae=mae,
                        dead=net.dead_neurons() if act == "relu" else 0,
                        net=net, history=hist))
    return res


NAMES = {"sigmoid": "Конфигурация А (Сигмоида)", "relu": "Конфигурация В (ReLU)"}
results = {a: run_series(a) for a in ("sigmoid", "relu")}

for a, res in results.items():
    print(f"\n=== {NAMES[a]} ===")
    for r in res:
        st = f"{r['epochs']} эпох" if r["converged"] else f"НЕ сошёлся за {r['epochs']}"
        extra = f", мёртвых нейронов: {r['dead']}" if a == "relu" else ""
        print(f"seed={r['seed']}: {st}, Es={r['Es']:.5f}, "
              f"accuracy={r['acc']*100:.0f}%, MAE={r['mae']:.3f}{extra}")


def summarize(res):
    conv = [r for r in res if r["converged"]]
    base = conv if conv else res
    ep = [r["epochs"] for r in conv]
    return dict(ep_mean=np.mean([r["epochs"] for r in base]),
                ep_std=np.std(ep) if ep else float("nan"),
                Es=np.mean([r["Es"] for r in base]),
                acc=np.mean([r["acc"] for r in base]) * 100,
                mae=np.mean([r["mae"] for r in base]),
                n=sum(r["converged"] for r in res))


S = {a: summarize(results[a]) for a in results}
print("\n=== Сводная таблица (средние по сошедшимся запускам) ===")
print(f"{'Метрика':<22}{'Сигмоида (А)':<16}{'ReLU (В)':<16}")
print(f"{'Эпох (среднее)':<22}{S['sigmoid']['ep_mean']:<16.1f}{S['relu']['ep_mean']:<16.1f}")
print(f"{'Эпох (СКО)':<22}{S['sigmoid']['ep_std']:<16.1f}{S['relu']['ep_std']:<16.1f}")
print(f"{'Es итог':<22}{S['sigmoid']['Es']:<16.5f}{S['relu']['Es']:<16.5f}")
print(f"{'Accuracy, %':<22}{S['sigmoid']['acc']:<16.1f}{S['relu']['acc']:<16.1f}")
print(f"{'MAE (шкала [c0;c1])':<22}{S['sigmoid']['mae']:<16.3f}{S['relu']['mae']:<16.3f}")
print(f"{'Сошлось из 5':<22}{S['sigmoid']['n']:<16}{S['relu']['n']:<16}")
print(f"{'Запусков с мёртв. нейр.':<22}{'-':<16}{sum(r['dead'] > 0 for r in results['relu']):<16}")


def representative(res):
    for r in res:
        if r["converged"]:
            return r
    return res[0]


rep = {a: representative(results[a]) for a in results}
col = {"sigmoid": "tab:blue", "relu": "tab:green"}

plt.figure(figsize=(8, 5))
for a in rep:
    plt.plot(rep[a]["history"], label=NAMES[a], color=col[a])
plt.axhline(EE, color="gray", linestyle="--", label=f"Ee = {EE}")
plt.yscale("log")
plt.xlabel("Эпоха")
plt.ylabel("Суммарная ошибка Es (лог. шкала)")
plt.title("Сходимость: Сигмоида vs ReLU (вариант 8)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_convergence.png"), dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
w = 0.35
idx = np.arange(len(SEEDS))
for k, a in enumerate(("sigmoid", "relu")):
    bars = ax.bar(idx + (k - 0.5) * w, [r["epochs"] for r in results[a]], w,
                  label=NAMES[a], color=col[a])
    for b, r in zip(bars, results[a]):
        if not r["converged"]:
            b.set_hatch("//")
            b.set_edgecolor("black")
ax.set_xticks(idx)
ax.set_xticklabels([f"seed={s}" for s in SEEDS])
ax.set_ylabel("Эпох до сходимости (или MAX_EPOCHS)")
ax.set_title("Устойчивость по 5 запускам (штриховка = не сошёлся)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_epochs_spread.png"), dpi=150)
plt.close()

g = np.linspace(-10, 10, 200)
AA, BB = np.meshgrid(g, g)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, a in zip(axes, ("sigmoid", "relu")):
    net = rep[a]["net"]
    Z = np.array([[net.predict(normalize(np.array([AA[i, j], BB[i, j]])))
                   for j in range(200)] for i in range(200)])
    cs = ax.contourf(AA, BB, Z, levels=20, cmap="coolwarm", vmin=0, vmax=1)
    ax.contour(AA, BB, Z, levels=[0.5], colors="k", linewidths=1.5)
    ax.scatter(X_raw[:, 0], X_raw[:, 1], c=Y, cmap="coolwarm", vmin=0, vmax=1,
               edgecolors="black", s=120, linewidths=1.5, zorder=5)
    ax.set_title(NAMES[a])
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    fig.colorbar(cs, ax=ax, label="ŷ (норм.)")
plt.suptitle("Выход сети на плоскости (A, B), чёрная линия — уровень 0.5")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_decision_surface.png"), dpi=150)
plt.close()


fig, ax = plt.subplots(figsize=(6, 5))
vals = [S["sigmoid"]["mae"], S["relu"]["mae"]]
bars = ax.bar(["Сигмоида (А)", "ReLU (В)"], vals, color=[col["sigmoid"], col["relu"]])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom")
ax.set_ylabel("MAE (шкала [c0; c1])")
ax.set_title("Точность восстановления исходной шкалы")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_mae_comparison.png"), dpi=150)
plt.close()
print(f"\nГрафики сохранены в: {OUTPUT_DIR}")

def run_inference(net, A, B):
    o = net.predict(normalize(np.array([A, B], dtype=float)))
    y = denormalize(o)
    return o, y, (C0 if abs(y - C0) < abs(y - C1) else C1)


def demo(net, title):
    print(f"\n=== Режим функционирования: {title} ===")
    print(f"{'A':>6}{'B':>6}   {'ŷ_норм':>8}   {'ŷ_реал':>8}   класс")
    for A, B in list(map(tuple, X_raw)) + [(-5, 0), (1, -2), (10, -10)]:
        o, y, c = run_inference(net, A, B)
        print(f"{A:>6.0f}{B:>6.0f}   {o:>8.4f}   {y:>8.3f}   {c:.0f}")


demo(rep["sigmoid"]["net"], NAMES["sigmoid"])
demo(rep["relu"]["net"], NAMES["relu"])

if __name__ == "__main__":
    print("\n=== Интерактивный режим (модель ReLU) ===")
    print("Введите A и B из [-10;10] ('q' для выхода).")
    while True:
        raw = input("A = ")
        if raw.strip().lower() == "q":
            break
        try:
            A = float(raw)
            B = float(input("B = "))
        except ValueError:
            print("Ошибка: введите числа.")
            continue
        if not (-10 <= A <= 10 and -10 <= B <= 10):
            print("Ошибка: значения должны быть в [-10;10].")
            continue
        o, y, c = run_inference(rep["relu"]["net"], A, B)
        print(f"ŷ_норм = {o:.4f}   ŷ_реал = {y:.3f}   класс = {c:.0f} "
              f"({'c0' if c == C0 else 'c1'})")