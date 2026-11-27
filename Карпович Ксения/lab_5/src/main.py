import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURNAME, INITIALS, VARIANT = "Карпович", "К.А.", 6
LETTERS = ["К", "А", "Р"]
H_VAR, ETA, M_MAX = 3, 0.2, 10
EE, MAX_EPOCHS, TAU = 0.05, 5000, 0.6
N_NOISE, SEEDS, BASE_SEED = 50, [1, 2, 3, 4, 5], 1
ROWS, COLS = 7, 5
N_IN, N_OUT = ROWS * COLS, 3
HERE = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(HERE, "plots")


# ---------- данные ----------
def read_dataset(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("//")]
    X, labels, inst, names = [], [], [], []
    for i in range(0, len(lines), ROWS + 1):
        head, rows = lines[i], lines[i + 1:i + 1 + ROWS]
        if not head.startswith("@") or len(rows) != ROWS \
                or any(len(r) != COLS or set(r) - set("#.") for r in rows):
            raise ValueError(f"Ошибка формата в объекте {head!r}")
        parts = head[1:].split()
        X.append(grid_to_vector(rows))
        labels.append(int(parts[0]) - 1)
        inst.append(int(parts[1]))
        names.append(parts[2] if len(parts) > 2 else "?")
    return np.array(X), np.array(labels), np.array(inst), names


def grid_to_vector(rows):
    return np.array([1.0 if ch == "#" else 0.0 for r in rows for ch in r])


def vector_to_grid(x):
    return ["".join("#" if x[r * COLS + c] > 0.5 else "." for c in range(COLS)) for r in range(ROWS)]


def one_hot(labels, n=N_OUT):
    Y = np.zeros((len(labels), n))
    for k, lab in enumerate(labels):
        Y[k, lab] = 1.0
    return Y


def split_by_instance(X, y, inst):
    tr = inst <= 7
    return X[tr], y[tr], inst[tr], X[~tr], y[~tr], inst[~tr]


def hamming(a, b):
    return int(np.sum(a != b))


def dataset_quality(X, y):
    intra, inter, per_class, pairs = [], [], {c: [] for c in range(N_OUT)}, {}
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = hamming(X[i], X[j])
            if y[i] == y[j]:
                intra.append(d)
                per_class[y[i]].append(d)
            else:
                inter.append(d)
                pairs.setdefault(tuple(sorted((y[i], y[j]))), []).append(d)
    return {"n_fill": int(X.sum()), "intra": np.mean(intra), "inter": np.mean(inter),
            "per_class": {c: np.mean(v) for c, v in per_class.items()},
            "pairs": {k: np.mean(v) for k, v in pairs.items()}}


# ---------- сеть ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def softmax(z):
    e = np.exp(z - np.max(z))
    return e / np.sum(e)


def init_params(H, seed):
    rng = np.random.default_rng(seed)
    return {"W1": rng.uniform(-0.5, 0.5, (H, N_IN)), "b1": rng.uniform(-0.5, 0.5, H),
            "W2": rng.uniform(-0.5, 0.5, (N_OUT, H)), "b2": rng.uniform(-0.5, 0.5, N_OUT)}


def n_params(H):
    return H * N_IN + H + N_OUT * H + N_OUT


def forward(P, x):
    h = sigmoid(P["W1"] @ x + P["b1"])
    return h, softmax(P["W2"] @ h + P["b2"])


def backward(P, x, y, h, p):
    d2 = p - y  # градиент CE по входу softmax
    d1 = (P["W2"].T @ d2) * h * (1.0 - h)
    return {"W2": np.outer(d2, h), "b2": d2, "W1": np.outer(d1, x), "b1": d1}


def sgd_step(P, grads, eta):
    for k in P:
        P[k] -= eta * grads[k]


def cross_entropy(y, p):
    return -float(np.sum(y * np.log(p + 1e-12)))


def evaluate_losses(P, X, Y):
    ps = predict_proba(P, X)
    es = np.mean([0.5 * np.sum((y - p) ** 2) for y, p in zip(Y, ps)])
    ce = np.mean([cross_entropy(y, p) for y, p in zip(Y, ps)])
    return es, ce


def train(X, Y, H, eta=ETA, seed=BASE_SEED):
    P = init_params(H, seed)
    rng = np.random.default_rng(seed + 1000)
    hist = {"es": [], "ce": [], "converged": False}
    for epoch in range(1, MAX_EPOCHS + 1):
        for idx in rng.permutation(len(X)):
            h, p = forward(P, X[idx])
            sgd_step(P, backward(P, X[idx], Y[idx], h, p), eta)
        es, ce = evaluate_losses(P, X, Y)
        hist["es"].append(es)
        hist["ce"].append(ce)
        if es <= EE:
            hist["converged"] = True
            break
    hist["epochs"] = epoch
    return P, hist


# ---------- метрики, шум, градиенты ----------
def predict_proba(P, X):
    return np.array([forward(P, x)[1] for x in X])


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    M = np.zeros((N_OUT, N_OUT), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M


def format_cm(M):
    s = "      " + "  ".join(f"{L:>3}" for L in LETTERS) + "  <- ответ сети\n"
    return s + "".join(f"  {LETTERS[i]}   " + "  ".join(f"{v:>3}" for v in M[i]) + "\n" for i in range(N_OUT))


def add_noise(x, m, rng):
    x2 = x.copy()
    if m > 0:
        idx = rng.choice(N_IN, size=m, replace=False)
        x2[idx] = 1.0 - x2[idx]
    return x2


def noise_test(P, X, y, seed=123):
    rng = np.random.default_rng(seed)
    total, per_cls = np.zeros(M_MAX + 1), np.zeros((M_MAX + 1, N_OUT))
    for m in range(M_MAX + 1):
        hits = np.array([np.mean([np.argmax(forward(P, add_noise(x, m, rng))[1]) == y[k]
                                  for _ in range(N_NOISE)]) for k, x in enumerate(X)])
        total[m] = hits.mean()
        per_cls[m] = [hits[y == c].mean() for c in range(N_OUT)]
    return total, per_cls


def numerical_gradient(P, x, y, eps=1e-5):
    num = {}
    for k in P:
        g = np.zeros_like(P[k])
        for i in np.ndindex(P[k].shape):
            old = P[k][i]
            P[k][i] = old + eps
            lp = cross_entropy(y, forward(P, x)[1])
            P[k][i] = old - eps
            lm = cross_entropy(y, forward(P, x)[1])
            P[k][i] = old
            g[i] = (lp - lm) / (2 * eps)
        num[k] = g
    return num


def gradient_check(P, x, y):
    an = backward(P, x, y, *forward(P, x))
    nu = numerical_gradient(P, x, y)
    return {k: np.linalg.norm(an[k] - nu[k]) / (np.linalg.norm(an[k]) + np.linalg.norm(nu[k]))
            for k in ["W1", "b1", "W2", "b2"]}


# ---------- порог отказа и режим функционирования ----------
def decide(p, tau=TAU):
    k = int(np.argmax(p))
    return (k if p[k] >= tau else None), float(p[k])


def reject_stats(P, X, y=None):
    rejected = right = 0
    for k, x in enumerate(X):
        p = forward(P, x)[1]
        if decide(p)[0] is None:
            rejected += 1
            right += int(y is not None and np.argmax(p) == y[k])
    return rejected, right


def functioning(P, rows, title=""):
    p = forward(P, grid_to_vector(rows))[1]
    cls, conf = decide(p)
    print(f"--- {title} ---")
    print("\n".join("   " + r for r in rows))
    print("   " + ", ".join(f"p{i + 1}({LETTERS[i]})={p[i]:.4f}" for i in range(N_OUT)))
    print(f"   Класс: {LETTERS[int(np.argmax(p))]}, уверенность {conf:.4f} -> "
          + (f"НЕ РАСПОЗНАНО (< {TAU})" if cls is None else f"буква «{LETTERS[cls]}»") + "\n")


def interactive_mode(P):
    print("\nВведите 7 строк по 5 символов '#'/'.' (пустая строка - выход)")
    while True:
        rows = []
        while len(rows) < ROWS:
            s = input(f"  строка {len(rows) + 1}: ").strip()
            if not s:
                return
            if len(s) == COLS and not set(s) - set("#."):
                rows.append(s)
            else:
                print("  нужно ровно 5 символов '#' или '.'")
        functioning(P, rows, "Введённая сетка")


# ---------- графики ----------
def title(text, n_fill):
    return f"{text}\n{SURNAME} {INITIALS}, вариант {VARIANT}, Nfill = {n_fill}"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_grid(ax, x, label):
    ax.imshow(x.reshape(ROWS, COLS), cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="lightgray", lw=0.6)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(label, fontsize=8)


def plot_dataset(X, y, inst, n_fill):
    fig, axes = plt.subplots(N_OUT, 10, figsize=(12, 5.4))
    for c in range(N_OUT):
        for n in range(1, 11):
            k = np.where((y == c) & (inst == n))[0][0]
            draw_grid(axes[c, n - 1], X[k], f"{LETTERS[c]}{n}" + (" (тест)" if n >= 8 else ""))
    fig.suptitle(title("Датасет: 30 сеток 7×5", n_fill))
    save(fig, "00_dataset.png")


def plot_convergence(hists, n_fill):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (H, h), c in zip(hists.items(), ["tab:blue", "tab:orange", "tab:green"]):
        ep = np.arange(1, len(h["es"]) + 1)
        ax.plot(ep, h["es"], color=c, lw=2, label=f"Es, H={H}")
        ax.plot(ep, h["ce"], color=c, lw=1.5, ls="--", label=f"CE, H={H}")
    ax.axhline(EE, color="red", ls=":", label=f"Ee = {EE}")
    ax.set(yscale="log", xlabel="Эпоха", ylabel="Ошибка (лог. шкала)",
           title=title("Сходимость: Es и CE от эпохи", n_fill))
    ax.grid(alpha=0.3, which="both")
    ax.legend(ncol=2)
    save(fig, "01_convergence.png")


def plot_epochs_scatter(study, n_fill):
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (H, runs) in enumerate(study.items()):
        ep = [r["epochs"] for r in runs]
        ax.scatter(i + np.linspace(-0.12, 0.12, len(ep)), ep, s=60, zorder=3, label=f"H={H}")
        ax.hlines(np.mean(ep), i - 0.25, i + 0.25, colors="k", lw=2)
    ax.set_xticks(range(len(study)), [f"H = {H}" for H in study])
    ax.set(ylabel="Число эпох до Es ≤ 0.05", title=title("Разброс числа эпох по 5 запускам", n_fill))
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "02_epochs_scatter.png")


def plot_confusion(M, n_fill, name, text):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(M, cmap="Blues", vmin=0)
    for i in range(N_OUT):
        for j in range(N_OUT):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=16,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    ax.set_xticks(range(N_OUT), LETTERS)
    ax.set_yticks(range(N_OUT), LETTERS)
    ax.set(xlabel="Ответ сети", ylabel="Истинный класс")
    ax.set_title(title(text, n_fill), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, name)


def plot_noise(total, per_cls, n_fill):
    m = np.arange(M_MAX + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m, total * 100, "k-o", lw=2.5, label="Общая")
    for c, mk in zip(range(N_OUT), "s^D"):
        ax.plot(m, per_cls[:, c] * 100, marker=mk, lw=1.5, label=f"Буква {LETTERS[c]}")
    ax.axhline(80, color="red", ls="--", label="80 %")
    ax.set(xlabel="Число инвертированных клеток m", ylabel="Accuracy на тесте, %", ylim=(0, 105),
           xticks=m, title=title(f"Устойчивость к шуму ({N_NOISE} зашумлений на пример)", n_fill))
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "05_noise.png")


def plot_weights(P, n_fill):
    H, vmax = P["W1"].shape[0], np.max(np.abs(P["W1"]))
    fig, axes = plt.subplots(1, H, figsize=(3.2 * H + 1, 4.6))
    for j, ax in enumerate(np.atleast_1d(axes)):
        w = P["W1"][j].reshape(ROWS, COLS)
        im = ax.imshow(w, cmap="bwr", vmin=-vmax, vmax=vmax)
        for r, c in np.ndindex(w.shape):
            ax.text(c, r, f"{w[r, c]:.1f}", ha="center", va="center", fontsize=7)
        v = P["W2"][:, j]
        ax.set_title(f"Нейрон {j + 1}\nv = [" + ", ".join(f"{LETTERS[k]}:{v[k]:+.2f}" for k in range(N_OUT))
                     + "]", fontsize=9)
        ax.set_xticks(range(COLS), range(1, COLS + 1))
        ax.set_yticks(range(ROWS), range(1, ROWS + 1))
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), fraction=0.03)
    fig.suptitle(title("Карты весов скрытых нейронов", n_fill), y=1.08)
    fig.savefig(os.path.join(PLOTS, "06_weights.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_uncertain(items, n_fill):
    fig, axes = plt.subplots(1, len(items), figsize=(3.4 * len(items), 4.6))
    for ax, it in zip(axes, items):
        draw_grid(ax, it["x"], f"Истина: {LETTERS[it['y']]}, экз. №{it['inst']}\n"
                  + "\n".join(f"p({LETTERS[k]}) = {it['p'][k]:.4f}" for k in range(N_OUT)))
    fig.suptitle(title("Три самых неуверенных тестовых примера", n_fill))
    save(fig, "07_uncertain.png")


# ---------- главная программа ----------
def main():
    os.makedirs(PLOTS, exist_ok=True)
    X, y, inst, _ = read_dataset(os.path.join(HERE, "dataset.txt"))
    Xtr, ytr, itr, Xte, yte, ite = split_by_instance(X, y, inst)
    Ytr = one_hot(ytr)
    q = dataset_quality(X, y)
    n_fill = q["n_fill"]
    print(f"{SURNAME} {INITIALS}, вариант {VARIANT}: классы {', '.join(LETTERS)}; "
          f"H = {H_VAR}, eta = {ETA}, m_max = {M_MAX}")
    print(f"Объектов: {len(X)} (обучение {len(Xtr)}, тест {len(Xte)})")

    print("\n=== Контроль качества датасета ===")
    print(f"N_fill = {n_fill}")
    print(f"Хэмминг внутри классов = {q['intra']:.2f} ({'OK' if q['intra'] >= 3 else '< 3!'}), "
          f"между классами = {q['inter']:.2f}")
    print("  внутри: " + ", ".join(f"{LETTERS[c]} {q['per_class'][c]:.2f}" for c in range(N_OUT)))
    print("  между:  " + ", ".join(f"{LETTERS[a]}-{LETTERS[b]} {d:.2f}" for (a, b), d in q["pairs"].items()))
    plot_dataset(X, y, inst, n_fill)

    print("\n=== Аудит градиентов (отн. ошибка аналитический vs численный) ===")
    P0 = init_params(H_VAR, BASE_SEED)
    print("  " + ", ".join(f"{k}: {v:.1e}" for k, v in gradient_check(P0, Xtr[0], Ytr[0]).items()))
    h, p = forward(P0, Xtr[0])
    g = backward(P0, Xtr[0], Ytr[0], h, p)
    print(f"Шаг обучения на К1: h = {np.round(h, 4)}, p = {np.round(p, 4)}")
    print(f"  delta2 = {np.round(p - Ytr[0], 4)}, delta1 = {np.round(g['b1'], 4)}")
    ce0 = cross_entropy(Ytr[0], p)
    sgd_step(P0, g, ETA)
    print(f"  CE до шага = {ce0:.5f}, после шага = {cross_entropy(Ytr[0], forward(P0, Xtr[0])[1]):.5f}")

    print(f"\n=== Базовый эксперимент (H = {H_VAR}, eta = {ETA}) ===")
    P, hist = train(Xtr, Ytr, H_VAR)
    es, ce = evaluate_losses(P, Xtr, Ytr)
    pte = predict_proba(P, Xte)
    yptr, ypte = predict_proba(P, Xtr).argmax(1), pte.argmax(1)
    ok, conf_te = ypte == yte, pte.max(1)
    print(f"Эпох: {hist['epochs']} (критерий {'достигнут' if hist['converged'] else 'НЕ достигнут'}), "
          f"Es = {es:.4f}, CE = {ce:.4f}")
    print(f"Accuracy: обучение = {accuracy(ytr, yptr):.1%}, тест = {accuracy(yte, ypte):.1%}")
    print("Матрица ошибок (обучение):\n" + format_cm(confusion_matrix(ytr, yptr)))
    print("Матрица ошибок (тест):\n" + format_cm(confusion_matrix(yte, ypte)))
    print(f"Уверенность на тесте: верные {conf_te[ok].mean():.4f}, неверные "
          + (f"{conf_te[~ok].mean():.4f}" if (~ok).any() else "-"))
    for k in np.where(~ok)[0]:
        print(f"  ошибка: {LETTERS[yte[k]]}{ite[k]} -> {LETTERS[ypte[k]]}, p = {np.round(pte[k], 4)}")
    plot_confusion(confusion_matrix(yte, ypte), n_fill, "03_confusion_test.png", "Матрица ошибок, тест")
    plot_confusion(confusion_matrix(ytr, yptr), n_fill, "04_confusion_train.png", "Матрица ошибок, обучение")

    print("\n=== Исследование H (5 запусков) ===")
    study, conv = {}, {}
    for H in [2, H_VAR, 2 * H_VAR]:
        study[H] = []
        for s in SEEDS:
            Ps, hs = train(Xtr, Ytr, H, ETA, s)
            study[H].append({"epochs": hs["epochs"], "conv": hs["converged"],
                             "acc": accuracy(yte, predict_proba(Ps, Xte).argmax(1))})
            if s == BASE_SEED:
                conv[H] = hs
        ep, ac = [r["epochs"] for r in study[H]], [r["acc"] * 100 for r in study[H]]
        print(f"  H={H}: эпохи {np.mean(ep):.1f} ({min(ep)}-{max(ep)}), acc {np.mean(ac):.1f}% "
              f"({min(ac):.1f}-{max(ac):.1f}), не сошлось {sum(not r['conv'] for r in study[H])}, "
              f"параметров {n_params(H)}")
    plot_convergence(conv, n_fill)
    plot_epochs_scatter(study, n_fill)

    print(f"\n=== Шумовой тест ({N_NOISE} зашумлений на пример) ===")
    total, per_cls = noise_test(P, Xte, yte)
    print("   m |  общая | " + " | ".join(f"   {L}  " for L in LETTERS))
    for m in range(M_MAX + 1):
        print(f"  {m:2d} | {total[m]:6.1%} | " + " | ".join(f"{v:6.1%}" for v in per_cls[m]))
    below = [m for m in range(M_MAX + 1) if total[m] < 0.8]
    print("Accuracy < 80% впервые при m = " + (str(below[0]) if below else "-"))
    plot_noise(total, per_cls, n_fill)

    print("\n=== Карты весов ===")
    for j in range(H_VAR):
        v, w = P["W2"][:, j], P["W1"][j]
        top = np.argsort(-np.abs(w))[:4]
        print(f"  Нейрон {j + 1}: «за» {LETTERS[np.argmax(v)]}, «против» {LETTERS[np.argmin(v)]}; "
              f"v = {np.round(v, 2)}; |w| max: "
              + ", ".join(f"({i // COLS + 1},{i % COLS + 1}) {w[i]:+.2f}" for i in top))
    plot_weights(P, n_fill)

    print("\n=== Три самых неуверенных тестовых примера ===")
    order = np.argsort(conf_te)[:3]
    for k in order:
        print(f"  {LETTERS[yte[k]]}{ite[k]}: ответ {LETTERS[ypte[k]]}, "
              + ", ".join(f"p({LETTERS[c]})={pte[k, c]:.4f}" for c in range(N_OUT)))
    plot_uncertain([{"x": Xte[k], "y": yte[k], "inst": ite[k], "p": pte[k]} for k in order], n_fill)

    print("\n=== Кросс-тест на чужом почерке ===")
    Xf, yf, iff, _ = read_dataset(os.path.join(HERE, "foreign.txt"))
    pf = predict_proba(P, Xf)
    for k in range(len(Xf)):
        print(f"  {LETTERS[yf[k]]}{iff[k]} -> {LETTERS[pf[k].argmax()]}, p = {np.round(pf[k], 4)}")
    print(f"Accuracy: чужой почерок {accuracy(yf, pf.argmax(1)):.1%}, свой тест {accuracy(yte, ypte):.1%}")

    print("\n=== «Чужая» буква и порог отказа tau = 0.6 ===")
    Xa, _, _, na = read_dataset(os.path.join(HERE, "alien.txt"))
    pa = forward(P, Xa[0])[1]
    print(f"«{na[0]}» -> «{LETTERS[pa.argmax()]}», p = {np.round(pa, 4)}, решение: "
          + ("НЕ РАСПОЗНАНО" if decide(pa)[0] is None else "принято"))
    for name, XX, yy in [("тест", Xte, yte), ("чужой почерк", Xf, yf), ("«чужая» буква", Xa, None)]:
        r, w = reject_stats(P, XX, yy)
        print(f"  {name}: всего {len(XX)}, отвергнуто {r}, из них были бы верными {w}")

    print("\n=== Режим функционирования ===")
    demo = [(f"обучение: {LETTERS[c]}1", Xtr[(ytr == c) & (itr == 1)][0]) for c in range(N_OUT)]
    demo += [(f"тест: {LETTERS[c]}{8 + c}", Xte[(yte == c) & (ite == 8 + c)][0]) for c in range(N_OUT)]
    rng = np.random.default_rng(7)
    for c, m in [(0, 3), (2, 5)]:
        demo.append((f"искажённая {LETTERS[c]}8, изменено клеток: {m}",
                     add_noise(Xte[(yte == c) & (ite == 8)][0], m, rng)))
    demo.append((f"«чужая» буква {na[0]}", Xa[0]))
    demo += [(f"чужой почерк: {LETTERS[yf[k]]}{iff[k]}", Xf[k]) for k in (1, 5)]
    for text, x in demo:
        functioning(P, vector_to_grid(x), text)

    if "--interactive" in sys.argv:
        interactive_mode(P)


if __name__ == "__main__":
    main()
