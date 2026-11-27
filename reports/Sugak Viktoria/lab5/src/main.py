import os
import re
import numpy as np
import matplotlib.pyplot as plt

SURNAME = "Сугак"
VARIANT = 4
CLASSES = ["С", "У", "Г"]
H_BASE = 6
H_LIST = [2, H_BASE, 2 * H_BASE]
ETA = 0.2
M_MAX = 6
EE = 0.05
EPOCH_LIMIT = 5000
TAU = 0.6
SEEDS = [1, 2, 3, 4, 5]
NOISE_REPEATS = 50
R, C = 7, 5
NIN = R * C
NCLS = 3
FOLDER = os.path.dirname(os.path.abspath(__file__))
F_DATA = os.path.join(FOLDER, "dataset.txt")
F_OTHER = os.path.join(FOLDER, "other.txt")
F_ALIEN = os.path.join(FOLDER, "alien.txt")


def make_item(head, grid):
    if len(grid) != R or any(len(g) != C for g in grid):
        raise ValueError(f"Неверная сетка в объекте «{head}»: нужно {R} строк по {C} символов")
    nums = [int(n) for n in re.findall(r"\d+", head)]
    vec = np.array([1.0 if ch == "#" else 0.0 for g in grid for ch in g])
    return {"head": head, "cls": nums[0], "inst": nums[1] if len(nums) > 1 else 0, "grid": grid, "x": vec}


def parse_file(path):
    items, head, grid = [], None, []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if set(s) <= set("#."):
                grid.append(s)
            else:
                if head is not None:
                    items.append(make_item(head, grid))
                head, grid = s, []
    if head is not None:
        items.append(make_item(head, grid))
    return items


def parse_optional(path):
    return parse_file(path) if os.path.exists(path) else None


def grid_of(x):
    return ["".join("#" if x[r * C + c] > 0.5 else "." for c in range(C)) for r in range(R)]


def show_grid(x, pad="  "):
    for row in grid_of(x):
        print(pad + row)


def cell_name(i):
    return f"({i // C + 1},{i % C + 1})"


def split(items):
    tr = [o for o in items if o["inst"] <= 7]
    te = [o for o in items if o["inst"] >= 8]
    return tr, te


def make_xy(items):
    X = np.array([o["x"] for o in items])
    labels = np.array([o["cls"] - 1 for o in items])
    Y = np.zeros((len(items), NCLS))
    Y[np.arange(len(items)), labels] = 1.0
    return X, Y, labels


def hamming(a, b):
    return int(np.sum(a != b))


def check_quality(items):
    nfill = int(sum(o["x"].sum() for o in items))
    by = {k: [o["x"] for o in items if o["cls"] == k] for k in range(1, NCLS + 1)}
    print(f"Nfill = {nfill}")
    inner_all = []
    for k, vs in by.items():
        d = [hamming(vs[i], vs[j]) for i in range(len(vs)) for j in range(i + 1, len(vs))]
        inner_all += d
        warn = "  <-- меньше 3, перепишите экземпляры" if np.mean(d) < 3 else ""
        print(f"Внутри класса {CLASSES[k - 1]}: среднее расстояние Хэмминга {np.mean(d):.2f}{warn}")
    outer_all = []
    for a in range(1, NCLS + 1):
        for b in range(a + 1, NCLS + 1):
            d = [hamming(u, v) for u in by[a] for v in by[b]]
            outer_all += d
            print(f"Между {CLASSES[a - 1]} и {CLASSES[b - 1]}: {np.mean(d):.2f}")
    print(f"Среднее внутриклассовое: {np.mean(inner_all):.2f}, межклассовое: {np.mean(outer_all):.2f}")
    return nfill


def sigm(z):
    return 1.0 / (1.0 + np.exp(-z))


def soft(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class Net:
    def __init__(self, hidden, seed):
        self.H = hidden
        self.rng = np.random.default_rng(seed)
        self.p = {
            "W1": self.rng.uniform(-0.3, 0.3, (hidden, NIN)),
            "b1": self.rng.uniform(-0.3, 0.3, hidden),
            "W2": self.rng.uniform(-0.3, 0.3, (NCLS, hidden)),
            "b2": self.rng.uniform(-0.3, 0.3, NCLS),
        }

    def count(self):
        return sum(v.size for v in self.p.values())

    def forward(self, X):
        X = np.atleast_2d(X)
        h = sigm(X @ self.p["W1"].T + self.p["b1"])
        out = soft(h @ self.p["W2"].T + self.p["b2"])
        return h, out

    def proba(self, X):
        return self.forward(X)[1]

    def ce_one(self, x, y):
        return float(-np.sum(y * np.log(self.proba(x)[0] + 1e-12)))

    def grads(self, x, y):
        h, out = self.forward(x)
        h, out = h[0], out[0]
        g2 = out - y
        g1 = (self.p["W2"].T @ g2) * h * (1 - h)
        return {"W1": np.outer(g1, x), "b1": g1, "W2": np.outer(g2, h), "b2": g2}

    def num_grads(self, x, y, eps=1e-5):
        res = {}
        for k, arr in self.p.items():
            g = np.zeros_like(arr)
            for ix in np.ndindex(arr.shape):
                keep = arr[ix]
                arr[ix] = keep + eps
                up = self.ce_one(x, y)
                arr[ix] = keep - eps
                down = self.ce_one(x, y)
                arr[ix] = keep
                g[ix] = (up - down) / (2 * eps)
            res[k] = g
        return res

    def update(self, x, y):
        g = self.grads(x, y)
        for k in self.p:
            self.p[k] -= ETA * g[k]

    def errors(self, X, Y):
        P = self.proba(X)
        es = float(np.mean(np.sum((Y - P) ** 2, axis=1)))
        ce = float(np.mean(-np.sum(Y * np.log(P + 1e-12), axis=1)))
        return es, ce


def fit(net, X, Y):
    hes, hce = [], []
    for ep in range(1, EPOCH_LIMIT + 1):
        for i in net.rng.permutation(len(X)):
            net.update(X[i], Y[i])
        es, ce = net.errors(X, Y)
        hes.append(es)
        hce.append(ce)
        if es <= EE:
            return {"epochs": ep, "ok": True, "es": es, "ce": ce, "hes": hes, "hce": hce}
    return {"epochs": EPOCH_LIMIT, "ok": False, "es": hes[-1], "ce": hce[-1], "hes": hes, "hce": hce}


def grad_audit(net, X, Y, n=3):
    worst = 0.0
    for i in range(n):
        a = net.grads(X[i], Y[i])
        b = net.num_grads(X[i], Y[i])
        for k in a:
            worst = max(worst, float(np.max(np.abs(a[k] - b[k]))))
    return worst


def acc_of(true, pred):
    return float(np.mean(true == pred))


def conf_mat(true, pred):
    m = np.zeros((NCLS, NCLS), dtype=int)
    for t, p in zip(true, pred):
        m[t, p] += 1
    return m


def print_mat(m, title):
    print(title)
    print("истина\\прогноз " + "".join(f"{c:>5}" for c in CLASSES))
    for i in range(NCLS):
        print(f"{CLASSES[i]:<14} " + "".join(f"{m[i, j]:>5}" for j in range(NCLS)))


def flip_cells(x, m, rng):
    z = x.copy()
    pos = rng.choice(NIN, size=m, replace=False) if m > 0 else np.array([], dtype=int)
    z[pos] = 1.0 - z[pos]
    return z, sorted(int(q) for q in pos)


def noise_curve(net, X, labels):
    rng = np.random.default_rng(2024)
    total, per_cls = [], []
    for m in range(M_MAX + 1):
        hit = np.zeros(len(X))
        for i, x in enumerate(X):
            Z = np.array([flip_cells(x, m, rng)[0] for _ in range(NOISE_REPEATS)])
            hit[i] = np.mean(net.proba(Z).argmax(axis=1) == labels[i])
        total.append(hit.mean())
        per_cls.append([hit[labels == k].mean() for k in range(NCLS)])
    return np.array(total), np.array(per_cls)


def report_noise(total, per_cls):
    for m in range(M_MAX + 1):
        row = " | ".join(f"{CLASSES[k]} {per_cls[m][k] * 100:5.1f}%" for k in range(NCLS))
        print(f"m={m}: общая {total[m] * 100:5.1f}% | {row}")
    below = [m for m in range(M_MAX + 1) if total[m] < 0.8]
    if below:
        m0 = below[0]
        print(f"Accuracy ниже 80% впервые при m = {m0}")
        print(f"Самая хрупкая буква при этом m: {CLASSES[int(np.argmin(per_cls[m0]))]}")
    else:
        print(f"До m = {M_MAX} accuracy не падает ниже 80%")
        print(f"Самая хрупкая буква при m = {M_MAX}: {CLASSES[int(np.argmin(per_cls[-1]))]}")


def describe_neurons(net):
    W1, W2 = net.p["W1"], net.p["W2"]
    for j in range(net.H):
        top = np.argsort(-np.abs(W1[j]))[:4]
        v = W2[:, j]
        cells = ", ".join(f"{cell_name(i)} {W1[j, i]:+.2f}" for i in top)
        pos = [CLASSES[k] for k in range(NCLS) if v[k] > 0] or ["-"]
        neg = [CLASSES[k] for k in range(NCLS) if v[k] < 0] or ["-"]
        vs = " ".join(f"v({CLASSES[k]})={v[k]:+.2f}" for k in range(NCLS))
        print(f"Нейрон {j + 1}: {vs}")
        print(f"   за: {','.join(pos)}; против: {','.join(neg)}; главная буква: {CLASSES[int(np.argmax(v))]}")
        print(f"   наибольшие |w|: {cells}")


def verdict(p):
    return "не распознано" if p.max() < TAU else CLASSES[int(np.argmax(p))]


def reject_row(P, labels):
    conf = P.max(axis=1)
    rej = conf < TAU
    if labels is None:
        return len(P), int(rej.sum()), None, None
    right = P.argmax(axis=1) == labels
    return len(P), int(rej.sum()), int((rej & right).sum()), int(right.sum())


def print_reject(rows):
    print(f"{'набор':<18}{'всего':>7}{'отвергнуто':>12}{'из них верных':>15}{'верных всего':>14}")
    for name, (n, rj, rr, tot) in rows:
        print(f"{name:<18}{n:>7}{rj:>12}{('-' if rr is None else rr):>15}{('-' if tot is None else tot):>14}")


def tau_table(Pte, lte, Pf, lf, Pa):
    print(f"{'tau':>5}{'тест: верных отверг.':>24}{'чужой почерк: верных отверг.':>32}{'чужая буква':>14}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        def cnt(P, lab):
            right = P.argmax(axis=1) == lab
            rej = P.max(axis=1) < t
            return f"{int((rej & right).sum())}/{int(right.sum())}"
        a = cnt(Pte, lte)
        b = cnt(Pf, lf) if Pf is not None else "нет данных"
        c = ("отвергнута" if Pa.max() < t else "принята") if Pa is not None else "нет данных"
        print(f"{t:>5}{a:>24}{b:>32}{c:>14}")


def freq_map(items, k):
    return np.mean([o["x"] for o in items if o["cls"] == k], axis=0)


def list_cells(mask):
    ids = np.where(mask)[0]
    return ", ".join(cell_name(i) for i in ids) if len(ids) else "нет"


def analyse_confusions(items, errs):
    if not errs:
        print("Ошибок на тесте и чужом почерке нет")
        return
    cnt = {}
    for e in errs:
        key = tuple(sorted((e["true"], e["pred"])))
        cnt[key] = cnt.get(key, 0) + 1
    (a, b), n = max(cnt.items(), key=lambda kv: kv[1])
    print(f"Чаще всего путаются: {CLASSES[a]} и {CLASSES[b]} (ошибок: {n})")
    fa, fb = freq_map(items, a + 1), freq_map(items, b + 1)
    print(f"Общие клетки (закрашены в половине и более экземпляров обеих букв): {list_cells((fa >= 0.5) & (fb >= 0.5))}")
    for e in errs:
        x = e["x"] > 0.5
        f_true = freq_map(items, e["true"] + 1) >= 0.5
        f_pred = freq_map(items, e["pred"] + 1) >= 0.5
        print(f"{e['src']}: {CLASSES[e['true']]} №{e['inst']} принято за {CLASSES[e['pred']]} (p={e['p']:.4f})")
        print(f"   лишние клетки, типичные для {CLASSES[e['pred']]}: {list_cells(x & f_pred & ~f_true)}")
        print(f"   отсутствуют клетки, типичные для {CLASSES[e['true']]}: {list_cells(~x & f_true)}")


def classify_show(net, x, label, truth=None, note=None):
    p = net.proba(x)[0]
    k = int(np.argmax(p))
    print(f"[{label}]")
    if note:
        print("  " + note)
    show_grid(x)
    print("  " + "  ".join(f"p{j + 1}={p[j]:.4f}" for j in range(NCLS)))
    print(f"  класс: {CLASSES[k]}, уверенность: {p[k]:.4f}, решение при tau={TAU}: {verdict(p)}")
    if truth is not None:
        print(f"  истина: {CLASSES[truth]} -> {'верно' if truth == k else 'ошибка'}")
    print()


def pick(items, cls, inst):
    return next(o for o in items if o["cls"] == cls and o["inst"] == inst)


def demo(net, items, other, alien):
    print("\n" + "=" * 70 + "\nРЕЖИМ ФУНКЦИОНИРОВАНИЯ: ДЕМОНСТРАЦИЯ\n" + "=" * 70)
    for cls, inst in ((1, 2), (2, 5), (3, 7)):
        classify_show(net, pick(items, cls, inst)["x"], f"обучающий {CLASSES[cls - 1]} №{inst}", cls - 1)
    for cls, inst in ((1, 8), (2, 9), (3, 10)):
        classify_show(net, pick(items, cls, inst)["x"], f"тестовый {CLASSES[cls - 1]} №{inst}", cls - 1)
    rng = np.random.default_rng(5)
    for cls, inst, m in ((1, 9, 2), (3, 8, 4)):
        z, pos = flip_cells(pick(items, cls, inst)["x"], m, rng)
        classify_show(net, z, f"искажённый {CLASSES[cls - 1]} №{inst}", cls - 1,
                      note=f"изменено клеток: {m}: " + ", ".join(cell_name(i) for i in pos))
    if alien:
        classify_show(net, alien[0]["x"], "чужая буква")
    if other:
        for o in other[:2]:
            classify_show(net, o["x"], f"чужой почерк «{o['head']}»", o["cls"] - 1)


def read_console_grid():
    rows = []
    while len(rows) < R:
        s = input(f"  строка {len(rows) + 1}/{R}: ").strip()
        if s.lower() == "q":
            return None
        if not re.fullmatch(r"[#.]{5}", s):
            print("  Нужно ровно 5 символов # или .")
            continue
        rows.append(s)
    return np.array([1.0 if ch == "#" else 0.0 for r in rows for ch in r])


def interactive(net):
    print("\nВвод сетки: 7 строк по 5 символов (# и .), q для выхода")
    while True:
        x = read_console_grid()
        if x is None:
            break
        classify_show(net, x, "введённая сетка")


def head_title(text, info):
    return f"{text}\n{info}"


def fig_convergence(runs, info):
    plt.figure(figsize=(8.5, 5.5))
    for H, r in runs.items():
        ln, = plt.plot(r["hes"], label=f"Es, H={H}")
        plt.plot(r["hce"], "--", color=ln.get_color(), label=f"CE, H={H}")
    plt.axhline(EE, color="gray", linestyle=":", label="Ee")
    plt.yscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("Es (сплошная), CE (пунктир)")
    plt.title(head_title("Сходимость для трёх значений H", info))
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "fig_convergence.png"), dpi=150)
    plt.close()


def fig_epochs(series, info):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    w = 0.15
    cols = ["tab:blue", "tab:green", "tab:red"]
    for g, (H, res) in enumerate(series.items()):
        for s, r in enumerate(res):
            xp = g + (s - 2) * w
            ax.bar(xp, r["epochs"], w * 0.9, color=cols[g] if r["ok"] else "none",
                   edgecolor=cols[g], hatch=None if r["ok"] else "//")
            ax.text(xp, r["epochs"], str(r["epochs"]), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([f"H={H}" for H in series])
    ax.set_ylabel("Эпох до остановки")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_title(head_title("Число эпох по 5 запускам (штриховка: критерий не достигнут)", info))
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "fig_epochs.png"), dpi=150)
    plt.close()


def fig_confusion(m, info):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(m, cmap="Blues")
    for i in range(NCLS):
        for j in range(NCLS):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if m[i, j] > m.max() / 2 else "black")
    ax.set_xticks(range(NCLS))
    ax.set_yticks(range(NCLS))
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Предсказано")
    ax.set_ylabel("Истина")
    ax.set_title(head_title("Матрица ошибок, тест", info))
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "fig_confusion.png"), dpi=150)
    plt.close()


def fig_noise(total, per_cls, info):
    ms = np.arange(M_MAX + 1)
    plt.figure(figsize=(8, 5.2))
    plt.plot(ms, total * 100, "k-o", linewidth=2.4, label="общая")
    for k in range(NCLS):
        plt.plot(ms, per_cls[:, k] * 100, "--s", markersize=4, label=f"буква {CLASSES[k]}")
    plt.axhline(80, color="red", linestyle=":", label="80%")
    plt.xlabel("Число инвертированных клеток m")
    plt.ylabel("Средняя accuracy, %")
    plt.ylim(-2, 104)
    plt.title(head_title("Устойчивость к шуму", info))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FOLDER, "fig_noise.png"), dpi=150)
    plt.close()


def fig_weights(net, info):
    W1, W2 = net.p["W1"], net.p["W2"]
    lim = np.abs(W1).max()
    ncol = 3
    nrow = int(np.ceil(net.H / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 4.4 * nrow))
    axes = np.array(axes).reshape(-1)
    im = None
    for j, ax in enumerate(axes):
        if j >= net.H:
            ax.axis("off")
            continue
        w = W1[j].reshape(R, C)
        im = ax.imshow(w, cmap="seismic", vmin=-lim, vmax=lim)
        for r in range(R):
            for c in range(C):
                ax.text(c, r, f"{w[r, c]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(w[r, c]) > 0.55 * lim else "black")
        ax.set_title(f"нейрон {j + 1}, v=" + ",".join(f"{W2[k, j]:+.1f}" for k in range(NCLS)), fontsize=9)
        ax.set_xticks(range(C))
        ax.set_yticks(range(R))
        ax.set_xticklabels(range(1, C + 1))
        ax.set_yticklabels(range(1, R + 1))
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, label="вес w")
    fig.suptitle(head_title("Карты весов скрытых нейронов", info))
    plt.savefig(os.path.join(FOLDER, "fig_weights.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_uncertain(found, info):
    fig, axes = plt.subplots(1, len(found), figsize=(3.6 * len(found), 6))
    for ax, (o, p) in zip(np.atleast_1d(axes), found):
        ax.imshow(o["x"].reshape(R, C), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, C, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, R, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(C))
        ax.set_yticks(range(R))
        ax.set_xticklabels(range(1, C + 1))
        ax.set_yticklabels(range(1, R + 1))
        ax.set_title(f"{CLASSES[o['cls'] - 1]} №{o['inst']}\n" +
                     "\n".join(f"p({CLASSES[k]})={p[k]:.4f}" for k in range(NCLS)), fontsize=9)
    fig.suptitle(head_title("Три самых неуверенных тестовых примера", info), y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(FOLDER, "fig_uncertain.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    items = parse_file(F_DATA)
    if len(items) != 30:
        raise ValueError(f"В dataset.txt найдено {len(items)} объектов, нужно 30")
    other = parse_optional(F_OTHER)
    alien = parse_optional(F_ALIEN)
    tr, te = split(items)
    Xtr, Ytr, ltr = make_xy(tr)
    Xte, Yte, lte = make_xy(te)

    print("=== ДАТАСЕТ ===")
    print(f"Обучающих: {len(tr)}, тестовых: {len(te)}")
    nfill = check_quality(items)
    info = f"{SURNAME}, вариант {VARIANT}, Nfill={nfill}"

    print("\n=== ПРОВЕРКА ГРАДИЕНТОВ ===")
    print(f"Макс. разница аналитического и численного градиента: {grad_audit(Net(H_BASE, 99), Xtr, Ytr):.2e}")

    print(f"\n=== БАЗОВЫЙ ЭКСПЕРИМЕНТ: H={H_BASE}, eta={ETA} ===")
    base = Net(H_BASE, SEEDS[0])
    r = fit(base, Xtr, Ytr)
    print(f"Эпох: {r['epochs']}, критерий достигнут: {r['ok']}, Es={r['es']:.4f}, CE={r['ce']:.4f}")
    Ptr, Pte = base.proba(Xtr), base.proba(Xte)
    ptr, pte = Ptr.argmax(axis=1), Pte.argmax(axis=1)
    acc_tr, acc_te = acc_of(ltr, ptr), acc_of(lte, pte)
    print(f"Accuracy: обучающая {acc_tr * 100:.1f}%, тестовая {acc_te * 100:.1f}%")
    print_mat(conf_mat(ltr, ptr), "Матрица ошибок, обучающая:")
    cm_te = conf_mat(lte, pte)
    print_mat(cm_te, "Матрица ошибок, тестовая:")
    conf = Pte.max(axis=1)
    good = pte == lte
    print("Средняя уверенность на верных тестовых: " + (f"{conf[good].mean():.4f}" if good.any() else "нет"))
    print("Средняя уверенность на неверных тестовых: " + (f"{conf[~good].mean():.4f}" if (~good).any() else "нет ошибок"))

    print(f"\n=== ИССЛЕДОВАНИЕ СКРЫТОГО СЛОЯ: H={H_LIST} ===")
    series = {}
    for H in H_LIST:
        series[H] = []
        for s in SEEDS:
            net = Net(H, s)
            res = fit(net, Xtr, Ytr)
            res["acc"] = acc_of(lte, net.proba(Xte).argmax(axis=1))
            res["params"] = net.count()
            series[H].append(res)
    print(f"{'H':>3}{'параметров':>12}{'эпохи ср.':>11}{'min-max':>14}{'test acc ср.':>14}{'min-max':>16}{'не сошлось':>12}")
    for H, res in series.items():
        ep = [x["epochs"] for x in res]
        ac = [x["acc"] * 100 for x in res]
        bad = sum(not x["ok"] for x in res)
        print(f"{H:>3}{res[0]['params']:>12}{np.mean(ep):>11.0f}{f'{min(ep)}-{max(ep)}':>14}"
              f"{np.mean(ac):>13.1f}%{f'{min(ac):.0f}%-{max(ac):.0f}%':>16}{f'{bad}/{len(res)}':>12}")
    print(f"1 тестовый пример = {100 / len(te):.1f} п. п.")

    print("\n=== УСТОЙЧИВОСТЬ К ШУМУ ===")
    total, per_cls = noise_curve(base, Xte, lte)
    report_noise(total, per_cls)

    print("\n=== КАРТЫ ВЕСОВ ===")
    describe_neurons(base)

    print("\n=== ТРИ САМЫХ НЕУВЕРЕННЫХ ТЕСТОВЫХ ПРИМЕРА ===")
    order = np.argsort(Pte.max(axis=1))[:3]
    found = [(te[i], Pte[i]) for i in order]
    for o, p in found:
        print(f"{CLASSES[o['cls'] - 1]} №{o['inst']}: " +
              ", ".join(f"p({CLASSES[k]})={p[k]:.4f}" for k in range(NCLS)))
        show_grid(o["x"])

    print("\n=== ЧУЖОЙ ПОЧЕРК, ЧУЖАЯ БУКВА, ПОРОГ ОТКАЗА ===")
    errs = [{"src": "тест", "true": int(lte[i]), "pred": int(pte[i]), "inst": te[i]["inst"],
             "p": float(Pte[i].max()), "x": Xte[i]} for i in np.where(pte != lte)[0]]
    rows = [("тест", reject_row(Pte, lte))]
    Pf = lf = Pa = None
    if other:
        Xf, _, lf = make_xy(other)
        Pf = base.proba(Xf)
        pf = Pf.argmax(axis=1)
        print(f"Чужой почерк: accuracy {acc_of(lf, pf) * 100:.1f}% (свой тест {acc_te * 100:.1f}%)")
        print_mat(conf_mat(lf, pf), "Матрица ошибок, чужой почерк:")
        rows.append(("чужой почерк", reject_row(Pf, lf)))
        errs += [{"src": "чужой почерк", "true": int(lf[i]), "pred": int(pf[i]), "inst": other[i]["inst"],
                  "p": float(Pf[i].max()), "x": Xf[i]} for i in np.where(pf != lf)[0]]
    else:
        print("other.txt не найден")
    if alien:
        Pa = base.proba(np.array([o["x"] for o in alien]))
        for p in Pa:
            print(f"Чужая буква отнесена к «{CLASSES[int(np.argmax(p))]}» с вероятностью {p.max():.4f}; "
                  f"решение: {verdict(p)}")
        rows.append(("чужая буква", reject_row(Pa, None)))
    else:
        print("alien.txt не найден")
    print_reject(rows)

    print("\n=== ПУТАНИЦА БУКВ ===")
    analyse_confusions(items, errs)

    print("\n=== ВЛИЯНИЕ ПОРОГА ===")
    tau_table(Pte, lte, Pf, lf, Pa)

    fig_convergence({H: next(x for x in res) for H, res in series.items()}, info)
    fig_epochs(series, info)
    fig_confusion(cm_te, info)
    fig_noise(total, per_cls, info)
    fig_weights(base, info)
    fig_uncertain(found, info)
    print("\nГрафики сохранены: fig_convergence, fig_epochs, fig_confusion, fig_noise, fig_weights, fig_uncertain (png)")

    demo(base, items, other, alien)
    if input("Интерактивный ввод сетки? (y или Enter для пропуска): ").strip().lower() == "y":
        interactive(base)


main()