import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAM = "Пасевич"
INI = "К.Ю."
GRP = "ИИ-26"
CLS = ["П", "А", "С"]
VAR = 9
HID = 4
LR = 0.20
MMAX = 10
ETHR = 0.05
EMAX = 5000
TAU = 0.6
NR, NC = 7, 5
NIN, NOUT = NR * NC, 3
SEED0 = 1

def load_data(path):
    with open(path, encoding="utf-8") as f:
        rows = [x.strip() for x in f if x.strip()]
    data, k = [], 0
    while k < len(rows):
        tok = rows[k].lower().split()
        assert tok[0] == "class", f"bad header: {rows[k]}"
        lab, num = int(tok[1]), int(tok[3])
        blk = rows[k + 1:k + 1 + NR]
        assert len(blk) == NR and all(len(r) == NC for r in blk), \
            f"bad grid: class {lab}, #{num}"
        v = np.array([1.0 if c == "#" else 0.0 for r in blk for c in r])
        data.append((lab - 1, num, v))
        k += 1 + NR
    return data

def hot(i, n=NOUT):
    v = np.zeros(n); v[i] = 1.0
    return v

def split_ds(data):
    a = [d for d in data if d[1] <= 7]
    b = [d for d in data if d[1] >= 8]
    f = lambda L: (np.array([d[2] for d in L]),
                   np.array([hot(d[0]) for d in L]),
                   np.array([d[0] for d in L]),
                   [d[1] for d in L])
    return f(a), f(b)

def to_mat(v):
    return v.reshape(NR, NC)

def mat_str(v):
    return "\n".join("".join("#" if x > 0.5 else "." for x in r) for r in to_mat(v))

def parse_mat(lines):
    assert len(lines) == NR and all(len(r) == NC and set(r) <= set("#.") for r in lines), \
        "need 7 lines of 5 chars from '#' and '.'"
    return np.array([1.0 if c == "#" else 0.0 for r in lines for c in r])

def qc(data):
    X = np.array([d[2] for d in data]); y = np.array([d[0] for d in data])
    nf = int(X.sum())
    hd = lambda a, b: int(np.sum(a != b))
    w = {i: [] for i in range(3)}; btw = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = hd(X[i], X[j])
            if y[i] == y[j]: w[y[i]].append(d)
            else: btw.append((min(y[i], y[j]), max(y[i], y[j]), d))
    wm = {i: float(np.mean(v)) for i, v in w.items()}
    bm = {}
    for a in range(3):
        for b in range(a + 1, 3):
            bm[(a, b)] = float(np.mean([d for (p, q, d) in btw if (p, q) == (a, b)]))
    return (nf, wm, float(np.mean(sum(w.values(), []))),
            bm, float(np.mean([d for *_, d in btw])))

def sig(z):
    return 1.0 / (1.0 + np.exp(-z))

def sm(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def init(H, rng):
    return {"W": rng.uniform(-0.5, 0.5, (H, NIN)), "b": rng.uniform(-0.5, 0.5, H),
            "V": rng.uniform(-0.5, 0.5, (NOUT, H)), "c": rng.uniform(-0.5, 0.5, NOUT)}

def npar(H):
    return H * NIN + H + NOUT * H + NOUT

def fwd(P, x):
    h = sig(x @ P["W"].T + P["b"])
    p = sm(h @ P["V"].T + P["c"])
    return h, p

def ce(p, y):
    return -np.sum(y * np.log(p + 1e-12), axis=-1)

def bwd(P, x, y):
    h, p = fwd(P, x)
    d = p - y
    dh = (P["V"].T @ d) * h * (1 - h)
    return {"V": np.outer(d, h), "c": d, "W": np.outer(dh, x), "b": dh}

def es_ce(P, X, Y):
    _, p = fwd(P, X)
    return float(np.mean(ce(p, Y)))

def es_mse(P, X, Y):
    _, p = fwd(P, X)
    return float(np.mean(0.5 * np.sum((Y - p) ** 2, axis=1)))

def fit(X, Y, H, eta=LR, seed=0, ethr=ETHR, emax=EMAX):
    rng = np.random.default_rng(seed)
    P = init(H, rng)
    h1, h2 = [es_ce(P, X, Y)], [es_ce(P, X, Y)]
    ep = 0
    while ep < emax and h1[-1] > ethr:
        for i in rng.permutation(len(X)):
            g = bwd(P, X[i], Y[i])
            for k in P: P[k] -= eta * g[k]
        ep += 1
        e = es_ce(P, X, Y)
        h1.append(e); h2.append(e)
    return P, ep, h1, h2, h1[-1] <= ethr

def pred(P, X):
    _, p = fwd(P, X)
    return np.argmax(p, axis=1), np.max(p, axis=1), p

def acc(t, p):
    return float(np.mean(np.asarray(t) == np.asarray(p)))

def cm(t, p, n=3):
    M = np.zeros((n, n), dtype=int)
    for a, b in zip(t, p): M[a, b] += 1
    return M

def flip(x, m, rng):
    z = x.copy()
    idx = rng.choice(NIN, size=m, replace=False)
    z[idx] = 1 - z[idx]
    return z

def noise(P, X, t, mmax=MMAX, rep=50, seed=123):
    rng = np.random.default_rng(seed)
    all_, per = [], []
    for m in range(mmax + 1):
        ok = np.zeros(len(X))
        for i, x in enumerate(X):
            nb = np.array([flip(x, m, rng) for _ in range(rep)])
            pr, _, _ = pred(P, nb)
            ok[i] = np.mean(pr == t[i])
        all_.append(float(ok.mean()))
        per.append([float(ok[t == k].mean()) for k in range(3)])
    return np.array(all_), np.array(per)

def grad_check(P, x, y, eps=1e-5):
    ga = bwd(P, x, y)
    worst = 0.0
    for k in P:
        flat = P[k].reshape(-1)
        gn = np.zeros_like(flat)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps; lp = ce(fwd(P, x)[1], y)
            flat[i] = old - eps; lm = ce(fwd(P, x)[1], y)
            flat[i] = old
            gn[i] = (lp - lm) / (2 * eps)
        gk = ga[k].reshape(-1)
        rel = np.max(np.abs(gk - gn) / (np.abs(gk) + np.abs(gn) + 1e-12))
        worst = max(worst, rel)
    return worst

def judge(P, x, tau=TAU):
    _, p = fwd(P, x)
    k = int(np.argmax(p)); cf = float(p[k])
    out = CLS[k] if cf >= tau else "не распознано"
    return p, k, cf, out

def run(P, x, title=""):
    p, k, cf, out = judge(P, x)
    print(f"--- {title}")
    print(mat_str(x))
    print("p1={:.4f}  p2={:.4f}  p3={:.4f}".format(*p))
    print(f"Класс: {CLS[k]} (№{k+1}); уверенность: {cf:.4f}; решение (τ={TAU}): {out}\n")

def rej(P, X, t, name):
    pr, cf, _ = pred(P, X)
    r = cf < TAU
    ok = int(np.sum(r & (pr == t))) if t is not None else 0
    print(f"{name}: всего {len(X)}, отвергнуто {int(r.sum())}, "
          f"из них были бы верными {ok}"
          + ("" if t is not None else " (буквы нет в наборе - верного ответа не существует)"))

def hdr(s, nf):
    return f"{s}\n{FAM} {INI}, гр. {GRP}, вариант {VAR}, Nfill={nf}"

def plots(nf, cv, ep_h, M, na, nc, P, Xte, pr, cf, tt, ti):
    plt.figure(figsize=(8, 5))
    for H, (a, b) in cv.items():
        l, = plt.plot(a, label=f"Es, H={H}")
        plt.plot(b, "--", color=l.get_color(), label=f"CE, H={H}")
    plt.yscale("log"); plt.xlabel("эпоха"); plt.ylabel("ошибка")
    plt.grid(alpha=.3); plt.legend()
    plt.title(hdr("Сходимость", nf)); plt.tight_layout()
    plt.savefig("fig_convergence.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4.5))
    for i, (H, e) in enumerate(ep_h.items()):
        plt.scatter([i] * len(e), e, s=50)
    plt.xticks(range(len(ep_h)), [f"H={H}" for H in ep_h])
    plt.ylabel("число эпох"); plt.grid(alpha=.3)
    plt.title(hdr("Эпохи по 5 запускам", nf)); plt.tight_layout()
    plt.savefig("fig_epochs.png", dpi=130); plt.close()

    plt.figure(figsize=(4.5, 4.2)); plt.imshow(M, cmap="Blues")
    for i in range(3):
        for j in range(3): plt.text(j, i, M[i, j], ha="center", va="center", fontsize=14)
    plt.xticks(range(3), CLS); plt.yticks(range(3), CLS)
    plt.xlabel("предсказано"); plt.ylabel("истина")
    plt.title(hdr("Матрица ошибок (тест)", nf))
    plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4.5)); ms = np.arange(len(na))
    plt.plot(ms, 100 * na, "k-o", lw=2, label="общая")
    for k in range(3):
        plt.plot(ms, 100 * nc[:, k], "--s", label=f"буква {CLS[k]}")
    plt.axhline(80, color="r", ls=":")
    plt.xlabel("m, число инвертированных клеток"); plt.ylabel("accuracy, %")
    plt.grid(alpha=.3); plt.legend()
    plt.title(hdr("Устойчивость к шуму", nf))
    plt.tight_layout(); plt.savefig("fig_noise.png", dpi=130); plt.close()

    H = P["W"].shape[0]
    fig, axs = plt.subplots(1, H, figsize=(2.6 * H, 3.6))
    vmax = np.max(np.abs(P["W"]))
    for j in range(H):
        ax = axs[j] if H > 1 else axs
        im = ax.imshow(P["W"][j].reshape(NR, NC), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"нейрон {j+1}")
        ax.set_xticks(range(NC)); ax.set_yticks(range(NR))
        ax.set_xticklabels(range(1, NC + 1)); ax.set_yticklabels(range(1, NR + 1))
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle(hdr("Карты весов W (красный +, синий -)", nf))
    plt.savefig("fig_weights.png", dpi=130, bbox_inches="tight"); plt.close()

    order = np.argsort(cf)[:3]
    _, pa = fwd(P, Xte)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4.4))
    for ax, i in zip(axs, order):
        ax.imshow(to_mat(Xte[i]), cmap="Greys")
        ax.set_title(f"{CLS[tt[i]]} №{ti[i]}\n"
                     + " ".join(f"{CLS[k]}={pa[i,k]:.3f}" for k in range(3)), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(hdr("Три самых неуверенных теста", nf))
    plt.savefig("fig_uncertain.png", dpi=130, bbox_inches="tight"); plt.close()
    return order

def neurons(P):
    print("\n=== Карты весов: интерпретация нейронов ===")
    for j in range(P["W"].shape[0]):
        w = P["W"][j]
        top = np.argsort(-np.abs(w))[:4]
        cells = ", ".join(f"({t//NC+1},{t%NC+1}):{w[t]:+.2f}" for t in top)
        v = P["V"][:, j]
        a, b = CLS[int(np.argmax(v))], CLS[int(np.argmin(v))]
        print(f"нейрон {j+1}: v=[{', '.join(f'{CLS[k]}:{v[k]:+.2f}' for k in range(3))}] -> "
              f"усиливает «{a}», подавляет «{b}»; макс |w| в клетках (строка,столбец): {cells}")

def main():
    data = load_data("dataset.txt")
    print("=== Контроль качества датасета ===")
    nf, wm, wa, bm, ba = qc(data)
    print(f"Nfill = {nf}")
    for k in range(3):
        print(f"внутри класса {CLS[k]}: среднее Хэмминга = {wm[k]:.2f}")
    print(f"внутри классов (общее) = {wa:.2f}"
          + ("   !!! < 3: перепишите экземпляры" if wa < 3 else ""))
    for (a, b), v in bm.items():
        print(f"между {CLS[a]} и {CLS[b]}: {v:.2f}")
    print(f"между классами (общее) = {ba:.2f}")

    (Xtr, Ytr, ctr, itr), (Xte, Yte, cte, ite) = split_ds(data)

    print("\n=== Проверка градиентов (численно) ===")
    Pc = init(HID, np.random.default_rng(0))
    print(f"макс. относительное отклонение: {grad_check(Pc, Xtr[0], Ytr[0]):.2e}")

    print("\n=== Базовый эксперимент ===")
    P, ep, h1, h2, ok = fit(Xtr, Ytr, HID, LR, SEED0)
    prtr, _, _ = pred(P, Xtr); prte, cf, _ = pred(P, Xte)
    Mtr, Mte = cm(ctr, prtr), cm(cte, prte)
    print(f"H={HID}, eta={LR}: эпох = {ep}, критерий достигнут: {ok}, "
          f"Es(CE) = {h1[-1]:.4f}")
    print(f"accuracy train = {acc(ctr, prtr):.3f}, test = {acc(cte, prte):.3f}")
    print("Матрица ошибок (train):\n", Mtr, "\nМатрица ошибок (test):\n", Mte)
    r, w = prte == cte, prte != cte
    print(f"средняя уверенность на верных тестах: {cf[r].mean():.3f}" if r.any() else "верных нет")
    print(f"средняя уверенность на неверных тестах: {cf[w].mean():.3f}" if w.any() else "неверных тестовых примеров нет")
    for a, b, s in zip(cte, prte, ite):
        if a != b: print(f"  ошибка: {CLS[a]} №{s} -> {CLS[b]}")

    print("\n=== Исследование H (5 запусков) ===")
    cv, ep_h = {}, {}
    print("H | параметров | эпохи ср (min-max) | test acc ср (min-max) | не достигли")
    for H in (2, HID, 2 * HID):
        es, aс, f = [], [], 0
        for s in range(5):
            Ph, e, ea, eb, okh = fit(Xtr, Ytr, H, LR, seed=100 + s)
            es.append(e); aс.append(acc(cte, pred(Ph, Xte)[0])); f += (not okh)
            if s == 0: cv[H] = (ea, eb)
        ep_h[H] = es
        print(f"{H} | {npar(H)} | {np.mean(es):.0f} ({min(es)}-{max(es)}) | "
              f"{100*np.mean(aс):.1f}% ({100*min(aс):.1f}-{100*max(aс):.1f}) | {f}")

    print("\n=== Шум ===")
    na, nc = noise(P, Xte, cte)
    for m in range(len(na)):
        print(f"m={m}: общая {100*na[m]:.1f}%  "
              + "  ".join(f"{CLS[k]} {100*nc[m,k]:.1f}%" for k in range(3)))
    below = [m for m in range(len(na)) if na[m] < 0.8]
    print("наименьшее m с accuracy < 80%:", below[0] if below else f"не достигнуто при m<={MMAX}")

    neurons(P)
    order = plots(nf, cv, ep_h, Mte, na, nc, P, Xte, prte, cf, cte, ite)
    _, pa = fwd(P, Xte)
    print("\n=== Три самых неуверенных тестовых примера ===")
    for i in order:
        print(f"{CLS[cte[i]]} №{ite[i]}  p="
              + ", ".join(f"{CLS[k]}:{pa[i,k]:.4f}" for k in range(3)))
        print(mat_str(Xte[i]))

    print("\n=== Чужой почерк / чужая буква / порог отказа ===")
    rej(P, Xte, cte, "Тест (мой почерк)")
    try:
        fr = load_data("foreign.txt")
        Xf = np.array([d[2] for d in fr]); cf_ = np.array([d[0] for d in fr])
        prf = pred(P, Xf)[0]
        print(f"accuracy на чужом почерке = {acc(cf_, prf):.3f}; матрица:\n{cm(cf_, prf)}")
        rej(P, Xf, cf_, "Чужой почерк")
    except FileNotFoundError:
        fr = None; print("foreign.txt не найден - пропуск")
    try:
        al = load_data("alien.txt"); Xa = np.array([d[2] for d in al])
        pa, ca, _ = pred(P, Xa)
        for i in range(len(Xa)):
            print(f"чужая буква -> класс {CLS[pa[i]]}, p={ca[i]:.4f}")
        rej(P, Xa, None, "Чужая буква")
    except FileNotFoundError:
        al = None; print("alien.txt не найден - пропуск")

    print("\n=== Режим функционирования (демонстрация) ===")
    for i in range(3): run(P, Xtr[i], f"обучающий: {CLS[ctr[i]]} №{itr[i]}")
    for i in range(3): run(P, Xte[i], f"тестовый: {CLS[cte[i]]} №{ite[i]}")
    rng = np.random.default_rng(7)
    for m, i in ((3, 0), (6, 4)):
        run(P, flip(Xte[i], m, rng),
            f"искажённый: {CLS[cte[i]]} №{ite[i]}, изменено клеток: {m}")
    if al: run(P, al[0][2], "чужая буква")
    if fr:
        for d in fr[:2]:
            run(P, d[2], f"чужой почерк: {CLS[d[0]]} №{d[1]}")

    if "--interactive" in sys.argv:
        print("Введите 7 строк по 5 символов (#/.), пустая строка - выход")
        while True:
            lines = []
            for _ in range(NR):
                s = input().strip()
                if not s: return
                lines.append(s)
            try: run(P, parse_mat(lines), "введённая сетка")
            except AssertionError as e: print("Ошибка:", e)

if __name__ == "__main__":
    main()