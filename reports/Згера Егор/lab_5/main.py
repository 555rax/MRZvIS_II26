import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURNAME  = "Згера"
INITIALS = "З.А."
GROUP    = "XX-00"
LETTERS  = ["З", "Г", "Е"]
VARIANT  = 5
H_VAR    = 4
ETA      = 0.10
MMAX     = 8
EE       = 0.05
MAX_EPOCHS = 5000
TAU      = 0.6
ROWS, COLS = 7, 5
N_IN, N_OUT = ROWS * COLS, 3
BASE_SEED = 1

def read_dataset(path):
    with open(path, encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    items, i = [], 0
    while i < len(lines):
        parts = lines[i].lower().split()
        assert parts[0] == "class", f"ожидался заголовок, строка: {lines[i]}"
        cls, smp = int(parts[1]), int(parts[3])
        grid = lines[i + 1:i + 1 + ROWS]
        assert len(grid) == ROWS and all(len(r) == COLS for r in grid), f"плохая сетка: класс {cls}, №{smp}"
        vec = np.array([1.0 if ch == "#" else 0.0 for r in grid for ch in r])
        items.append((cls - 1, smp, vec))
        i += 1 + ROWS
    return items

def one_hot(k, n=N_OUT):
    y = np.zeros(n); y[k] = 1.0
    return y

def split_train_test(items):
    tr = [it for it in items if it[1] <= 7]
    te = [it for it in items if it[1] >= 8]
    f = lambda L: (np.array([it[2] for it in L]),
                   np.array([one_hot(it[0]) for it in L]),
                   np.array([it[0] for it in L]),
                   [it[1] for it in L])
    return f(tr), f(te)

def to_grid(vec):
    return vec.reshape(ROWS, COLS)

def grid_text(vec):
    return "\n".join("".join("#" if v > 0.5 else "." for v in row) for row in to_grid(vec))

def parse_grid_lines(lines):
    assert len(lines) == ROWS and all(len(r) == COLS and set(r) <= set("#.") for r in lines), \
        "нужно 7 строк по 5 символов из '#' и '.'"
    return np.array([1.0 if ch == "#" else 0.0 for r in lines for ch in r])

def dataset_quality(items):
    X = np.array([it[2] for it in items]); c = np.array([it[0] for it in items])
    nfill = int(X.sum())
    ham = lambda a, b: int(np.sum(a != b))
    intra = {k: [] for k in range(3)}; inter = []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = ham(X[i], X[j])
            if c[i] == c[j]: intra[c[i]].append(d)
            else: inter.append((min(c[i], c[j]), max(c[i], c[j]), d))
    intra_mean = {k: float(np.mean(v)) for k, v in intra.items()}
    inter_mean = {}
    for a in range(3):
        for b in range(a + 1, 3):
            inter_mean[(a, b)] = float(np.mean([d for (p, q, d) in inter if (p, q) == (a, b)]))
    return nfill, intra_mean, float(np.mean(sum(intra.values(), []))), inter_mean, float(np.mean([d for *_, d in inter]))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)

def init_params(H, rng):
    return {"W": rng.uniform(-0.5, 0.5, (H, N_IN)), "b": rng.uniform(-0.5, 0.5, H),
            "V": rng.uniform(-0.5, 0.5, (N_OUT, H)), "c": rng.uniform(-0.5, 0.5, N_OUT)}

def n_params(H):
    return H * N_IN + H + N_OUT * H + N_OUT

def forward(P, x):
    h = sigmoid(x @ P["W"].T + P["b"])
    p = softmax(h @ P["V"].T + P["c"])
    return h, p

def cross_entropy(p, y):
    return -np.sum(y * np.log(p + 1e-12), axis=-1)

def backward(P, x, y):
    h, p = forward(P, x)
    d_out = p - y
    d_hid = (P["V"].T @ d_out) * h * (1 - h)
    return {"V": np.outer(d_out, h), "c": d_out, "W": np.outer(d_hid, x), "b": d_hid}

def Es_mse(P, X, Y):
    _, p = forward(P, X)
    return float(np.mean(0.5 * np.sum((Y - p) ** 2, axis=1)))

def CE_mean(P, X, Y):
    _, p = forward(P, X)
    return float(np.mean(cross_entropy(p, Y)))

def train(X, Y, H, eta=ETA, seed=0, ee=EE, max_epochs=MAX_EPOCHS):
    rng = np.random.default_rng(seed)
    P = init_params(H, rng)
    hist_es, hist_ce = [Es_mse(P, X, Y)], [CE_mean(P, X, Y)]
    epoch = 0
    while epoch < max_epochs and hist_es[-1] > ee:
        for i in rng.permutation(len(X)):
            g = backward(P, X[i], Y[i])
            for k in P: P[k] -= eta * g[k]
        epoch += 1
        hist_es.append(Es_mse(P, X, Y)); hist_ce.append(CE_mean(P, X, Y))
    return P, epoch, hist_es, hist_ce, hist_es[-1] <= ee

def predict(P, X):
    _, p = forward(P, X)
    return np.argmax(p, axis=1), np.max(p, axis=1), p

def accuracy(true, pred):
    return float(np.mean(np.asarray(true) == np.asarray(pred)))

def confusion(true, pred, n=3):
    M = np.zeros((n, n), dtype=int)
    for t, p in zip(true, pred): M[t, p] += 1
    return M

def add_noise(x, m, rng):
    z = x.copy()
    idx = rng.choice(N_IN, size=m, replace=False)
    z[idx] = 1 - z[idx]
    return z

def noise_curve(P, X, true, mmax=MMAX, reps=50, seed=123):
    rng = np.random.default_rng(seed)
    overall, per = [], []
    for m in range(mmax + 1):
        ok = np.zeros(len(X))
        for i, x in enumerate(X):
            noisy = np.array([add_noise(x, m, rng) for _ in range(reps)])
            pr, _, _ = predict(P, noisy)
            ok[i] = np.mean(pr == true[i])
        overall.append(float(ok.mean()))
        per.append([float(ok[true == k].mean()) for k in range(3)])
    return np.array(overall), np.array(per)

def numeric_grad_check(P, x, y, eps=1e-5):
    ga = backward(P, x, y)
    worst = 0.0
    for k in P:
        flat = P[k].reshape(-1)
        gnum = np.zeros_like(flat)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps; lp = cross_entropy(forward(P, x)[1], y)
            flat[i] = old - eps; lm = cross_entropy(forward(P, x)[1], y)
            flat[i] = old
            gnum[i] = (lp - lm) / (2 * eps)
        ga_k = ga[k].reshape(-1)
        rel = np.max(np.abs(ga_k - gnum) / (np.abs(ga_k) + np.abs(gnum) + 1e-12))
        worst = max(worst, rel)
    return worst

def decide(P, x, tau=TAU):
    _, p = forward(P, x)
    k = int(np.argmax(p)); conf = float(p[k])
    verdict = LETTERS[k] if conf >= tau else "не распознано"
    return p, k, conf, verdict

def run_mode(P, x, title=""):
    p, k, conf, verdict = decide(P, x)
    print(f"--- {title}")
    print(grid_text(x))
    print("p1={:.4f}  p2={:.4f}  p3={:.4f}".format(*p))
    print(f"Класс: {LETTERS[k]} (№{k+1}); уверенность: {conf:.4f}; решение (τ={TAU}): {verdict}\n")

def reject_stats(P, X, true, name):
    pr, conf, _ = predict(P, X)
    rej = conf < TAU
    would_ok = int(np.sum(rej & (pr == true))) if true is not None else 0
    print(f"{name}: всего {len(X)}, отвергнуто {int(rej.sum())}, из них были бы верными {would_ok}"
          + ("" if true is not None else " (буквы нет в наборе - верного ответа не существует)"))

def ttl(s, nfill):
    return f"{s}\n{SURNAME}, вариант {VARIANT}, Nfill={nfill}"

def plot_all(nfill, conv, epochs_by_H, M_test, noise_all, noise_cls, P, te_X, pr_te, conf_te, te_true, te_idx):

    plt.figure(figsize=(8, 5))
    for H, (es, ce) in conv.items():
        l, = plt.plot(es, label=f"Es, H={H}")
        plt.plot(ce, "--", color=l.get_color(), label=f"CE, H={H}")
    plt.yscale("log"); plt.xlabel("эпоха"); plt.ylabel("ошибка"); plt.grid(alpha=.3); plt.legend()
    plt.title(ttl("Сходимость", nfill)); plt.tight_layout(); plt.savefig("fig_convergence.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4.5))
    for i, (H, ep) in enumerate(epochs_by_H.items()):
        plt.scatter([i] * len(ep), ep, s=50)
    plt.xticks(range(len(epochs_by_H)), [f"H={H}" for H in epochs_by_H]); plt.ylabel("число эпох")
    plt.grid(alpha=.3); plt.title(ttl("Эпохи по 5 запускам", nfill)); plt.tight_layout()
    plt.savefig("fig_epochs.png", dpi=130); plt.close()

    plt.figure(figsize=(4.5, 4.2)); plt.imshow(M_test, cmap="Blues")
    for i in range(3):
        for j in range(3): plt.text(j, i, M_test[i, j], ha="center", va="center", fontsize=14)
    plt.xticks(range(3), LETTERS); plt.yticks(range(3), LETTERS)
    plt.xlabel("предсказано"); plt.ylabel("истина"); plt.title(ttl("Матрица ошибок (тест)", nfill))
    plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4.5)); ms = np.arange(len(noise_all))
    plt.plot(ms, 100 * noise_all, "k-o", lw=2, label="общая")
    for k in range(3): plt.plot(ms, 100 * noise_cls[:, k], "--s", label=f"буква {LETTERS[k]}")
    plt.axhline(80, color="r", ls=":"); plt.xlabel("m, число инвертированных клеток"); plt.ylabel("accuracy, %")
    plt.grid(alpha=.3); plt.legend(); plt.title(ttl("Устойчивость к шуму", nfill))
    plt.tight_layout(); plt.savefig("fig_noise.png", dpi=130); plt.close()

    H = P["W"].shape[0]
    fig, axs = plt.subplots(1, H, figsize=(2.6 * H, 3.6))
    vmax = np.max(np.abs(P["W"]))
    for j in range(H):
        ax = axs[j] if H > 1 else axs
        im = ax.imshow(P["W"][j].reshape(ROWS, COLS), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"нейрон {j+1}"); ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
        ax.set_xticklabels(range(1, COLS + 1)); ax.set_yticklabels(range(1, ROWS + 1))
    fig.colorbar(im, ax=axs, shrink=0.8)
    fig.suptitle(ttl("Карты весов W (красный +, синий -)", nfill)); plt.savefig("fig_weights.png", dpi=130, bbox_inches="tight"); plt.close()

    order = np.argsort(conf_te)[:3]
    _, p_all = forward(P, te_X)
    fig, axs = plt.subplots(1, 3, figsize=(10, 4.4))
    for ax, i in zip(axs, order):
        ax.imshow(to_grid(te_X[i]), cmap="Greys")
        ax.set_title(f"{LETTERS[te_true[i]]} №{te_idx[i]}\n" + " ".join(f"{LETTERS[k]}={p_all[i,k]:.3f}" for k in range(3)), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(ttl("Три самых неуверенных теста", nfill)); plt.savefig("fig_uncertain.png", dpi=130, bbox_inches="tight"); plt.close()
    return order

def describe_neurons(P):
    print("\n=== Карты весов: интерпретация нейронов ===")
    for j in range(P["W"].shape[0]):
        w = P["W"][j]
        top = np.argsort(-np.abs(w))[:4]
        cells = ", ".join(f"({t//COLS+1},{t%COLS+1}):{w[t]:+.2f}" for t in top)
        v = P["V"][:, j]
        pos, neg = LETTERS[int(np.argmax(v))], LETTERS[int(np.argmin(v))]
        print(f"нейрон {j+1}: v=[{', '.join(f'{LETTERS[k]}:{v[k]:+.2f}' for k in range(3))}] -> "
              f"усиливает «{pos}», подавляет «{neg}»; макс |w| в клетках (строка,столбец): {cells}")

def main():
    items = read_dataset("dataset.txt")
    print("=== Контроль качества датасета ===")
    nfill, intra, intra_all, inter, inter_all = dataset_quality(items)
    print(f"Nfill = {nfill}")
    for k in range(3): print(f"внутри класса {LETTERS[k]}: среднее Хэмминга = {intra[k]:.2f}")
    print(f"внутри классов (общее) = {intra_all:.2f}" + ("   !!! < 3: перепишите экземпляры" if intra_all < 3 else ""))
    for (a, b), v in inter.items(): print(f"между {LETTERS[a]} и {LETTERS[b]}: {v:.2f}")
    print(f"между классами (общее) = {inter_all:.2f}")

    (Xtr, Ytr, ctr, itr), (Xte, Yte, cte, ite) = split_train_test(items)

    print("\n=== Проверка градиентов (численно) ===")
    Pchk = init_params(H_VAR, np.random.default_rng(0))
    print(f"макс. относительное отклонение: {numeric_grad_check(Pchk, Xtr[0], Ytr[0]):.2e}")

    print("\n=== Базовый эксперимент ===")
    P, ep, es_h, ce_h, ok = train(Xtr, Ytr, H_VAR, ETA, BASE_SEED)
    prtr, _, _ = predict(P, Xtr); prte, conf_te, _ = predict(P, Xte)
    Mtr, Mte = confusion(ctr, prtr), confusion(cte, prte)
    print(f"H={H_VAR}, eta={ETA}: эпох = {ep}, критерий достигнут: {ok}, Es = {es_h[-1]:.4f}, CE = {ce_h[-1]:.4f}")
    print(f"accuracy train = {accuracy(ctr, prtr):.3f}, test = {accuracy(cte, prte):.3f}")
    print("Матрица ошибок (train):\n", Mtr, "\nМатрица ошибок (test):\n", Mte)
    right, wrong = prte == cte, prte != cte
    print(f"средняя уверенность на верных тестах: {conf_te[right].mean():.3f}" if right.any() else "верных нет")
    print(f"средняя уверенность на неверных тестах: {conf_te[wrong].mean():.3f}" if wrong.any() else "неверных тестовых примеров нет")
    for t, q, s in zip(cte, prte, ite):
        if t != q: print(f"  ошибка: {LETTERS[t]} №{s} -> {LETTERS[q]}")

    print("\n=== Исследование H (5 запусков) ===")
    conv, epochs_by_H = {}, {}
    print("H | параметров | эпохи ср (min-max) | test acc ср (min-max) | не достигли")
    for H in (2, H_VAR, 2 * H_VAR):
        eps, accs, fails = [], [], 0
        for s in range(5):
            Ph, e, es, ce, okh = train(Xtr, Ytr, H, ETA, seed=100 + s)
            eps.append(e); accs.append(accuracy(cte, predict(Ph, Xte)[0])); fails += (not okh)
            if s == 0: conv[H] = (es, ce)
        epochs_by_H[H] = eps
        print(f"{H} | {n_params(H)} | {np.mean(eps):.0f} ({min(eps)}-{max(eps)}) | "
              f"{100*np.mean(accs):.1f}% ({100*min(accs):.1f}-{100*max(accs):.1f}) | {fails}")

    print("\n=== Шум ===")
    nall, ncls = noise_curve(P, Xte, cte)
    for m in range(len(nall)):
        print(f"m={m}: общая {100*nall[m]:.1f}%  " + "  ".join(f"{LETTERS[k]} {100*ncls[m,k]:.1f}%" for k in range(3)))
    below = [m for m in range(len(nall)) if nall[m] < 0.8]
    print("наименьшее m с accuracy < 80%:", below[0] if below else f"не достигнуто при m<={MMAX}")

    describe_neurons(P)
    order = plot_all(nfill, conv, epochs_by_H, Mte, nall, ncls, P, Xte, prte, conf_te, cte, ite)
    _, p_all = forward(P, Xte)
    print("\n=== Три самых неуверенных тестовых примера ===")
    for i in order:
        print(f"{LETTERS[cte[i]]} №{ite[i]}  p=" + ", ".join(f"{LETTERS[k]}:{p_all[i,k]:.4f}" for k in range(3)))
        print(grid_text(Xte[i]))

    print("\n=== Чужой почерк / чужая буква / порог отказа ===")
    reject_stats(P, Xte, cte, "Тест (мой почерк)")
    try:
        fr = read_dataset("foreign.txt")
        Xf = np.array([it[2] for it in fr]); cf = np.array([it[0] for it in fr])
        prf = predict(P, Xf)[0]
        print(f"accuracy на чужом почерке = {accuracy(cf, prf):.3f}; матрица:\n{confusion(cf, prf)}")
        reject_stats(P, Xf, cf, "Чужой почерк")
    except FileNotFoundError:
        fr = None; print("foreign.txt не найден - пропуск")
    try:
        al = read_dataset("alien.txt"); Xa = np.array([it[2] for it in al])
        pa, ca, _ = predict(P, Xa)
        for i in range(len(Xa)):
            print(f"чужая буква -> класс {LETTERS[pa[i]]}, p={ca[i]:.4f}")
        reject_stats(P, Xa, None, "Чужая буква")
    except FileNotFoundError:
        al = None; print("alien.txt не найден - пропуск")

    print("\n=== Режим функционирования (демонстрация) ===")
    for i in range(3): run_mode(P, Xtr[i], f"обучающий: {LETTERS[ctr[i]]} №{itr[i]}")
    for i in range(3): run_mode(P, Xte[i], f"тестовый: {LETTERS[cte[i]]} №{ite[i]}")
    rng = np.random.default_rng(7)
    for m, i in ((3, 0), (6, 4)):
        run_mode(P, add_noise(Xte[i], m, rng), f"искажённый: {LETTERS[cte[i]]} №{ite[i]}, изменено клеток: {m}")
    if al: run_mode(P, al[0][2], "чужая буква")
    if fr:
        for it in fr[:2]: run_mode(P, it[2], f"чужой почерк: {LETTERS[it[0]]} №{it[1]}")

    if "--interactive" in sys.argv:
        print("Введите 7 строк по 5 символов (#/.), пустая строка - выход")
        while True:
            lines = []
            for _ in range(ROWS):
                s = input().strip()
                if not s: return
                lines.append(s)
            try: run_mode(P, parse_grid_lines(lines), "введённая сетка")
            except AssertionError as e: print("Ошибка:", e)

if __name__ == "__main__":
    main()