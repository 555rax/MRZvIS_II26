import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
SURNAME, VARIANT = "Петручик", 10
H_VAR, LR, M_MAX = 3, 0.05, 8
LETTERS = ["П", "Е", "Т"]           
EE, MAX_EPOCHS = 0.05, 5000
TAU = 0.6                           
SEED = 42
SEEDS = [1, 2, 3, 4, 5]
ROWS, COLS, D = 7, 5, 35
 
def read_dataset(path):
    """Читает файл: заголовок '>> класс=.. буква=.. экз=..' и 7 строк по 5 символов."""
    X, cls, inst, grids = [], [], [], []
    with open(path, encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        head = lines[i]
        assert head.startswith(">>"), f"ожидался заголовок, строка {i + 1}"
        kv = dict(p.split("=") for p in head[2:].split())
        rows = lines[i + 1:i + 1 + ROWS]
        assert len(rows) == ROWS and all(len(r) == COLS for r in rows), f"неверная сетка у {head}"
        X.append([1.0 if ch == "#" else 0.0 for r in rows for ch in r])
        cls.append(int(kv["класс"]) - 1)     
        inst.append(int(kv["экз"]))
        grids.append(rows)
        i += 1 + ROWS
    return np.array(X), np.array(cls), np.array(inst), grids
 
 
def split_train_test(X, cls, inst):
    tr = inst <= 7
    return X[tr], cls[tr], X[~tr], cls[~tr]
 
 
def hamming(a, b):
    return int(np.sum(a != b))
 
 
def dataset_quality(X, cls):
    """N_fill, среднее попарное расстояние Хэмминга внутри классов и между классами."""
    n_fill = int(X.sum())
    intra, inter = [], []
    per_class = {}
    for c in range(3):
        idx = np.where(cls == c)[0]
        d = [hamming(X[i], X[j]) for a, i in enumerate(idx) for j in idx[a + 1:]]
        per_class[c] = float(np.mean(d))
        intra += d
    for c1 in range(3):
        for c2 in range(c1 + 1, 3):
            inter += [hamming(X[i], X[j]) for i in np.where(cls == c1)[0] for j in np.where(cls == c2)[0]]
    return n_fill, float(np.mean(intra)), float(np.mean(inter)), per_class
 
 
def one_hot(c, k=3):
    y = np.zeros(k)
    y[c] = 1.0
    return y
 
def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -700, 700)))
 
 
def softmax(z):
    e = np.exp(z - z.max())            
    return e / e.sum()
 
 
def init_weights(h, rng, scale=0.3):
    return {"W1": rng.uniform(-scale, scale, (h, D)), "b1": rng.uniform(-scale, scale, h),
            "W2": rng.uniform(-scale, scale, (3, h)), "b2": rng.uniform(-scale, scale, 3)}
 
 
def n_params(h):
    return h * D + h + 3 * h + 3
 
 
def forward(w, x):
    h = sigmoid(w["W1"] @ x + w["b1"])
    p = softmax(w["W2"] @ h + w["b2"])
    return h, p
 
 
def cross_entropy(p, y):
    return float(-np.sum(y * np.log(np.clip(p, 1e-12, 1.0))))
 
 
def backward(w, x, h, p, y):
    """Градиенты CE по параметрам: dE/dz = p - y; dE/ds_j = (V^T (p-y))_j * h_j (1-h_j)."""
    dz = p - y
    dW2 = np.outer(dz, h)
    db2 = dz
    ds = (w["W2"].T @ dz) * h * (1.0 - h)
    dW1 = np.outer(ds, x)
    db1 = ds
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
 
 
def mean_ce(w, X, C):
    return float(np.mean([cross_entropy(forward(w, x)[1], one_hot(c)) for x, c in zip(X, C)]))
 
 
def predict(w, X):
    P = np.array([forward(w, x)[1] for x in X])
    return P.argmax(1), P
 
 
def train(Xtr, Ctr, h, seed, lr=LR, ee=EE, max_epochs=MAX_EPOCHS, Xte=None, Cte=None):
    """Онлайн-обучение: пример за примером, порядок перемешивается в каждой эпохе."""
    rng = np.random.default_rng(seed)
    w = init_weights(h, rng)
    hist, hist_te = [], []
    for epoch in range(1, max_epochs + 1):
        for i in rng.permutation(len(Xtr)):
            x, y = Xtr[i], one_hot(Ctr[i])
            hh, p = forward(w, x)
            g = backward(w, x, hh, p, y)
            for k in w:
                w[k] = w[k] - lr * g[k]
        es = mean_ce(w, Xtr, Ctr)
        hist.append(es)
        if Xte is not None:
            hist_te.append(mean_ce(w, Xte, Cte))
        if es <= ee:
            return w, epoch, hist, hist_te, True
    return w, max_epochs, hist, hist_te, False
 
 
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))
 
 
def confusion_matrix(y_true, y_pred, k=3):
    M = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M
 
def add_noise(x, m, rng):
    z = x.copy()
    if m > 0:
        idx = rng.choice(D, size=m, replace=False)
        z[idx] = 1.0 - z[idx]
    return z
 
 
def noise_study(w, Xte, Cte, m_max, reps=50, seed=0):
    rng = np.random.default_rng(seed)
    overall = np.zeros(m_max + 1)
    per_class = np.zeros((3, m_max + 1))
    for m in range(m_max + 1):
        ok = np.zeros(len(Xte))
        for i, x in enumerate(Xte):
            hits = 0
            for _ in range(reps):
                hits += int(forward(w, add_noise(x, m, rng))[1].argmax() == Cte[i])
            ok[i] = hits / reps
        overall[m] = ok.mean()
        for c in range(3):
            per_class[c, m] = ok[Cte == c].mean()
    return overall, per_class
 
def numerical_gradient(w, x, y, eps=1e-5):
    g = {}
    for k in w:
        arr = np.array(w[k], dtype=float)
        gk = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + eps
            w2 = dict(w); w2[k] = arr.copy()
            lp = cross_entropy(forward(w2, x)[1], y)
            arr[idx] = old - eps
            w2[k] = arr.copy()
            lm = cross_entropy(forward(w2, x)[1], y)
            arr[idx] = old
            gk[idx] = (lp - lm) / (2 * eps)
        g[k] = gk
    return g
 
 
def gradient_check(w, x, y):
    ga = backward(w, x, *forward(w, x), y)
    gn = numerical_gradient(w, x, y)
    worst = 0.0
    for k in w:
        num = np.abs(ga[k] - gn[k])
        den = np.maximum(np.abs(ga[k]) + np.abs(gn[k]), 1e-12)
        worst = max(worst, float(np.max(num / den)))
    return worst
 
def grid_to_vec(rows):
    if len(rows) != ROWS or any(len(r) != COLS or set(r) - {"#", "."} for r in rows):
        raise ValueError("нужно 7 строк по 5 символов '#' и '.'")
    return np.array([1.0 if ch == "#" else 0.0 for r in rows for ch in r])
 
 
def run_mode(w, rows, title="", true_cls=None, changed=None):
    x = grid_to_vec(rows)
    _, p = forward(w, x)
    c = int(p.argmax())
    conf = float(p[c])
    print(f"--- {title}" + (f" (изменено клеток: {changed})" if changed is not None else ""))
    print("\n".join(rows))
    print("вероятности: p1 = %.4f (П), p2 = %.4f (Е), p3 = %.4f (Т)" % tuple(p))
    print(f"класс: {LETTERS[c]}, уверенность: {conf:.4f}")
    verdict = LETTERS[c] if conf >= TAU else "не распознано"
    extra = ""
    if true_cls is not None and true_cls >= 0:
        extra = f"  [истинная: {LETTERS[true_cls]}, {'верно' if c == true_cls else 'ОШИБКА'}]"
    print(f"решение (tau = {TAU}): {verdict}{extra}")
    return p
 
 
def interactive_mode(w):
    print("Введите 7 строк по 5 символов ('#' и '.'), q - выход.")
    while True:
        rows = []
        while len(rows) < ROWS:
            try:
                s = input(f"строка {len(rows) + 1}> ").strip()
            except (EOFError, KeyboardInterrupt):
                return
            if s.lower() in ("q", "exit"):
                return
            rows.append(s)
        try:
            run_mode(w, rows, "ввод пользователя")
        except ValueError as e:
            print("Ошибка ввода:", e)
 
def show_grid(ax, x, title=""):
    ax.imshow(x.reshape(ROWS, COLS), cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-.5, COLS, 1), minor=True); ax.set_yticks(np.arange(-.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="gray", lw=0.5); ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=9)
 
 
if __name__ == "__main__":
    X, C, I, grids = read_dataset("dataset.txt")
    n_fill, intra, inter, per_class = dataset_quality(X, C)
    TAG = f"{SURNAME}, вариант {VARIANT}, N_fill = {n_fill}"
    print(f"N_fill = {n_fill}")
    print(f"среднее расстояние Хэмминга: внутри классов = {intra:.2f}, между классами = {inter:.2f}")
    for c in range(3):
        print(f"  класс {LETTERS[c]}: внутриклассовое среднее = {per_class[c]:.2f}")
    print("внутриклассовое среднее ≥ 3:", intra >= 3)
    Xtr, Ctr, Xte, Cte = split_train_test(X, C, I)
    print(f"обучающая: {len(Xtr)}, тестовая: {len(Xte)}")
 
    w, ep, hist, hist_te, ok = train(Xtr, Ctr, H_VAR, SEED, Xte=Xte, Cte=Cte)
    print(f"\nБАЗОВЫЙ: H = {H_VAR}, eta = {LR}, эпох = {ep}, Es = {hist[-1]:.5f}, критерий достигнут: {ok}")
    ptr, Ptr = predict(w, Xtr)
    pte, Pte = predict(w, Xte)
    print(f"accuracy train = {accuracy(Ctr, ptr):.4f}, test = {accuracy(Cte, pte):.4f}")
    Mtr, Mte = confusion_matrix(Ctr, ptr), confusion_matrix(Cte, pte)
    print("матрица ошибок (train):\n", Mtr)
    print("матрица ошибок (test):\n", Mte)
    conf_te = Pte.max(1)
    right = pte == Cte
    print("средняя уверенность на верных тестовых: %.4f" % (conf_te[right].mean() if right.any() else float("nan")))
    print("средняя уверенность на неверных тестовых: %.4f" % (conf_te[~right].mean() if (~right).any() else float("nan")))
    print("ошибочные тестовые примеры (буква, экз, предсказание):",
          [(LETTERS[Cte[i]], int(I[I > 7][i]), LETTERS[pte[i]]) for i in np.where(~right)[0]])
 
    err = gradient_check(w, Xtr[0], one_hot(Ctr[0]))
    print(f"\nПроверка градиентов: макс. относительная ошибка = {err:.2e}")
 
    print("\nИССЛЕДОВАНИЕ H")
    study = {}
    for h in [2, H_VAR, 2 * H_VAR]:
        runs = []
        for s in SEEDS:
            wi, e, hi, hti, oki = train(Xtr, Ctr, h, s, Xte=Xte, Cte=Cte)
            acc = accuracy(Cte, predict(wi, Xte)[0])
            runs.append({"seed": s, "w": wi, "epochs": e, "hist": hi, "hist_te": hti, "ok": oki, "acc": acc,
                         "Es": hi[-1]})
        study[h] = runs
        E_ = np.array([r["epochs"] for r in runs]); A_ = np.array([r["acc"] for r in runs])
        print(f"H={h}: эпохи {E_.mean():.1f} [{E_.min()}-{E_.max()}] (σ={E_.std():.1f}), "
              f"test acc {A_.mean() * 100:.1f}% [{A_.min() * 100:.1f}-{A_.max() * 100:.1f}], "
              f"не достигли: {sum(not r['ok'] for r in runs)}, параметров: {n_params(h)}, "
              f"эпохи по seed: {[int(v) for v in E_]}, acc по seed: {[round(float(a) * 100, 1) for a in A_]}")

    overall, per_cls = noise_study(w, Xte, Cte, M_MAX)
    print("\nШУМ: accuracy(m) общий:", np.round(overall * 100, 1))
    for c in range(3):
        print(f"   {LETTERS[c]}:", np.round(per_cls[c] * 100, 1))
    below = np.where(overall < 0.8)[0]
    print("наименьшее m, при котором accuracy < 80%:", int(below[0]) if len(below) else "не достигнуто при m ≤ %d" % M_MAX)
 
    print("\nВЕСА СКРЫТЫХ НЕЙРОНОВ")
    for j in range(H_VAR):
        wj = w["W1"][j].reshape(ROWS, COLS)
        top = np.dstack(np.unravel_index(np.argsort(-np.abs(wj).ravel())[:4], wj.shape))[0]
        cells = [(int(r) + 1, int(c) + 1, round(float(wj[r, c]), 2)) for r, c in top]
        print(f"нейрон {j + 1}: v_k{j + 1} (П,Е,Т) = {np.round(w['W2'][:, j], 2)}, b = {w['b1'][j]:.2f}; "
              f"наибольшие |w| (строка, столбец, вес): {cells}")

    order = np.argsort(conf_te)[:3]
    print("\nСАМЫЕ НЕУВЕРЕННЫЕ ТЕСТОВЫЕ")
    for i in order:
        g = grids[int(np.where(I > 7)[0][i])]
        print(f"{LETTERS[Cte[i]]}, экз {int(I[I > 7][i])}: p = {np.round(Pte[i], 4)}, предсказание {LETTERS[pte[i]]}")
        print("\n".join(g))

    Xf, Cf, If, Gf = read_dataset("foreign.txt")
    pf, Pf = predict(w, Xf)
    print(f"\nЧУЖОЙ ПОЧЕРК: accuracy = {accuracy(Cf, pf) * 100:.1f}% ({int(np.sum(pf == Cf))}/6)")
    print("матрица ошибок:\n", confusion_matrix(Cf, pf))
    for i in range(6):
        print(f"  {LETTERS[Cf[i]]} экз{If[i]}: p = {np.round(Pf[i], 4)} -> {LETTERS[pf[i]]}")
    Xa, Ca, Ia, Ga = read_dataset("alien.txt")
    pa, Pa = predict(w, Xa)
    print(f"ЧУЖАЯ БУКВА Н: класс {LETTERS[pa[0]]}, p = {np.round(Pa[0], 4)}, уверенность {Pa[0].max():.4f}")
 
    def reject_stats(P, C_):
        conf = P.max(1); pred = P.argmax(1)
        rej = conf < TAU
        would_ok = int(np.sum(rej & (pred == C_))) if C_ is not None else 0
        return int(rej.sum()), would_ok, len(P)
    print("\nПОРОГ ОТКАЗА tau = 0.6 (отвергнуто / из них были бы верны / всего):")
    print("  тест:", reject_stats(Pte, Cte))
    print("  чужой почерк:", reject_stats(Pf, Cf))
    print("  чужая буква:", reject_stats(Pa, None))
    for name, P_, C_ in [("тест", Pte, Cte), ("чужой почерк", Pf, Cf)]:
        kept = P_.max(1) >= TAU
        if kept.any():
            print(f"  accuracy среди принятых ({name}): {accuracy(C_[kept], P_.argmax(1)[kept]) * 100:.1f}% "
                  f"({int(kept.sum())} из {len(P_)})")
 
    cols = ["tab:blue", "tab:orange", "tab:green"]
    plt.figure(figsize=(8, 5))
    for (h, runs), col in zip(study.items(), cols):
        plt.semilogy(runs[0]["hist"], color=col, label=f"Es (train), H={h}")
        plt.semilogy(runs[0]["hist_te"], color=col, ls=":", label=f"CE (test), H={h}")
    plt.axhline(EE, color="r", ls="--", label="Ee = 0.05")
    plt.xlabel("эпоха"); plt.ylabel("кросс-энтропия (лог. шкала)"); plt.title(f"Сходимость\n{TAG}", fontsize=10)
    plt.legend(fontsize=8); plt.grid(alpha=.3, which="both"); plt.tight_layout(); plt.savefig("fig_conv.png", dpi=150); plt.close()
 
    plt.figure(figsize=(6, 4.5))
    for k, (h, runs) in enumerate(study.items()):
        plt.scatter([k] * 5, [r["epochs"] for r in runs], s=40, color=cols[k])
    plt.xticks(range(3), [f"H={h}" for h in study]); plt.ylabel("число эпох"); plt.yscale("log")
    plt.title(f"Эпохи по 5 запускам\n{TAG}", fontsize=10); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig("fig_epochs.png", dpi=150); plt.close()
 
    plt.figure(figsize=(4.8, 4.5))
    plt.imshow(Mte, cmap="Blues")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, Mte[i, j], ha="center", va="center", fontsize=14, color="k" if Mte[i, j] < Mte.max() * .6 else "w")
    plt.xticks(range(3), LETTERS); plt.yticks(range(3), LETTERS)
    plt.xlabel("предсказано"); plt.ylabel("истинный класс"); plt.title(f"Матрица ошибок (тест)\n{TAG}", fontsize=10)
    plt.tight_layout(); plt.savefig("fig_cm.png", dpi=150); plt.close()
 
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(M_MAX + 1), overall * 100, "k-o", lw=2, label="общая")
    for c in range(3):
        plt.plot(range(M_MAX + 1), per_cls[c] * 100, "--s", label=LETTERS[c], alpha=.8)
    plt.axhline(80, color="r", ls=":", label="80 %")
    plt.xlabel("число испорченных клеток m"); plt.ylabel("accuracy, %")
    plt.title(f"Устойчивость к шуму\n{TAG}", fontsize=10); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("fig_noise.png", dpi=150); plt.close()
 
    fig, axs = plt.subplots(1, H_VAR, figsize=(3.2 * H_VAR, 3.6))
    for j, ax in enumerate(axs):
        wj = w["W1"][j].reshape(ROWS, COLS); lim = np.abs(wj).max()
        im = ax.imshow(wj, cmap="seismic", vmin=-lim, vmax=lim)
        ax.set_title(f"нейрон {j + 1}\nv(П,Е,Т)=({w['W2'][0, j]:.1f}; {w['W2'][1, j]:.1f}; {w['W2'][2, j]:.1f})", fontsize=9)
        ax.set_xticks(range(COLS), range(1, COLS + 1)); ax.set_yticks(range(ROWS), range(1, ROWS + 1))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Карты весов скрытых нейронов\n{TAG}", fontsize=10); plt.tight_layout()
    plt.savefig("fig_weights.png", dpi=150); plt.close()
 
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.8))
    for ax, i in zip(axs, order):
        pi = Pte[i]
        show_grid(ax, Xte[i], f"{LETTERS[Cte[i]]} (экз {int(I[I > 7][i])}) → {LETTERS[pte[i]]}\np=({pi[0]:.2f}; {pi[1]:.2f}; {pi[2]:.2f})")
    fig.suptitle(f"Три самых неуверенных тестовых примера\n{TAG}", fontsize=10); plt.tight_layout()
    plt.savefig("fig_uncertain.png", dpi=150); plt.close()
 
    fig, axs = plt.subplots(3, 10, figsize=(14, 5.6))
    for c in range(3):
        for n in range(10):
            idx = np.where((C == c) & (I == n + 1))[0][0]
            show_grid(axs[c, n], X[idx], f"{LETTERS[c]} №{n + 1}")
    fig.suptitle(f"Датасет: 30 сеток 7×5 (№1–7 — обучающие, №8–10 — тестовые). {TAG}", fontsize=10)
    plt.tight_layout(); plt.savefig("fig_dataset.png", dpi=130); plt.close()
 
    np.savez("model.npz", **w)
 
    print("\n===== РЕЖИМ ФУНКЦИОНИРОВАНИЯ =====")
    def pick(c, n):
        return grids[int(np.where((C == c) & (I == n))[0][0])]
    for c, n in [(0, 2), (1, 5), (2, 7)]:
        run_mode(w, pick(c, n), f"обучающая: {LETTERS[c]} №{n}", true_cls=c)
    for c, n in [(0, 9), (1, 8), (2, 10)]:
        run_mode(w, pick(c, n), f"тестовая: {LETTERS[c]} №{n}", true_cls=c)
    rng = np.random.default_rng(7)
    for c, n, m in [(0, 1, 4), (2, 1, 7)]:
        base = grid_to_vec(pick(c, n))
        z = add_noise(base, m, rng)
        rows = ["".join("#" if v else "." for v in z[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        run_mode(w, rows, f"искажённая: {LETTERS[c]} №{n}", true_cls=c, changed=m)
    run_mode(w, Ga[0], "чужая буква Н", true_cls=None)
    run_mode(w, Gf[1], "чужой почерк: П №2", true_cls=int(Cf[1]))
    run_mode(w, Gf[4], "чужой почерк: Т №1", true_cls=int(Cf[4]))
 
    if "--interactive" in sys.argv:
        interactive_mode(w)
