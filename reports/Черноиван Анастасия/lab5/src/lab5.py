import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURNAME, VARIANT = "Черноиван", 8
H_VAR, ETA, MMAX = 6, 0.10, 8
EE, MAX_EPOCH, TAU = 0.05, 5000, 0.6
LETTERS = ["Ч", "Е", "Р"]


def read_dataset(path):
    X, cls, num = [], [], []
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    for i in range(0, len(lines), 8):
        head = lines[i].split()
        c, n = int(head[2]), int(head[-1])
        rows = lines[i + 1:i + 8]
        v = [1 if ch == "#" else 0 for r in rows for ch in r]
        X.append(v); cls.append(c - 1); num.append(n)
    return np.array(X, float), np.array(cls), np.array(num)


def one_hot(c, k=3):
    Y = np.zeros((len(c), k))
    Y[np.arange(len(c)), c] = 1
    return Y


def split(X, c, num):
    tr, te = num <= 7, num >= 8
    return X[tr], c[tr], X[te], c[te], num[te]


def quality(X, c):
    nfill = int(X.sum())
    intra, inter = [], []
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            d = np.sum(X[i] != X[j])
            (intra if c[i] == c[j] else inter).append(d)
    per = [np.mean([np.sum(X[i] != X[j]) for i in range(len(X)) for j in range(i + 1, len(X))
                    if c[i] == c[j] == k]) for k in range(3)]
    return nfill, np.mean(intra), np.mean(inter), per


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def init(H, rng):
    return {"V": rng.uniform(-0.5, 0.5, (H, 35)), "b": np.zeros(H),
            "W": rng.uniform(-0.5, 0.5, (3, H)), "d": np.zeros(3)}


def forward(P, x):
    h = sigmoid(P["V"] @ x + P["b"])
    p = softmax(P["W"] @ h + P["d"])
    return h, p


def ce(p, y):
    return -np.sum(y * np.log(p + 1e-12))


def backward(P, x, y, h, p):
    dz = p - y
    dh = (P["W"].T @ dz) * h * (1 - h)
    return {"W": np.outer(dz, h), "d": dz, "V": np.outer(dh, x), "b": dh}


def train(X, c, H, seed, eta=ETA):
    rng = np.random.default_rng(seed)
    P = init(H, rng); Y = one_hot(c)
    hist_es, hist_ce = [], []
    for ep in range(1, MAX_EPOCH + 1):
        es = 0
        for i in rng.permutation(len(X)):
            h, p = forward(P, X[i])
            es += ce(p, Y[i])
            g = backward(P, X[i], Y[i], h, p)
            for k in P:
                P[k] -= eta * g[k]
        es /= len(X)
        hist_es.append(es)
        hist_ce.append(np.mean([ce(forward(P, X[i])[1], Y[i]) for i in range(len(X))]))
        if es <= EE:
            break
    return P, ep, hist_es, hist_ce, es <= EE


def n_params(H):
    return 35 * H + H + 3 * H + 3


def predict_all(P, X):
    return np.array([forward(P, x)[1] for x in X])


def accuracy(P, X, c):
    return np.mean(predict_all(P, X).argmax(1) == c)


def confusion(P, X, c):
    M = np.zeros((3, 3), int)
    for t, a in zip(c, predict_all(P, X).argmax(1)):
        M[t, a] += 1
    return M


def confidence_split(P, X, c):
    pr = predict_all(P, X)
    ok = pr.argmax(1) == c
    return (pr.max(1)[ok].mean() if ok.any() else np.nan,
            pr.max(1)[~ok].mean() if (~ok).any() else np.nan)


def add_noise(x, m, rng):
    z = x.copy()
    idx = rng.choice(35, m, replace=False)
    z[idx] = 1 - z[idx]
    return z


def noise_test(P, X, c, mmax, reps=50, seed=0):
    rng = np.random.default_rng(seed)
    tot, per = [], []
    for m in range(mmax + 1):
        hit = np.zeros(len(X))
        for i in range(len(X)):
            hit[i] = np.mean([forward(P, add_noise(X[i], m, rng))[1].argmax() == c[i]
                              for _ in range(reps)])
        tot.append(hit.mean())
        per.append([hit[c == k].mean() for k in range(3)])
    return np.array(tot), np.array(per)


def numeric_grad_check(P, x, y, eps=1e-5):
    h, p = forward(P, x)
    g = backward(P, x, y, h, p)
    worst = 0
    for k in P:
        it = np.nditer(P[k], flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = P[k][idx]
            P[k][idx] = old + eps; e1 = ce(forward(P, x)[1], y)
            P[k][idx] = old - eps; e2 = ce(forward(P, x)[1], y)
            P[k][idx] = old
            num = (e1 - e2) / (2 * eps)
            worst = max(worst, abs(num - g[k][idx]))
    return worst


def show_grid(x):
    return "\n".join("".join("#" if x[r * 5 + k] else "." for k in range(5)) for r in range(7))


def classify(P, x, note=""):
    _, p = forward(P, x)
    k = p.argmax()
    dec = LETTERS[k] if p[k] >= TAU else "не распознано"
    print(f"{note}\n{show_grid(x)}\np = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}); "
          f"класс {LETTERS[k]}; уверенность {p[k]:.4f}; решение: {dec}\n")
    return k, p[k]


def parse_input(lines):
    return np.array([1.0 if ch == "#" else 0.0 for r in lines for ch in r])


def interactive(P):
    print("=== РУЧНОЙ ВВОД ===")
    while True:
        print("Введите 7 строк по 5 символов (# и .), пустая строка – выход:")
        rows = []
        while len(rows) < 7:
            try:
                r = input().strip()
            except EOFError:
                return
            if not r:
                return
            if len(r) != 5 or set(r) - {"#", "."}:
                print("нужно ровно 5 символов # или ., повторите строку")
                continue
            rows.append(r)
        classify(P, parse_input(rows), "[ввод пользователя]")


def title(nfill, extra=""):
    return f"{SURNAME}, вариант {VARIANT}, Nfill={nfill} {extra}".strip()


def main():
    X, c, num = read_dataset("dataset.txt")
    Xtr, ctr, Xte, cte, nte = split(X, c, num)
    nfill, intra, inter, per = quality(X, c)
    print(f"Nfill={nfill}; Хэмминг внутри класса={intra:.2f} (по классам {np.round(per,2)}), между классами={inter:.2f}")
    if intra < 3:
        print("ВНИМАНИЕ: экземпляры слишком похожи!")


    P, ep, es_h, ce_h, ok = train(Xtr, ctr, H_VAR, seed=1)
    print(f"\nБАЗА H={H_VAR}: эпох={ep}, Es={es_h[-1]:.4f}, остановка по критерию={ok}")
    print(f"acc train={accuracy(P,Xtr,ctr):.3f}, acc test={accuracy(P,Xte,cte):.3f}")
    print("матрица ошибок train:\n", confusion(P, Xtr, ctr))
    Mte = confusion(P, Xte, cte)
    print("матрица ошибок test:\n", Mte)
    cok, cbad = confidence_split(P, Xte, cte)
    print(f"уверенность: верные={cok:.3f}, ошибочные={cbad}")
    pr_te = predict_all(P, Xte)
    wrong = [(LETTERS[cte[i]], int(nte[i]), LETTERS[pr_te[i].argmax()]) for i in range(len(Xte)) if pr_te[i].argmax() != cte[i]]
    print("ошибочные тестовые (истина, №, предсказано):", wrong)
    print("проверка градиента, max|num-ан| =", f"{numeric_grad_check(P, Xtr[0], one_hot(ctr)[0]):.2e}")


    Hs = [2, H_VAR, 2 * H_VAR]
    res, curves = {}, {}
    print("\nИССЛЕДОВАНИЕ H (5 запусков, seed=1..5):")
    for H in Hs:
        eps_, accs, fail = [], [], 0
        for s in range(1, 6):
            Ph, e, eh, ch, okh = train(Xtr, ctr, H, seed=s)
            eps_.append(e); accs.append(accuracy(Ph, Xte, cte)); fail += (not okh)
            if s == 1:
                curves[H] = (eh, ch)
        res[H] = eps_
        print(f"H={H:2d}: эпохи {np.mean(eps_):.0f} [{min(eps_)}-{max(eps_)}], acc test {np.mean(accs):.3f} "
              f"[{min(accs):.3f}-{max(accs):.3f}], не достигли критерия {fail}, параметров {n_params(H)}")


    tot, per_c = noise_test(P, Xte, cte, MMAX)
    below = [m for m in range(MMAX + 1) if tot[m] < 0.8]
    print("\nШУМ accuracy(m):", np.round(tot, 3), "\nпо классам:\n", np.round(per_c, 3).T)
    print("наименьшее m, при котором acc<80%:", below[0] if below else "нет")


    print("\nНЕЙРОНЫ (строка,столбец 1..7,1..5; w_kj выходного слоя):")
    for j in range(H_VAR):
        w = P["V"][j]
        top = np.argsort(-np.abs(w))[:3]
        cells = ", ".join(f"({t//5+1},{t%5+1}):{w[t]:+.2f}" for t in top)
        print(f"нейрон {j+1}: выход Ч/Е/Р = {np.round(P['W'][:,j],2)}; макс.|w|: {cells}")


    conf = pr_te.max(1)
    worst3 = np.argsort(conf)[:3]
    print("\n3 САМЫХ НЕУВЕРЕННЫХ ТЕСТОВЫХ:")
    for i in worst3:
        print(f"{LETTERS[cte[i]]} №{nte[i]}\n{show_grid(Xte[i])}\np={np.round(pr_te[i],4)}")


    Xc, cc, _ = read_dataset("cross.txt")
    prc = predict_all(P, Xc)
    print(f"\nЧУЖОЙ ПОЧЕРК: acc={np.mean(prc.argmax(1)==cc):.3f}, предсказано={[LETTERS[k] for k in prc.argmax(1)]}, "
          f"истина={[LETTERS[k] for k in cc]}")
    Xf, _, _ = read_dataset("foreign.txt")
    pf = forward(P, Xf[0])[1]
    print(f"ЧУЖАЯ БУКВА 'Н': класс {LETTERS[pf.argmax()]}, p={np.round(pf,4)}")


    print("\nПОРОГ ОТКАЗА tau=0.6:")
    for name, pr_, cl in [("тест", pr_te, cte), ("чужой почерк", prc, cc)]:
        rej = pr_.max(1) < TAU
        print(f"{name}: отвергнуто {rej.sum()} из {len(pr_)}, из них верных {(rej & (pr_.argmax(1)==cl)).sum()}")
    print(f"чужая буква: отвергнута={pf.max() < TAU}")


    t = title(nfill)
    plt.figure(figsize=(7, 4))
    for H, col in zip(Hs, ["tab:red", "tab:green", "tab:blue"]):
        plt.semilogy(curves[H][0], color=col, label=f"Es, H={H}")
        plt.semilogy(curves[H][1], color=col, ls="--", label=f"CE, H={H}")
    plt.xlabel("эпоха"); plt.ylabel("ошибка"); plt.title(t); plt.legend(fontsize=7); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("g1_convergence.png", dpi=130); plt.close()

    plt.figure(figsize=(5, 4))
    for k, H in enumerate(Hs):
        plt.scatter([k] * 5, res[H])
    plt.xticks(range(3), [f"H={h}" for h in Hs]); plt.ylabel("число эпох"); plt.title(t, fontsize=9)
    plt.tight_layout(); plt.savefig("g2_epochs.png", dpi=130); plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow(Mte, cmap="Blues")
    for a in range(3):
        for b in range(3):
            plt.text(b, a, Mte[a, b], ha="center", va="center", fontsize=14)
    plt.xticks(range(3), LETTERS); plt.yticks(range(3), LETTERS)
    plt.xlabel("предсказано"); plt.ylabel("истина"); plt.title(t, fontsize=8)
    plt.tight_layout(); plt.savefig("g3_confusion.png", dpi=130); plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(range(MMAX + 1), tot * 100, "k-o", label="общая")
    for k in range(3):
        plt.plot(range(MMAX + 1), per_c[:, k] * 100, "--", label=LETTERS[k])
    plt.axhline(80, color="r", lw=.8); plt.xlabel("m (число испорченных клеток)"); plt.ylabel("accuracy, %")
    plt.title(t); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig("g4_noise.png", dpi=130); plt.close()

    fig, ax = plt.subplots(2, 3, figsize=(8, 5.5))
    for j, a in enumerate(ax.ravel()):
        w = P["V"][j].reshape(7, 5); lim = np.abs(w).max()
        a.imshow(w, cmap="bwr", vmin=-lim, vmax=lim); a.set_title(f"нейрон {j+1}", fontsize=9)
        a.set_xticks(range(5), range(1, 6)); a.set_yticks(range(7), range(1, 8))
    fig.suptitle(t, fontsize=9); plt.tight_layout(); plt.savefig("g5_weights.png", dpi=130); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(8, 3.6))
    for a, i in zip(ax, worst3):
        a.imshow(Xte[i].reshape(7, 5), cmap="Greys")
        a.set_title(f"{LETTERS[cte[i]]} №{nte[i]}\np={np.round(pr_te[i],2)}", fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(t, fontsize=9); plt.tight_layout(); plt.savefig("g6_uncertain.png", dpi=130); plt.close()


    print("\n=== РЕЖИМ ФУНКЦИОНИРОВАНИЯ ===")
    for i in [0, 8, 15]:
        classify(P, Xtr[i], f"[обучающий, {LETTERS[ctr[i]]} №{i % 7 + 1}]")
    for i in [0, 4, 8]:
        classify(P, Xte[i], f"[тестовый, {LETTERS[cte[i]]} №{nte[i]}]")
    rng = np.random.default_rng(7)
    for i, m in [(1, 3), (4, 6)]:
        classify(P, add_noise(Xte[i], m, rng), f"[искажён: {LETTERS[cte[i]]} №{nte[i]}, изменено {m} клеток]")
    classify(P, Xf[0], "[чужая буква Н]")
    classify(P, Xc[2], "[чужой почерк, Е]")
    classify(P, Xc[4], "[чужой почерк, Р]")

    interactive(P)

if __name__ == "__main__":
    main()