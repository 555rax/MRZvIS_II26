import os
import numpy as np
import matplotlib.pyplot as plt


SURNAME    = "Кушнеревич"
INITIALS   = "Е.А."
GROUP      = "ИИ-26"
VARIANT_NO = 8

LETTERS = ["К", "У", "Ш"]

H_VAR = 6
ETA   = 0.1
M_MAX = 8

EE         = 0.05
MAX_EPOCHS = 5000
TAU        = 0.6
ES_KIND    = "mse"

BASE_SEED  = 1
N_IN, N_OUT = 35, 3


def read_dataset(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    X, cls, inst = [], [], []
    for k, i in enumerate(range(0, len(lines), 8), start=1):
        head = lines[i].replace("@", " ", 1).split() if lines[i].startswith("@") else []
        if len(head) != 2 or not (head[0].isdigit() and head[1].isdigit()):
            raise ValueError(f"Блок №{k}: ожидался заголовок вида '@ 1 3', а найдено {lines[i]!r}")
        rows = lines[i + 1:i + 8]
        if len(rows) != 7 or any(len(r) != 5 or set(r) - set("#.") for r in rows):
            raise ValueError(f"Блок №{k} ({lines[i]}): нужно ровно 7 строк по 5 символов '#' и '.'")
        X.append([1.0 if ch == "#" else 0.0 for r in rows for ch in r])
        cls.append(int(head[0]))
        inst.append(int(head[1]))
    return np.array(X), np.array(cls) - 1, np.array(inst)

def one_hot(labels, n_classes=N_OUT):
    Y = np.zeros((len(labels), n_classes))
    Y[np.arange(len(labels)), labels] = 1.0
    return Y

def split_dataset(X, y, inst):
    is_train = inst <= 7
    return X[is_train], y[is_train], X[~is_train], y[~is_train], is_train

def grid_text(x):
    return "\n".join("".join("█" if v else "·" for v in x[r * 5:(r + 1) * 5]) for r in range(7))

def hamming_matrix(X):
    return (X[:, None, :] != X[None, :, :]).sum(axis=2)

def dataset_quality(X, y, n_classes=N_OUT):
    D = hamming_matrix(X)
    iu = np.triu_indices(len(X), k=1)
    same = (y[:, None] == y[None, :])[iu]
    per_class = []
    for c in range(n_classes):
        sel = np.where(y == c)[0]
        sub = D[np.ix_(sel, sel)]
        per_class.append(sub[np.triu_indices(len(sel), k=1)].mean())
    return {"nfill": int(X.sum()), "intra": D[iu][same].mean(),
            "inter": D[iu][~same].mean(), "per_class": per_class}

def print_quality(q):
    print(f"Nfill (всего закрашенных клеток) = {q['nfill']}")
    print(f"Среднее расстояние Хэмминга внутри классов : {q['intra']:.2f}")
    print(f"Среднее расстояние Хэмминга между классами : {q['inter']:.2f}")
    for c, m in enumerate(q["per_class"]):
        note = "" if m >= 3 else "   <-- МЕНЬШЕ 3: экземпляры слишком похожи, перепиши их!"
        print(f"  класс {LETTERS[c]}: внутри {m:.2f}{note}")
    if q["intra"] < 3:
        print("ВНИМАНИЕ: общее внутриклассовое среднее < 3 — датасет нужно переделать.")


def init_params(n_h, seed, n_in=N_IN, n_out=N_OUT):
    rng = np.random.default_rng(seed)
    a1, a2 = np.sqrt(6 / (n_in + n_h)), np.sqrt(6 / (n_h + n_out))
    return {"W1": rng.uniform(-a1, a1, (n_h, n_in)),  "T1": np.zeros(n_h),
            "W2": rng.uniform(-a2, a2, (n_out, n_h)), "T2": np.zeros(n_out)}

def count_params(P):
    return sum(v.size for v in P.values())

def copy_params(P):
    return {k: v.copy() for k, v in P.items()}

def sigmoid(s):
    return 1.0 / (1.0 + np.exp(-np.clip(s, -60, 60)))

def softmax(s):
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)

def forward(P, X):
    h = sigmoid(X @ P["W1"].T - P["T1"])
    p = softmax(h @ P["W2"].T - P["T2"])
    return h, p

def predict_proba(P, X):
    return forward(P, X)[1]

def predict(P, X):
    return predict_proba(P, X).argmax(axis=-1)

def cross_entropy(P, X, Y):
    p = predict_proba(P, X)
    return float(-np.mean(np.sum(Y * np.log(p + 1e-12), axis=1)))

def squared_error(P, X, Y):
    p = predict_proba(P, X)
    return float(np.mean(0.5 * np.sum((p - Y) ** 2, axis=1)))

def epoch_error(P, X, Y):
    return squared_error(P, X, Y) if ES_KIND == "mse" else cross_entropy(P, X, Y)

def backward(P, x, y):
    h, p = forward(P, x)
    d2 = p - y
    d1 = (P["W2"].T @ d2) * h * (1 - h)
    return {"W1": np.outer(d1, x), "T1": -d1, "W2": np.outer(d2, h), "T2": -d2}

def sgd_step(P, x, y, eta):
    g = backward(P, x, y)
    for k in P:
        P[k] -= eta * g[k]

def train(P, X, Y, seed, eta=ETA, ee=EE, max_epochs=MAX_EPOCHS):
    rng = np.random.default_rng(seed + 1000)
    hist = {"Es": [], "CE": []}
    for ep in range(1, max_epochs + 1):
        for i in rng.permutation(len(X)):
            sgd_step(P, X[i], Y[i], eta)
        es = epoch_error(P, X, Y)
        hist["Es"].append(es)
        hist["CE"].append(cross_entropy(P, X, Y))
        if es <= ee:
            return ep, True, hist
    return max_epochs, False, hist


def accuracy(P, X, y):
    return float(np.mean(predict(P, X) == y))

def confusion_matrix(y_true, y_pred, n_classes=N_OUT):
    M = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t, p] += 1
    return M

def print_confusion(M, title):
    print(title)
    print("       " + " ".join(f"{l:>4}" for l in LETTERS) + "  <- ответ сети")
    for i, l in enumerate(LETTERS):
        print(f"  {l:>3}  " + " ".join(f"{M[i, j]:>4}" for j in range(len(LETTERS))))

def confidence_stats(P, X, y):
    probs = predict_proba(P, X)
    conf = probs.max(axis=1)
    ok = probs.argmax(axis=1) == y
    return (conf[ok].mean() if ok.any() else None,
            conf[~ok].mean() if (~ok).any() else None)

def print_table(headers, rows):
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(v)) for v in col) for col in cols]
    line = lambda r: " | ".join(str(v).ljust(w) for v, w in zip(r, widths))
    print(line(headers)); print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


def add_noise(x, m, rng):
    z = x.copy()
    idx = rng.choice(len(x), size=m, replace=False)
    z[idx] = 1 - z[idx]
    return z

def noise_curve(P, X, y, m_max=M_MAX, n_rep=50, seed=0, n_classes=N_OUT):
    rng = np.random.default_rng(seed)
    acc_all = np.zeros(m_max + 1)
    acc_cls = np.zeros((m_max + 1, n_classes))
    for m in range(m_max + 1):
        per_sample = np.zeros(len(X))
        for i, x in enumerate(X):
            Xn = np.array([add_noise(x, m, rng) for _ in range(n_rep)])
            per_sample[i] = np.mean(predict(P, Xn) == y[i])
        acc_all[m] = per_sample.mean()
        for c in range(n_classes):
            acc_cls[m, c] = per_sample[y == c].mean()
    return acc_all, acc_cls

def single_loss(P, x, y):
    _, p = forward(P, x)
    return -np.sum(y * np.log(p))

def numeric_gradient(P, x, y, eps=1e-5):
    grads = {}
    for k, A in P.items():
        g = np.zeros_like(A)
        for idx in np.ndindex(A.shape):
            old = A[idx]
            A[idx] = old + eps; Lp = single_loss(P, x, y)
            A[idx] = old - eps; Lm = single_loss(P, x, y)
            A[idx] = old
            g[idx] = (Lp - Lm) / (2 * eps)
        grads[k] = g
    return grads

def check_gradients(P, x, y):
    ana, num = backward(P, x, y), numeric_gradient(P, x, y)
    print("Проверка градиентов (аналитический vs численный):")
    for k in P:
        abs_err = np.abs(ana[k] - num[k]).max()
        rel_err = abs_err / (np.abs(num[k]).max() + 1e-12)
        print(f"  {k}: макс. |разница| = {abs_err:.2e}, относительная = {rel_err:.2e}   {'OK' if rel_err < 1e-6 else 'ОШИБКА!'}")

def manual_step(P, x, y, eta):
    n_h, n_in = P["W1"].shape
    n_o = P["W2"].shape[0]
    S1 = [sum(P["W1"][j][i] * x[i] for i in range(n_in)) - P["T1"][j] for j in range(n_h)]
    h  = [1 / (1 + np.exp(-s)) for s in S1]
    S2 = [sum(P["W2"][k][j] * h[j] for j in range(n_h)) - P["T2"][k] for k in range(n_o)]
    mx = max(S2); ex = [np.exp(s - mx) for s in S2]; tot = sum(ex)
    p  = [e / tot for e in ex]
    d2 = [p[k] - y[k] for k in range(n_o)]
    d1 = [sum(d2[k] * P["W2"][k][j] for k in range(n_o)) * h[j] * (1 - h[j]) for j in range(n_h)]
    new = {
        "W2": np.array([[P["W2"][k][j] - eta * d2[k] * h[j] for j in range(n_h)] for k in range(n_o)]),
        "T2": np.array([P["T2"][k] + eta * d2[k] for k in range(n_o)]),
        "W1": np.array([[P["W1"][j][i] - eta * d1[j] * x[i] for i in range(n_in)] for j in range(n_h)]),
        "T1": np.array([P["T1"][j] + eta * d1[j] for j in range(n_h)]),
    }
    return h, p, d2, d1, new

def rows_to_vector(rows):
    rows = [r.strip() for r in rows]
    if len(rows) != 7 or any(len(r) != 5 or set(r) - set("#.") for r in rows):
        return None
    return np.array([1.0 if ch == "#" else 0.0 for r in rows for ch in r])

def vec_to_rows(x):
    return ["".join("#" if v else "." for v in x[r * 5:(r + 1) * 5]) for r in range(7)]

def classify(P, rows, title=""):
    x = rows_to_vector(rows)
    if x is None:
        print("Ошибка ввода: нужно 7 строк по 5 символов '#' и '.'"); return
    p = predict_proba(P, x)
    k = int(p.argmax())
    print(f"=== {title} ===")
    print("Введённая сетка:")
    print(grid_text(x))
    for c in range(N_OUT):
        print(f"  p{c+1} ({LETTERS[c]}) = {p[c]:.4f}")
    print(f"Класс: {LETTERS[k]}")
    print(f"Уверенность: {p[k]:.4f}")
    print(f"Решение: {LETTERS[k]}" if p[k] >= TAU else f"Решение: не распознано (max p = {p[k]:.4f} < τ = {TAU})")
    print()

def reject_report(P, name, X, y=None):
    probs = predict_proba(P, X)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    rej = conf < TAU
    s = f"{name:<14}: всего {len(X)}, отвергнуто {int(rej.sum())}"
    if y is None:
        s += " (буквы нет среди классов — верно классифицировать нельзя)"
    else:
        s += f", из них верно были бы классифицированы: {int((pred[rej] == y[rej]).sum())}"
    print(s)


DATASET_PATH = r"C:\Users\User\Desktop\News\dataset.txt"

X_all, y_all, inst_all = read_dataset(DATASET_PATH)

assert len(X_all) == 30, f"Нужно 30 объектов, а в файле {len(X_all)}"
for c in range(N_OUT):
    assert sorted(inst_all[y_all == c]) == list(range(1, 11)), f"У класса {c+1} должны быть экземпляры 1..10"

Xtr, ytr, Xte, yte, is_train = split_dataset(X_all, y_all, inst_all)
Ytr, Yte = one_hot(ytr), one_hot(yte)

quality = dataset_quality(X_all, y_all)
NFILL = quality["nfill"]
TTL = f"{SURNAME}, вариант {VARIANT_NO}, Nfill={NFILL}"

print(f"Обучающая: {len(Xtr)} объектов, тестовая: {len(Xte)} объектов\n")
print_quality(quality)
print(f"\nПервый объект (класс {LETTERS[y_all[0]]}, экз. {inst_all[0]}) — сверь с листом:")
print(grid_text(X_all[0]))


x0, y0 = Xtr[0], Ytr[0]
P_chk = init_params(H_VAR, BASE_SEED)

check_gradients(P_chk, x0, y0)

h, p, d2, d1, new = manual_step(P_chk, x0, y0, ETA)
np.set_printoptions(precision=4, suppress=True)
print("\nШаг обучения вручную на примере: класс", LETTERS[ytr[0]], ", экземпляр 1")
print("  выходы скрытого слоя h =", np.array(h))
print("  вероятности p =", np.array(p), "(цель y =", y0, ")")
print("  ошибка выхода δ2 = p−y =", np.array(d2))
print("  ошибка скрытого δ1 =", np.array(d1))
print("  потеря до шага  CE = %.5f" % single_loss(P_chk, x0, y0))

P_copy = copy_params(P_chk)
sgd_step(P_copy, x0, y0, ETA)                                   
print(" макс. расхождение с sgd_step = %.1e" % max(np.abs(new[k] - P_copy[k]).max() for k in new))
print(" потеря после шага CE = %.5f (должна стать меньше)" % single_loss(new, x0, y0))


net = init_params(H_VAR, BASE_SEED)
epochs, reached, hist_base = train(net, Xtr, Ytr, BASE_SEED)

acc_tr = accuracy(net, Xtr, ytr)
acc_te = accuracy(net, Xte, yte)
print(f"Эпох: {epochs}  ({'критерий Es ≤ %.2f достигнут' % EE if reached else 'критерий НЕ достигнут, остановка по лимиту'})")
print(f"Итоговая Es = {hist_base['Es'][-1]:.4f},  CE = {hist_base['CE'][-1]:.4f}")
print(f"Accuracy: обучение = {acc_tr*100:.1f}%,  тест = {acc_te*100:.1f}%")

c_ok, c_bad = confidence_stats(net, Xte, yte)
print("Средняя уверенность на тесте: верные = " + (f"{c_ok:.4f}" if c_ok is not None else "нет") +
      ",  неверные = " + (f"{c_bad:.4f}" if c_bad is not None else "нет (ошибок нет)"))

cm_tr = confusion_matrix(ytr, predict(net, Xtr))
cm_te = confusion_matrix(yte, predict(net, Xte))
print()
print_confusion(cm_tr, "Матрица ошибок — ОБУЧЕНИЕ (строки: истина, столбцы: ответ сети)")
print()
print_confusion(cm_te, "Матрица ошибок — ТЕСТ")

fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm_te, cmap="Blues")
for i in range(N_OUT):
    for j in range(N_OUT):
        ax.text(j, i, cm_te[i, j], ha="center", va="center",
                color="white" if cm_te[i, j] > cm_te.max() / 2 else "black", fontsize=14)
ax.set_xticks(range(N_OUT)); ax.set_xticklabels(LETTERS)
ax.set_yticks(range(N_OUT)); ax.set_yticklabels(LETTERS)
ax.set_xlabel("Ответ сети"); ax.set_ylabel("Истинная буква")
ax.set_title("Матрица ошибок (тест)\n" + TTL, fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=150); plt.show()


H_LIST = [2, H_VAR, 2 * H_VAR]
SEEDS  = [BASE_SEED + k for k in range(5)]          

runs = {}                                           
for H in H_LIST:
    runs[H] = []
    for s in SEEDS:
        Pn = init_params(H, s)
        ep, okc, hist = train(Pn, Xtr, Ytr, s)
        runs[H].append((ep, okc, accuracy(Pn, Xte, yte), hist))
        print(f"H={H:2d} seed={s}: эпох={ep:4d}, критерий={'да' if okc else 'НЕТ'}, тест={runs[H][-1][2]*100:.1f}%")

headers = ["H", "параметров", "эпохи (среднее)", "эпохи (min–max)",
           "тест acc, % (среднее)", "тест acc, % (min–max)", "не достигли Ee (из 5)"]
table = []
for H in H_LIST:
    ep  = np.array([r[0] for r in runs[H]])
    acc = np.array([r[2] for r in runs[H]]) * 100
    bad = sum(1 for r in runs[H] if not r[1])
    table.append([H, count_params(init_params(H, 0)), f"{ep.mean():.1f}", f"{ep.min()}–{ep.max()}",
                  f"{acc.mean():.1f}", f"{acc.min():.1f}–{acc.max():.1f}", bad])
print()
print_table(headers, table)

colors = {H: c for H, c in zip(H_LIST, ["tab:blue", "tab:green", "tab:red"])}
plt.figure(figsize=(8, 5))
for H in H_LIST:
    hist = runs[H][0][3]                             
    plt.plot(range(1, len(hist["Es"]) + 1), hist["Es"], color=colors[H], label=f"Es, H={H}")
    plt.plot(range(1, len(hist["CE"]) + 1), hist["CE"], color=colors[H], ls="--", label=f"CE, H={H}")
plt.axhline(EE, color="gray", ls=":", label=f"Ee = {EE}")
plt.yscale("log"); plt.xlabel("Эпоха"); plt.ylabel("Ошибка (лог. шкала)")
plt.title("Сходимость обучения\n" + TTL, fontsize=11)
plt.legend(ncol=2, fontsize=8); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("fig_convergence.png", dpi=150); plt.show()

plt.figure(figsize=(6, 4.5))
rng_j = np.random.default_rng(0)
for xi, H in enumerate(H_LIST):
    ep  = np.array([r[0] for r in runs[H]])
    okc = np.array([r[1] for r in runs[H]])
    jit = rng_j.uniform(-0.08, 0.08, len(ep))
    plt.scatter(xi + jit[okc],  ep[okc],  color=colors[H], s=50)
    plt.scatter(xi + jit[~okc], ep[~okc], color=colors[H], s=70, marker="x")   # крестик — не достигли Ee
    plt.hlines(ep.mean(), xi - 0.2, xi + 0.2, color="black")
plt.xticks(range(3), [f"H={H}" for H in H_LIST])
plt.ylabel("Число эпох"); plt.grid(alpha=0.3)
plt.title("Разброс числа эпох (5 запусков; × — критерий не достигнут)\n" + TTL, fontsize=10)
plt.tight_layout(); plt.savefig("fig_epochs_scatter.png", dpi=150); plt.show()


acc_all, acc_cls = noise_curve(net, Xte, yte)

print("m | accuracy общая | " + " | ".join(LETTERS))
for m in range(M_MAX + 1):
    print(f"{m} | {acc_all[m]*100:6.1f}%       | " + " | ".join(f"{acc_cls[m, c]*100:5.1f}%" for c in range(N_OUT)))

below = np.where(acc_all < 0.8)[0]
if len(below):
    print(f"\nНаименьшее m, при котором accuracy < 80%:  m = {below[0]}")
else:
    print(f"\nДо m = {M_MAX} accuracy не опускается ниже 80%")

plt.figure(figsize=(7, 4.5))
plt.plot(range(M_MAX + 1), acc_all * 100, "k-o", lw=2.5, label="все классы")
for c in range(N_OUT):
    plt.plot(range(M_MAX + 1), acc_cls[:, c] * 100, "--s", label=f"класс {LETTERS[c]}")
plt.axhline(80, color="gray", ls=":")
plt.xlabel("Число испорченных клеток m"); plt.ylabel("Accuracy, %")
plt.ylim(0, 105); plt.grid(alpha=0.3); plt.legend()
plt.title("Устойчивость к шуму (тест, по 50 зашумлений)\n" + TTL, fontsize=10)
plt.tight_layout(); plt.savefig("fig_noise.png", dpi=150); plt.show()


W1, W2 = net["W1"], net["W2"]
vmax = np.abs(W1).max()                                          
n_h = W1.shape[0]
ncols = 3
nrows = int(np.ceil(n_h / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows))
axes = np.atleast_1d(axes).ravel()
for j in range(n_h):
    im = axes[j].imshow(W1[j].reshape(7, 5), cmap="bwr", vmin=-vmax, vmax=vmax)
    axes[j].set_title(f"нейрон {j+1}")
    axes[j].set_xticks(range(5)); axes[j].set_xticklabels(range(1, 6))
    axes[j].set_yticks(range(7)); axes[j].set_yticklabels(range(1, 8))
for k in range(n_h, len(axes)):
    axes[k].axis("off")
fig.suptitle("Карты весов скрытых нейронов (красный +, синий −)\n" + TTL, fontsize=11)
fig.colorbar(im, ax=axes.tolist(), fraction=0.025)
plt.savefig("fig_weight_maps.png", dpi=150, bbox_inches="tight"); plt.show()

mean_img = np.array([Xtr[ytr == c].mean(axis=0) for c in range(N_OUT)])   
distinct = mean_img - mean_img.mean(axis=0)                               
print("Нейрон | работает на букву (по знакам выходных весов) | карта похожа на | 3 клетки с наибольшим |вес|")
for j in range(n_h):
    rel = W2[:, j] - W2[:, j].mean()                  
    by_out  = LETTERS[int(rel.argmax())]
    against = LETTERS[int(rel.argmin())]
    by_map  = LETTERS[int((distinct @ W1[j]).argmax())]
    top = np.argsort(-np.abs(W1[j]))[:3]
    cells = ", ".join(f"({t // 5 + 1},{t % 5 + 1}){'+' if W1[j][t] > 0 else '−'}{abs(W1[j][t]):.2f}" for t in top)
    print(f"  {j+1}    | за {by_out}, против {against}  (веса на выход: {np.round(W2[:, j], 2)}) | {by_map} | {cells}")
print("(строка, столбец) нумеруются с 1")


probs_te = predict_proba(net, Xte)
conf = probs_te.max(axis=1)
worst = np.argsort(conf)[:3]                                      
test_idx = np.where(~is_train)[0]                                 

fig, axes = plt.subplots(1, 3, figsize=(10, 4.4))
for ax, i in zip(axes, worst):
    ax.imshow(Xte[i].reshape(7, 5), cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    pr = "  ".join(f"{LETTERS[c]}={probs_te[i, c]:.3f}" for c in range(N_OUT))
    ax.set_title(f"истина {LETTERS[yte[i]]} (экз. {inst_all[test_idx[i]]}), ответ {LETTERS[probs_te[i].argmax()]}\n{pr}", fontsize=9)
fig.suptitle("Три самых неуверенных тестовых примера\n" + TTL, fontsize=11)
plt.tight_layout(); plt.savefig("fig_uncertain.png", dpi=150); plt.show()

for i in worst:
    print(f"Истина {LETTERS[yte[i]]}, экземпляр №{inst_all[test_idx[i]]}; уверенность {conf[i]:.4f}")
    print(grid_text(Xte[i]))
    print("   p =", ", ".join(f"p({LETTERS[c]}) = {probs_te[i, c]:.4f}" for c in range(N_OUT)), "\n")