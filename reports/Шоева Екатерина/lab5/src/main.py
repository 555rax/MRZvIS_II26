import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

SURNAME = "Шоева"
LETTERS = ["Ш", "О", "Е"]
VARIANT = 11
H_VAR = 5
LEARNING_RATE = 0.20
M_MAX = 6
EE = 0.05
MAX_EPOCHS = 5000
TAU = 0.6
N_NOISE = 50
SEEDS = [1, 2, 3, 4, 5]
BASE_SEED = 1
NOISE_SEED = 12345
DEMO_SEED = 7

ROWS, COLS = 7, 5
N_INPUT = ROWS * COLS
N_CLASSES = 3
N_PER_CLASS = 10
TRAIN_LAST_INSTANCE = 7
H_LIST = [2, H_VAR, 2 * H_VAR]
EPS_LOG = 1e-12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, "dataset.txt")
FOREIGN_HAND_FILE = os.path.join(BASE_DIR, "foreign_hand.txt")
FOREIGN_LETTER_FILE = os.path.join(BASE_DIR, "foreign_letter.txt")


def rows_to_vector(rows):
    return np.array([1.0 if ch == "#" else 0.0 for row in rows for ch in row])


def vector_to_rows(x):
    return ["".join("#" if x[r * COLS + c] > 0.5 else "." for c in range(COLS)) for r in range(ROWS)]


def build_object(header, rows):
    if len(rows) != ROWS:
        raise ValueError(f"Объект '{header}': нужно {ROWS} строк по {COLS} символов, найдено {len(rows)}")
    nums = [int(t) for t in re.findall(r"\d+", header)]
    return {
        "header": header,
        "cls": nums[0] if len(nums) > 0 else 0,
        "inst": nums[1] if len(nums) > 1 else 0,
        "rows": rows,
        "x": rows_to_vector(rows),
    }


def read_objects(path):
    with open(path, encoding="utf-8-sig") as f:
        lines = f.read().splitlines()
    objects = []
    header = None
    rows = []
    for number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[#.]+", line):
            if len(line) != COLS:
                raise ValueError(f"{os.path.basename(path)}, строка {number}: в сетке должно быть {COLS} символов, найдено {len(line)}")
            rows.append(line)
        else:
            if header is not None:
                objects.append(build_object(header, rows))
            header = line
            rows = []
    if header is not None:
        objects.append(build_object(header, rows))
    return objects


def read_optional(path):
    if not os.path.exists(path):
        return None
    return read_objects(path)


def validate_dataset(objects):
    expected = {(c, i) for c in range(1, N_CLASSES + 1) for i in range(1, N_PER_CLASS + 1)}
    found = [(o["cls"], o["inst"]) for o in objects]
    if len(objects) != N_CLASSES * N_PER_CLASS:
        raise ValueError(f"В датасете {len(objects)} объектов, ожидается {N_CLASSES * N_PER_CLASS}")
    if set(found) != expected or len(set(found)) != len(found):
        missing = sorted(expected - set(found))
        raise ValueError(f"Заголовки не соответствуют схеме класс 1-3 / экземпляр 1-10. Не найдены: {missing}")


def one_hot(index, n=N_CLASSES):
    y = np.zeros(n)
    y[index] = 1.0
    return y


def split_dataset(objects):
    train = sorted([o for o in objects if o["inst"] <= TRAIN_LAST_INSTANCE], key=lambda o: (o["cls"], o["inst"]))
    test = sorted([o for o in objects if o["inst"] > TRAIN_LAST_INSTANCE], key=lambda o: (o["cls"], o["inst"]))
    return train, test


def to_arrays(objects):
    X = np.array([o["x"] for o in objects])
    idx = np.array([o["cls"] - 1 for o in objects])
    Y = np.array([one_hot(k) for k in idx])
    return X, Y, idx


def hamming(a, b):
    return int(np.sum(a != b))


def dataset_quality(objects):
    n_fill = int(sum(o["x"].sum() for o in objects))
    groups = {c: [o["x"] for o in objects if o["cls"] == c] for c in range(1, N_CLASSES + 1)}
    intra = {}
    intra_all = []
    for c, vecs in groups.items():
        d = [hamming(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        intra[c] = float(np.mean(d))
        intra_all += d
    inter = {}
    inter_all = []
    for c1 in range(1, N_CLASSES + 1):
        for c2 in range(c1 + 1, N_CLASSES + 1):
            d = [hamming(a, b) for a in groups[c1] for b in groups[c2]]
            inter[(c1, c2)] = float(np.mean(d))
            inter_all += d
    return {
        "n_fill": n_fill,
        "intra": intra,
        "inter": inter,
        "intra_mean": float(np.mean(intra_all)),
        "inter_mean": float(np.mean(inter_all)),
    }


def print_quality(q):
    print(f"Общее число закрашенных клеток N_fill = {q['n_fill']}")
    print("Среднее попарное расстояние Хэмминга внутри классов:")
    for c, v in q["intra"].items():
        flag = "   <-- меньше 3, экземпляры слишком похожи, перепишите" if v < 3 else ""
        print(f"  класс {c} ({LETTERS[c - 1]}): {v:.2f}{flag}")
    print(f"  в среднем по классам: {q['intra_mean']:.2f}")
    print("Среднее расстояние Хэмминга между разными классами:")
    for (c1, c2), v in q["inter"].items():
        print(f"  {LETTERS[c1 - 1]} - {LETTERS[c2 - 1]}: {v:.2f}")
    print(f"  в среднем между классами: {q['inter_mean']:.2f}")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class MLP:
    def __init__(self, n_hidden, lr=LEARNING_RATE, seed=0):
        self.H = n_hidden
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.uniform(-0.5, 0.5, size=(n_hidden, N_INPUT))
        self.b1 = self.rng.uniform(-0.5, 0.5, size=n_hidden)
        self.W2 = self.rng.uniform(-0.5, 0.5, size=(N_CLASSES, n_hidden))
        self.b2 = self.rng.uniform(-0.5, 0.5, size=N_CLASSES)

    def n_params(self):
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def forward(self, x):
        h = sigmoid(self.W1 @ x + self.b1)
        p = softmax(self.W2 @ h + self.b2)
        return h, p

    def predict_proba(self, x):
        return self.forward(np.asarray(x, dtype=float))[1]

    def predict_proba_batch(self, X):
        Hh = sigmoid(X @ self.W1.T + self.b1)
        return softmax(Hh @ self.W2.T + self.b2)

    def loss(self, x, y):
        p = self.forward(x)[1]
        return float(-np.sum(y * np.log(p + EPS_LOG)))

    def analytic_gradient(self, x, y):
        h, p = self.forward(x)
        d_out = p - y
        d_hidden = (self.W2.T @ d_out) * h * (1.0 - h)
        return {
            "W1": np.outer(d_hidden, x),
            "b1": d_hidden,
            "W2": np.outer(d_out, h),
            "b2": d_out,
        }

    def numerical_gradient(self, x, y, eps=1e-5):
        grads = {}
        for name in ("W1", "b1", "W2", "b2"):
            arr = getattr(self, name)
            g = np.zeros_like(arr)
            for idx in np.ndindex(arr.shape):
                old = arr[idx]
                arr[idx] = old + eps
                plus = self.loss(x, y)
                arr[idx] = old - eps
                minus = self.loss(x, y)
                arr[idx] = old
                g[idx] = (plus - minus) / (2.0 * eps)
            grads[name] = g
        return grads

    def train_step(self, x, y):
        g = self.analytic_gradient(x, y)
        self.W1 -= self.lr * g["W1"]
        self.b1 -= self.lr * g["b1"]
        self.W2 -= self.lr * g["W2"]
        self.b2 -= self.lr * g["b2"]

    def epoch_metrics(self, X, Y):
        P = self.predict_proba_batch(X)
        es = float(np.mean(np.sum((Y - P) ** 2, axis=1)))
        ce = float(np.mean(-np.sum(Y * np.log(P + EPS_LOG), axis=1)))
        return es, ce

    def train(self, X, Y, ee=EE, max_epochs=MAX_EPOCHS):
        hist_es, hist_ce = [], []
        es = ce = float("inf")
        for epoch in range(1, max_epochs + 1):
            for i in self.rng.permutation(len(X)):
                self.train_step(X[i], Y[i])
            es, ce = self.epoch_metrics(X, Y)
            hist_es.append(es)
            hist_ce.append(ce)
            if es <= ee:
                return {"epochs": epoch, "converged": True, "es": es, "ce": ce, "hist_es": hist_es, "hist_ce": hist_ce}
        return {"epochs": max_epochs, "converged": False, "es": es, "ce": ce, "hist_es": hist_es, "hist_ce": hist_ce}


def gradient_check(model, X, Y, n_examples=3):
    worst_abs, worst_rel = 0.0, 0.0
    for i in range(min(n_examples, len(X))):
        ga = model.analytic_gradient(X[i], Y[i])
        gn = model.numerical_gradient(X[i], Y[i])
        for name in ga:
            diff = np.abs(ga[name] - gn[name])
            scale = np.maximum(np.abs(ga[name]) + np.abs(gn[name]), 1e-12)
            worst_abs = max(worst_abs, float(diff.max()))
            worst_rel = max(worst_rel, float((diff / scale).max()))
    return worst_abs, worst_rel


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n=N_CLASSES):
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def confidence_stats(P, y_true):
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    ok = pred == y_true
    mean_ok = float(conf[ok].mean()) if ok.any() else None
    mean_bad = float(conf[~ok].mean()) if (~ok).any() else None
    return mean_ok, mean_bad, int(ok.sum()), int((~ok).sum())


def print_confusion(cm, title):
    print(title)
    head = "истина \\ прогноз".ljust(18) + "".join(f"{l:>6}" for l in LETTERS)
    print(head)
    for i in range(N_CLASSES):
        print(f"{LETTERS[i]:<18}" + "".join(f"{cm[i, j]:>6}" for j in range(N_CLASSES)))


def evaluate(model, X, idx):
    P = model.predict_proba_batch(X)
    pred = P.argmax(axis=1)
    return P, pred, accuracy(idx, pred)


def run_series(H, X, Y, Xte, idx_te, seeds=SEEDS):
    results = []
    for s in seeds:
        model = MLP(H, seed=s)
        res = model.train(X, Y)
        P = model.predict_proba_batch(Xte)
        res.update({
            "H": H,
            "seed": s,
            "model": model,
            "acc_test": accuracy(idx_te, P.argmax(axis=1)),
            "n_params": model.n_params(),
        })
        results.append(res)
    return results


def print_series_table(all_results):
    print(f"{'H':>3} | {'параметров':>10} | {'эпохи (ср.)':>11} | {'эпохи [min; max]':>18} | "
          f"{'test acc (ср.)':>14} | {'test acc [min; max]':>20} | {'не сошлось':>10}")
    print("-" * 104)
    for H, res in all_results.items():
        ep = np.array([r["epochs"] for r in res], dtype=float)
        acc = np.array([r["acc_test"] for r in res])
        fail = sum(not r["converged"] for r in res)
        print(f"{H:>3} | {res[0]['n_params']:>10} | {ep.mean():>11.1f} | "
              f"[{ep.min():>6.0f}; {ep.max():>6.0f}] | {acc.mean() * 100:>13.1f}% | "
              f"[{acc.min() * 100:>5.1f}%; {acc.max() * 100:>5.1f}%] | {fail:>6}/{len(res)}")
    print("Подробно по запускам:")
    for H, res in all_results.items():
        for r in res:
            print(f"  H={H:>2} seed={r['seed']}: эпох={r['epochs']:>5} сошлось={str(r['converged']):<5} "
                  f"Es={r['es']:.4f} CE={r['ce']:.4f} test acc={r['acc_test'] * 100:5.1f}%")


def corrupt(x, m, rng):
    z = x.copy()
    if m > 0:
        pos = rng.choice(N_INPUT, size=m, replace=False)
        z[pos] = 1.0 - z[pos]
    return z, ([] if m == 0 else sorted(int(p) for p in pos))


def noise_test(model, Xte, idx_te, m_max=M_MAX, n_noise=N_NOISE, seed=NOISE_SEED):
    rng = np.random.default_rng(seed)
    acc_total, acc_class = [], []
    for m in range(m_max + 1):
        per_example = np.zeros(len(Xte))
        for i, x in enumerate(Xte):
            Z = np.array([corrupt(x, m, rng)[0] for _ in range(n_noise)])
            pred = model.predict_proba_batch(Z).argmax(axis=1)
            per_example[i] = np.mean(pred == idx_te[i])
        acc_total.append(float(per_example.mean()))
        acc_class.append([float(per_example[idx_te == c].mean()) for c in range(N_CLASSES)])
    return np.array(acc_total), np.array(acc_class)


def print_noise(acc_total, acc_class):
    print(f"{'m':>3} | {'общая':>8} | " + " | ".join(f"{l:>8}" for l in LETTERS))
    for m in range(len(acc_total)):
        print(f"{m:>3} | {acc_total[m] * 100:>7.1f}% | " + " | ".join(f"{a * 100:>7.1f}%" for a in acc_class[m]))
    below = [m for m in range(len(acc_total)) if acc_total[m] < 0.8]
    if below:
        m80 = below[0]
        print(f"Наименьшее m, при котором accuracy падает ниже 80 %: m = {m80} ({acc_total[m80] * 100:.1f}%)")
        weakest = int(np.argmin(acc_class[m80]))
    else:
        m80 = None
        print(f"Accuracy не опускается ниже 80 % при m = 0..{len(acc_total) - 1}")
        weakest = int(np.argmin(acc_class[-1]))
    print(f"Самая хрупкая буква: {LETTERS[weakest]} (accuracy {acc_class[m80 if m80 is not None else -1][weakest] * 100:.1f}%)")
    return m80, weakest


def analyze_hidden(model):
    print("Анализ скрытых нейронов (строка, столбец – нумерация с 1):")
    for j in range(model.H):
        w = model.W1[j]
        v = model.W2[:, j]
        order = np.argsort(-np.abs(w))[:4]
        cells = ", ".join(f"({o // COLS + 1},{o % COLS + 1}) {w[o]:+.2f}" for o in order)
        plus = [LETTERS[k] for k in range(N_CLASSES) if v[k] > 0]
        minus = [LETTERS[k] for k in range(N_CLASSES) if v[k] < 0]
        main = LETTERS[int(np.argmax(v))]
        vtxt = ", ".join(f"v_{LETTERS[k]}{j + 1}={v[k]:+.2f}" for k in range(N_CLASSES))
        print(f"  нейрон {j + 1}: {vtxt}")
        print(f"     усиливает: {', '.join(plus) if plus else '-'}; подавляет: {', '.join(minus) if minus else '-'}; "
              f"сильнее всего работает на «{main}»")
        print(f"     наибольшие по модулю веса: {cells}")


def most_uncertain(model, test_objs, k=3):
    X = np.array([o["x"] for o in test_objs])
    P = model.predict_proba_batch(X)
    order = np.argsort(P.max(axis=1))[:k]
    return [(test_objs[i], P[i]) for i in order]


def print_grid(rows, indent="    "):
    for r in rows:
        print(indent + r)


def print_uncertain(items):
    for rank, (obj, p) in enumerate(items, 1):
        pred = int(np.argmax(p))
        print(f"  #{rank}: истинная буква {LETTERS[obj['cls'] - 1]}, экземпляр №{obj['inst']}, "
              f"прогноз {LETTERS[pred]}, уверенность {p.max():.4f}")
        print("     вероятности: " + ", ".join(f"p({LETTERS[k]})={p[k]:.4f}" for k in range(N_CLASSES)))
        print_grid(obj["rows"], "     ")


def decision_text(p, tau=TAU):
    return "не распознано" if p.max() < tau else LETTERS[int(np.argmax(p))]


def rejection_stats(P, idx, tau=TAU):
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    rejected = conf < tau
    n = len(P)
    res = {"n": n, "rejected": int(rejected.sum())}
    if idx is None:
        res["rejected_correct"] = None
        res["correct_total"] = None
    else:
        correct = pred == idx
        res["rejected_correct"] = int(np.sum(rejected & correct))
        res["correct_total"] = int(correct.sum())
    return res


def print_rejection(rows):
    print(f"{'набор':<22} | {'примеров':>8} | {'отвергнуто':>10} | {'из них верных':>13} | {'верных всего':>12}")
    for name, r in rows:
        rc = "n/a" if r["rejected_correct"] is None else str(r["rejected_correct"])
        ct = "n/a" if r["correct_total"] is None else str(r["correct_total"])
        print(f"{name:<22} | {r['n']:>8} | {r['rejected']:>10} | {rc:>13} | {ct:>12}")


def class_frequency(objects, cls):
    return np.mean([o["x"] for o in objects if o["cls"] == cls], axis=0)


def cells_text(mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return "нет"
    return ", ".join(f"({i // COLS + 1},{i % COLS + 1})" for i in idx)


def confusion_analysis(objects, errors):
    if not errors:
        print("Ошибочных классификаций на тесте и на чужом почерке нет – путаницы букв не обнаружено.")
        return
    pair_count = {}
    for e in errors:
        key = tuple(sorted((e["true"], e["pred"])))
        pair_count[key] = pair_count.get(key, 0) + 1
    (a, b), n = max(pair_count.items(), key=lambda kv: kv[1])
    print(f"Чаще всего путаются: {LETTERS[a]} и {LETTERS[b]} (ошибок: {n})")
    for e in errors:
        print(f"  {e['source']}: {LETTERS[e['true']]}, экземпляр №{e['inst']} принят за {LETTERS[e['pred']]} (p = {e['p']:.4f})")
    fa, fb = class_frequency(objects, a + 1), class_frequency(objects, b + 1)
    both = (fa >= 0.5) & (fb >= 0.5)
    print(f"Клетки (строка, столбец), закрашенные не менее чем в половине экземпляров обеих букв "
          f"({LETTERS[a]} и {LETTERS[b]}): {cells_text(both)}")
    for e in errors:
        x = e["x"] > 0.5
        typical_pred = class_frequency(objects, e["pred"] + 1) >= 0.5
        typical_true = class_frequency(objects, e["true"] + 1) >= 0.5
        print(f"  {e['source']}, {LETTERS[e['true']]} №{e['inst']}:")
        print(f"     закрашены клетки, типичные для «{LETTERS[e['pred']]}» и нетипичные для «{LETTERS[e['true']]}»: "
              f"{cells_text(x & typical_pred & ~typical_true)}")
        print(f"     не закрашены клетки, типичные для «{LETTERS[e['true']]}»: {cells_text(~x & typical_true)}")


def tau_sweep(P_te, idx_te, P_f, idx_f, P_l, taus=(0.5, 0.6, 0.7, 0.8, 0.9)):
    def parts(P, idx, t):
        conf = P.max(axis=1)
        ok = P.argmax(axis=1) == idx
        rej = conf < t
        return int((rej & ok).sum()), int(ok.sum()), int((rej & ~ok).sum()), int((~ok).sum())

    print(f"{'τ':>5} | {'тест: верных отвергнуто':>25} | {'тест: ошибок отвергнуто':>25} | "
          f"{'чужой почерк: верных откл.':>27} | {'чужая буква':>12}")
    for t in taus:
        a = parts(P_te, idx_te, t)
        te_ok, te_bad = f"{a[0]}/{a[1]}", f"{a[2]}/{a[3]}"
        if P_f is not None:
            b = parts(P_f, idx_f, t)
            fh = f"{b[0]}/{b[1]}"
        else:
            fh = "нет данных"
        if P_l is not None:
            lt = "отвергнута" if P_l.max(axis=1)[0] < t else "принята"
        else:
            lt = "нет данных"
        print(f"{t:>5} | {te_ok:>25} | {te_bad:>25} | {fh:>27} | {lt:>12}")


def classify_and_print(model, x, label, true_idx=None, note=None):
    p = model.predict_proba(x)
    k = int(np.argmax(p))
    print(f"[{label}]")
    if note:
        print(f"  {note}")
    print_grid(vector_to_rows(x), "  ")
    print("  вероятности: " + ", ".join(f"p{j + 1}({LETTERS[j]})={p[j]:.4f}" for j in range(N_CLASSES)))
    print(f"  класс: {LETTERS[k]},  уверенность: {p.max():.4f}")
    print(f"  решение (τ = {TAU}): {decision_text(p)}")
    if true_idx is not None:
        print(f"  истинная буква: {LETTERS[true_idx]} -> {'верно' if k == true_idx else 'ОШИБКА'}")
    print()


def find_object(objects, cls, inst):
    for o in objects:
        if o["cls"] == cls and o["inst"] == inst:
            return o
    return None


def demo_mode(model, objects, foreign_hand, foreign_letter):
    print("\n" + "=" * 80)
    print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ: демонстрация")
    print("=" * 80)
    print("--- 3 примера из обучающей выборки ---")
    for cls, inst in ((1, 1), (2, 4), (3, 7)):
        o = find_object(objects, cls, inst)
        classify_and_print(model, o["x"], f"обучающий: {LETTERS[cls - 1]}, экземпляр {inst}", cls - 1)
    print("--- 3 примера из тестовой выборки ---")
    for cls, inst in ((1, 8), (2, 9), (3, 10)):
        o = find_object(objects, cls, inst)
        classify_and_print(model, o["x"], f"тестовый: {LETTERS[cls - 1]}, экземпляр {inst}", cls - 1)
    print("--- 2 намеренно искажённых примера ---")
    rng = np.random.default_rng(DEMO_SEED)
    for cls, inst, m in ((1, 8, 2), (3, 10, 4)):
        o = find_object(objects, cls, inst)
        z, pos = corrupt(o["x"], m, rng)
        cells = ", ".join(f"({p // COLS + 1},{p % COLS + 1})" for p in pos)
        classify_and_print(model, z, f"искажённый: {LETTERS[cls - 1]}, экземпляр {inst}", cls - 1,
                           note=f"изменено клеток: {m} – {cells}")
    if foreign_letter:
        print("--- «чужая» буква ---")
        classify_and_print(model, foreign_letter[0]["x"], "чужая буква")
    else:
        print("--- «чужая» буква: файл foreign_letter.txt не найден, пропущено ---\n")
    if foreign_hand:
        print("--- чужой почерк ---")
        for o in foreign_hand[:2]:
            idx = o["cls"] - 1 if 1 <= o["cls"] <= N_CLASSES else None
            classify_and_print(model, o["x"], f"чужой почерк: заголовок «{o['header']}»", idx)
    else:
        print("--- чужой почерк: файл foreign_hand.txt не найден, пропущено ---\n")


def read_grid_from_console():
    rows = []
    for i in range(ROWS):
        line = input(f"  строка {i + 1}/{ROWS}: ").strip()
        if line.lower() in ("q", "quit", "exit"):
            return None
        if not re.fullmatch(r"[#.]{5}", line):
            print(f"  Ошибка: нужно ровно {COLS} символов '#' и '.'. Начните ввод заново.")
            return []
        rows.append(line)
    return rows


def interactive_mode(model):
    print("\n--- Интерактивный режим ---")
    print(f"Введите {ROWS} строк по {COLS} символов ('#' – закрашено, '.' – пусто). Для выхода введите q.")
    while True:
        rows = read_grid_from_console()
        if rows is None:
            print("Выход из интерактивного режима.")
            break
        if not rows:
            continue
        classify_and_print(model, rows_to_vector(rows), "введённая сетка")


def title_info(info, text):
    return f"{text}\n{info}"


def plot_convergence(series_by_h, info, filename="lab5_convergence.png"):
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
    plt.figure(figsize=(8.5, 5.5))
    for color, (H, res) in zip(colors, series_by_h.items()):
        run = next(r for r in res if r["seed"] == BASE_SEED)
        xs = range(1, len(run["hist_es"]) + 1)
        plt.plot(xs, run["hist_es"], color=color, linewidth=1.6, label=f"Es, H={H}")
        plt.plot(xs, run["hist_ce"], color=color, linewidth=1.2, linestyle="--", label=f"CE, H={H}")
    plt.axhline(EE, color="gray", linestyle=":", label=f"Ee = {EE}")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("Es (сплошная) и CE (пунктир), лог. шкала")
    plt.title(title_info(info, "Сходимость: Es и CE по эпохам для трёх значений H"))
    plt.legend(ncol=2, fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=150)
    plt.close()
    print(f"Сохранён график: {filename}")


def plot_epochs_bar(series_by_h, info, filename="lab5_epochs_bar.png"):
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple"]
    width = 0.15
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for gi, ((H, res), color) in enumerate(zip(series_by_h.items(), colors)):
        for si, r in enumerate(res):
            xpos = gi + (si - (len(res) - 1) / 2) * width
            ax.bar(xpos, r["epochs"], width * 0.92, color=color if r["converged"] else "none",
                   edgecolor=color, hatch=None if r["converged"] else "//")
            ax.text(xpos, r["epochs"], str(r["epochs"]), ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(range(len(series_by_h)))
    ax.set_xticklabels([f"H = {H}" for H in series_by_h])
    ax.set_ylabel("Число эпох до остановки")
    ax.set_title(title_info(info, "Разброс числа эпох по 5 запускам (штриховка – критерий не достигнут)"))
    ax.legend(handles=[Patch(facecolor="gray", label="сошлось"),
                       Patch(facecolor="none", edgecolor="gray", hatch="//", label="не сошлось")])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=150)
    plt.close()
    print(f"Сохранён график: {filename}")


def plot_confusion(cm, info, title, filename):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(LETTERS)
    ax.set_yticklabels(LETTERS)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")
    ax.set_title(title_info(info, title))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=150)
    plt.close()
    print(f"Сохранён график: {filename}")


def plot_noise(acc_total, acc_class, info, filename="lab5_noise.png"):
    ms = np.arange(len(acc_total))
    plt.figure(figsize=(8, 5.2))
    plt.plot(ms, acc_total * 100, "k-o", linewidth=2.4, label="общая")
    for c in range(N_CLASSES):
        plt.plot(ms, acc_class[:, c] * 100, "--s", markersize=4, label=f"буква {LETTERS[c]}")
    plt.axhline(80, color="red", linestyle=":", label="80 %")
    plt.xlabel("Число инвертированных клеток m")
    plt.ylabel("Средняя accuracy, %")
    plt.ylim(-2, 104)
    plt.title(title_info(info, f"Устойчивость к шуму ({N_NOISE} зашумлений на тестовый пример)"))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=150)
    plt.close()
    print(f"Сохранён график: {filename}")


def plot_weight_maps(model, info, filename="lab5_weight_maps.png"):
    H = model.H
    vmax = float(np.abs(model.W1).max())
    fig, axes = plt.subplots(1, H, figsize=(2.6 * H, 4.6))
    axes = np.atleast_1d(axes)
    for j, ax in enumerate(axes):
        w = model.W1[j].reshape(ROWS, COLS)
        im = ax.imshow(w, cmap="seismic", vmin=-vmax, vmax=vmax)
        for r in range(ROWS):
            for c in range(COLS):
                ax.text(c, r, f"{w[r, c]:.1f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(w[r, c]) > 0.55 * vmax else "black")
        v = model.W2[:, j]
        ax.set_title(f"нейрон {j + 1}\nv=" + ",".join(f"{v[k]:+.1f}" for k in range(N_CLASSES)), fontsize=9)
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xticklabels(range(1, COLS + 1), fontsize=7)
        ax.set_yticklabels(range(1, ROWS + 1), fontsize=7)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="вес w")
    fig.suptitle(title_info(info, f"Карты весов скрытых нейронов (v – веса к выходам {', '.join(LETTERS)})"), fontsize=11)
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён график: {filename}")


def plot_uncertain(items, info, filename="lab5_uncertain.png"):
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 6.2))
    axes = np.atleast_1d(axes)
    for ax, (obj, p) in zip(axes, items):
        ax.imshow(obj["x"].reshape(ROWS, COLS), cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
        ax.grid(which="minor", color="gray", linewidth=0.5)
        ax.tick_params(which="minor", length=0)
        ax.set_xticks(range(COLS))
        ax.set_yticks(range(ROWS))
        ax.set_xticklabels(range(1, COLS + 1))
        ax.set_yticklabels(range(1, ROWS + 1))
        ax.set_title(f"{LETTERS[obj['cls'] - 1]}, экз. №{obj['inst']}, прогноз {LETTERS[int(np.argmax(p))]}\n"
                     + "\n".join(f"p({LETTERS[k]}) = {p[k]:.4f}" for k in range(N_CLASSES)), fontsize=9)
    fig.suptitle(title_info(info, "Три самых неуверенных тестовых примера"), fontsize=11, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(os.path.join(BASE_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранён график: {filename}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if SURNAME == "Фамилия":
        print("ВНИМАНИЕ: в начале файла не заданы SURNAME и LETTERS – впишите свою фамилию и три буквы.\n")

    objects = read_objects(DATASET_FILE)
    validate_dataset(objects)
    train_objs, test_objs = split_dataset(objects)
    Xtr, Ytr, itr = to_arrays(train_objs)
    Xte, Yte, ite = to_arrays(test_objs)

    print("=" * 80)
    print("ДАТАСЕТ")
    print("=" * 80)
    print(f"Файл: {os.path.basename(DATASET_FILE)}; классы: {', '.join(f'{i + 1} – {l}' for i, l in enumerate(LETTERS))}")
    print(f"Объектов: {len(objects)}; обучающих (экз. 1–{TRAIN_LAST_INSTANCE}): {len(train_objs)}; "
          f"тестовых (экз. {TRAIN_LAST_INSTANCE + 1}–{N_PER_CLASS}): {len(test_objs)}")
    quality = dataset_quality(objects)
    print_quality(quality)
    n_fill = quality["n_fill"]
    info = f"{SURNAME}, вариант {VARIANT}, N_fill={n_fill}"

    print("\n" + "=" * 80)
    print("ПРОВЕРКА ГРАДИЕНТОВ (аналитический и численный, центральная разность)")
    print("=" * 80)
    check_model = MLP(H_VAR, seed=BASE_SEED)
    abs_err, rel_err = gradient_check(check_model, Xtr, Ytr)
    print(f"Сеть 35-{H_VAR}-3 со случайными весами, 3 обучающих примера, шаг eps = 1e-5:")
    print(f"  максимальная абсолютная разница: {abs_err:.3e}")
    print(f"  максимальная относительная разница: {rel_err:.3e}")
    print(f"  вывод: {'градиенты совпадают' if rel_err < 1e-5 else 'ЕСТЬ РАСХОЖДЕНИЕ – проверьте формулы'}")

    print("\n" + "=" * 80)
    print(f"БАЗОВЫЙ ЭКСПЕРИМЕНТ: H = {H_VAR}, η = {LEARNING_RATE}, seed = {BASE_SEED}")
    print("=" * 80)
    base = MLP(H_VAR, seed=BASE_SEED)
    res = base.train(Xtr, Ytr)
    print(f"Эпох: {res['epochs']} (критерий Es <= {EE} достигнут: {res['converged']})")
    print(f"Итоговая Es = {res['es']:.4f};  итоговая CE = {res['ce']:.4f}")
    P_tr, pred_tr, acc_tr = evaluate(base, Xtr, itr)
    P_te, pred_te, acc_te = evaluate(base, Xte, ite)
    print(f"Accuracy: обучающая {acc_tr * 100:.1f}% ({int(round(acc_tr * len(Xtr)))}/{len(Xtr)}), "
          f"тестовая {acc_te * 100:.1f}% ({int(round(acc_te * len(Xte)))}/{len(Xte)})")
    cm_tr = confusion_matrix(itr, pred_tr)
    cm_te = confusion_matrix(ite, pred_te)
    print_confusion(cm_tr, "Матрица ошибок, обучающая выборка:")
    print_confusion(cm_te, "Матрица ошибок, тестовая выборка:")
    mean_ok, mean_bad, n_ok, n_bad = confidence_stats(P_te, ite)
    print(f"Средняя уверенность на верно классифицированных тестовых ({n_ok}): "
          f"{'нет' if mean_ok is None else f'{mean_ok:.4f}'}")
    print(f"Средняя уверенность на неверно классифицированных тестовых ({n_bad}): "
          f"{'нет ошибок' if mean_bad is None else f'{mean_bad:.4f}'}")
    for i in np.where(pred_te != ite)[0]:
        o = test_objs[i]
        print(f"  ошибка: {LETTERS[o['cls'] - 1]}, экземпляр №{o['inst']} принят за {LETTERS[pred_te[i]]} "
              f"(p = {P_te[i].max():.4f})")

    print("\n" + "=" * 80)
    print(f"ИССЛЕДОВАНИЕ СКРЫТОГО СЛОЯ: H in {H_LIST}, η = {LEARNING_RATE}, 5 запусков")
    print("=" * 80)
    series = {H: run_series(H, Xtr, Ytr, Xte, ite) for H in H_LIST}
    print_series_table(series)
    print(f"Разрешение тестовой выборки: 1 пример = {100 / len(Xte):.1f} п. п.")

    print("\n" + "=" * 80)
    print("УСТОЙЧИВОСТЬ К ШУМУ (базовая сеть)")
    print("=" * 80)
    acc_total, acc_class = noise_test(base, Xte, ite)
    print_noise(acc_total, acc_class)

    print("\n" + "=" * 80)
    print("КАРТЫ ВЕСОВ СКРЫТЫХ НЕЙРОНОВ")
    print("=" * 80)
    analyze_hidden(base)

    print("\n" + "=" * 80)
    print("ТРИ САМЫХ НЕУВЕРЕННЫХ ТЕСТОВЫХ ПРИМЕРА")
    print("=" * 80)
    uncertain = most_uncertain(base, test_objs)
    print_uncertain(uncertain)

    print("\n" + "=" * 80)
    print(f"ЧУЖОЙ ПОЧЕРК, «ЧУЖАЯ» БУКВА И ПОРОГ ОТКАЗА τ = {TAU}")
    print("=" * 80)
    foreign_hand = read_optional(FOREIGN_HAND_FILE)
    foreign_letter = read_optional(FOREIGN_LETTER_FILE)
    rej_rows = [("тестовая выборка", rejection_stats(P_te, ite))]
    errors = [{"source": "тест", "true": int(ite[i]), "pred": int(pred_te[i]), "inst": test_objs[i]["inst"],
               "p": float(P_te[i].max()), "x": Xte[i]} for i in np.where(pred_te != ite)[0]]
    Pf = idx_f = Pl = None
    if foreign_hand:
        Xf = np.array([o["x"] for o in foreign_hand])
        idx_f = np.array([o["cls"] - 1 for o in foreign_hand])
        Pf, pred_f, acc_f = evaluate(base, Xf, idx_f)
        print(f"Чужой почерк: {len(Xf)} символов, accuracy = {acc_f * 100:.1f}% "
              f"({int(round(acc_f * len(Xf)))}/{len(Xf)}); на своём тесте {acc_te * 100:.1f}%")
        print_confusion(confusion_matrix(idx_f, pred_f), "Матрица ошибок, чужой почерк:")
        for o, p in zip(foreign_hand, Pf):
            print(f"  «{o['header']}»: истина {LETTERS[o['cls'] - 1]}, прогноз {LETTERS[int(np.argmax(p))]}, "
                  f"p = " + ", ".join(f"{v:.4f}" for v in p) + f", решение: {decision_text(p)}")
        rej_rows.append(("чужой почерк", rejection_stats(Pf, idx_f)))
        errors += [{"source": "чужой почерк", "true": int(idx_f[i]), "pred": int(pred_f[i]), "inst": foreign_hand[i]["inst"],
                    "p": float(Pf[i].max()), "x": Xf[i]} for i in np.where(pred_f != idx_f)[0]]
    else:
        print(f"Файл {os.path.basename(FOREIGN_HAND_FILE)} не найден – кросс-тест на чужом почерке пропущен.")
    if foreign_letter:
        Xl = np.array([o["x"] for o in foreign_letter])
        Pl = base.predict_proba_batch(Xl)
        for o, p in zip(foreign_letter, Pl):
            print(f"«Чужая» буква («{o['header']}»): отнесена к «{LETTERS[int(np.argmax(p))]}» "
                  f"с вероятностью {p.max():.4f}; вероятности: " + ", ".join(f"{v:.4f}" for v in p)
                  + f"; решение: {decision_text(p)}")
        rej_rows.append(("«чужая» буква", rejection_stats(Pl, None)))
    else:
        print(f"Файл {os.path.basename(FOREIGN_LETTER_FILE)} не найден – тест с «чужой» буквой пропущен.")
    print_rejection(rej_rows)
    r_test = rej_rows[0][1]
    if r_test["correct_total"]:
        print(f"Цена порога на тесте: отвергнуто {r_test['rejected_correct']} из {r_test['correct_total']} "
              f"верных примеров ({100 * r_test['rejected_correct'] / r_test['correct_total']:.1f}%)")

    print("\n" + "=" * 80)
    print("ПУТАНИЦА БУКВ И ПЕРЕКРЫТИЕ КЛЕТОК")
    print("=" * 80)
    confusion_analysis(objects, errors)

    print("\n" + "=" * 80)
    print("ВЛИЯНИЕ ПОРОГА ОТКАЗА τ")
    print("=" * 80)
    tau_sweep(P_te, ite, Pf, idx_f, Pl)

    print("\n" + "=" * 80)
    print("ГРАФИКИ")
    print("=" * 80)
    plot_convergence(series, info)
    plot_epochs_bar(series, info)
    plot_confusion(cm_te, info, "Матрица ошибок, тестовая выборка", "lab5_confusion_test.png")
    plot_noise(acc_total, acc_class, info)
    plot_weight_maps(base, info)
    plot_uncertain(uncertain, info)

    demo_mode(base, objects, foreign_hand, foreign_letter)

    choice = input("Запустить интерактивный режим ввода сетки? (y/Enter – пропустить): ").strip().lower()
    if choice == "y":
        interactive_mode(base)
    else:
        print("Интерактивный режим пропущен.")


if __name__ == "__main__":
    main()
