import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# КОНФИГУРАЦИЯ (Вариант 1)
# ==========================================
STUDENT_NAME = "Рудь В.В."
VARIANT = 1
H_VAR = 3  # Базовое число скрытых нейронов
ETA = 0.10  # Скорость обучения
M_MAX = 6  # Макс. число инвертированных клеток
EE = 0.05  # Критерий остановки (Средняя ошибка)
MAX_EPOCHS = 5000
TAU = 0.6  # Порог отказа

CLASS_NAMES = ['Р', 'У', 'Д']


# ==========================================
# 1. ГЕНЕРАЦИЯ И ЧТЕНИЕ ДАТАСЕТА (п.1-2)
# ==========================================
def generate_synthetic_dataset():
    dataset = []
    templates = [
        np.array([[1, 1, 1, 1, 0], [1, 0, 0, 1, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0]]),  # Р
        np.array([[1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0]]),  # У
        np.array([[0, 1, 1, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1]])  # Д
    ]
    for c in range(3):
        for i in range(10):
            img = templates[c].copy()
            shift = random.randint(-1, 1)
            img = np.roll(img, shift, axis=1)
            if shift > 0:
                img[:, :shift] = 0
            elif shift < 0:
                img[:, shift:] = 0
            noise_mask = np.random.rand(7, 5) < 0.05
            img = np.logical_xor(img, noise_mask).astype(int)
            dataset.append((c, i + 1, img.flatten()))
    return dataset


# ==========================================
# 2. КОНТРОЛЬ КАЧЕСТВА (п.3)
# ==========================================
def check_dataset_quality(dataset):
    print("\n--- Контроль качества датасета ---")
    nfill = sum([np.sum(item[2]) for item in dataset])
    print(f"Общее число закрашенных клеток Nfill: {nfill}")
    intra, inter = [], []
    for i in range(len(dataset)):
        for j in range(i + 1, len(dataset)):
            c1, _, v1 = dataset[i];
            c2, _, v2 = dataset[j]
            dist = np.sum(v1 != v2)
            if c1 == c2:
                intra.append(dist)
            else:
                inter.append(dist)
    print(f"Среднее внутриклассовое расстояние Хэмминга: {np.mean(intra):.2f} (должно быть > 3)")
    print(f"Среднее межклассовое расстояние Хэмминга: {np.mean(inter):.2f}")
    return nfill


# ==========================================
# 3. ЯДРО НЕЙРОСЕТИ (п.4)
# ==========================================
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]


def forward_pass(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1;
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2;
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


# ==========================================
# 4. ОБУЧЕНИЕ И АУДИТ (п.5, Задание 2)
# ==========================================
def train_mlp(X_train, y_train, H, lr, seed=42):
    np.random.seed(seed)
    n_samples, n_in = X_train.shape
    W1 = np.random.randn(n_in, H) * 0.1;
    b1 = np.zeros((1, H))
    W2 = np.random.randn(H, 3) * 0.1;
    b2 = np.zeros((1, 3))
    ce_history = []
    for epoch in range(MAX_EPOCHS):
        indices = np.random.permutation(n_samples)
        X_shuf, y_shuf = X_train[indices], y_train[indices]
        epoch_loss = 0
        for i in range(n_samples):
            x, y = X_shuf[i:i + 1], y_shuf[i:i + 1]
            Z1, A1, Z2, A2 = forward_pass(x, W1, b1, W2, b2)
            # Backprop
            delta2 = A2 - y
            dW2 = np.dot(A1.T, delta2);
            db2 = np.sum(delta2, axis=0, keepdims=True)
            delta1 = np.dot(delta2, W2.T) * (A1 * (1 - A1))
            dW1 = np.dot(x.T, delta1);
            db1 = np.sum(delta1, axis=0, keepdims=True)
            # Online update
            W1 -= lr * dW1;
            b1 -= lr * db1;
            W2 -= lr * dW2;
            b2 -= lr * db2
            epoch_loss += cross_entropy(y, A2)
        avg_loss = epoch_loss / n_samples
        ce_history.append(avg_loss)
        if avg_loss <= EE: break
    return W1, b1, W2, b2, epoch + 1, avg_loss, ce_history


def check_numerical_gradient():
    print("\n--- Аудит аналитических градиентов ---")
    np.random.seed(0)
    x = np.random.randn(1, 35);
    y = np.array([[1, 0, 0]])
    W1 = np.random.randn(35, 3) * 0.1;
    b1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 3) * 0.1;
    b2 = np.zeros((1, 3))
    _, A1, _, A2 = forward_pass(x, W1, b1, W2, b2)
    delta2 = A2 - y;
    delta1 = np.dot(delta2, W2.T) * (A1 * (1 - A1))
    ana_grad = np.dot(x.T, delta1)[0, 0]
    eps = 1e-5
    W1p = W1.copy();
    W1p[0, 0] += eps;
    loss_p = cross_entropy(y, forward_pass(x, W1p, b1, W2, b2)[3])
    W1m = W1.copy();
    W1m[0, 0] -= eps;
    loss_m = cross_entropy(y, forward_pass(x, W1m, b1, W2, b2)[3])
    num_grad = (loss_p - loss_m) / (2 * eps)
    print(f"Разница: {abs(ana_grad - num_grad):.2e} (норма < 1e-6)")


# ==========================================
# 5. МЕТРИКИ, ШУМ, МАТРИЦА ОШИБОК (п.6-8)
# ==========================================
def get_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_pass(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1), A2


def confusion_matrix(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    return cm


def noise_test_per_class(X_test, y_test, W1, b1, W2, b2, m_max, num_iter=50):
    res_overall = [];
    res_class = {c: [] for c in range(3)}
    for m in range(m_max + 1):
        acc_m = [];
        class_acc_m = {c: [] for c in range(3)}
        for _ in range(num_iter):
            X_noisy = X_test.copy()
            for i in range(X_noisy.shape[0]):
                flip = np.random.choice(35, m, replace=False)
                X_noisy[i, flip] = 1 - X_noisy[i, flip]
            preds, _ = get_predictions(X_noisy, W1, b1, W2, b2)
            true = np.argmax(y_test, axis=1)
            acc_m.append(np.mean(preds == true))
            for c in range(3):
                mask = (true == c)
                if np.sum(mask) > 0: class_acc_m[c].append(np.mean(preds[mask] == true[mask]))
        res_overall.append(np.mean(acc_m))
        for c in range(3): res_class[c].append(np.mean(class_acc_m[c]))
    return res_overall, res_class


# ==========================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ==========================================
if __name__ == "__main__":
    check_numerical_gradient()
    raw_data = generate_synthetic_dataset()
    Nfill = check_dataset_quality(raw_data)
    TITLE_SUFFIX = f"({STUDENT_NAME}, Вар. {VARIANT}, Nfill={Nfill})"

    X_train, y_train, X_test, y_test = [], [], [], []
    for c, num, vec in raw_data:
        y_vec = np.zeros(3);
        y_vec[c] = 1
        if num <= 7:
            X_train.append(vec); y_train.append(y_vec)
        else:
            X_test.append(vec); y_test.append(y_vec)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # --- П.6 БАЗОВЫЙ ЭКСПЕРИМЕНТ ---
    print(f"\n--- Базовый эксперимент (H={H_VAR}) ---")
    W1, b1, W2, b2, ep, loss, hist_ce_base = train_mlp(X_train, y_train, H_VAR, ETA)

    preds_train, _ = get_predictions(X_train, W1, b1, W2, b2)
    preds_test, probs_test = get_predictions(X_test, W1, b1, W2, b2)
    acc_train = np.mean(preds_train == np.argmax(y_train, axis=1))
    acc_test = np.mean(preds_test == np.argmax(y_test, axis=1))

    cm_train = confusion_matrix(np.argmax(y_train, axis=1), preds_train)
    cm_test = confusion_matrix(np.argmax(y_test, axis=1), preds_test)

    correct_mask = (preds_test == np.argmax(y_test, axis=1))
    conf_correct = np.mean(np.max(probs_test[correct_mask], axis=1)) if np.any(correct_mask) else 0
    conf_incorrect = np.mean(np.max(probs_test[~correct_mask], axis=1)) if np.any(~correct_mask) else 0

    print(f"Эпох: {ep} | Es: {loss:.4f} | Acc Train: {acc_train * 100:.1f}% | Acc Test: {acc_test * 100:.1f}%")
    print(f"Матрица ошибок (Train):\n{cm_train}")
    print(f"Матрица ошибок (Test):\n{cm_test}")
    print(f"Ср. уверенность (верные): {conf_correct:.4f} | (неверные): {conf_incorrect:.4f}")

    # --- П.7 ИССЛЕДОВАНИЕ СКРЫТОГО СЛОЯ ---
    print("\n--- Исследование скрытого слоя ---")
    H_values = [2, H_VAR, 2 * H_VAR]
    research_data = {H: {'epochs': [], 'acc': [], 'ce': [], 'not_conv': 0} for H in H_values}

    for H in H_values:
        for seed in range(5):
            w1, bb1, w2, bb2, ep, loss, hce = train_mlp(X_train, y_train, H, ETA, seed=seed)
            p_test, _ = get_predictions(X_test, w1, bb1, w2, bb2)
            acc = np.mean(p_test == np.argmax(y_test, axis=1))

            research_data[H]['epochs'].append(ep)
            research_data[H]['acc'].append(acc * 100)
            if ep == MAX_EPOCHS: research_data[H]['not_conv'] += 1
            if seed == 0: research_data[H]['ce'] = hce

        ed = research_data[H]['epochs'];
        ad = research_data[H]['acc']
        params = (35 * H + H) + (H * 3 + 3)
        print(
            f"H={H}: Эпохи {np.mean(ed):.0f} ({min(ed)}-{max(ed)}) | Acc {np.mean(ad):.1f}% ({min(ad):.1f}-{max(ad):.1f}) | "
            f"Не сошлось: {research_data[H]['not_conv']} | Параметров: {params}")

    # --- П.8 УСТОЙЧИВОСТЬ К ШУМУ ---
    noise_overall, noise_classes = noise_test_per_class(X_test, y_test, W1, b1, W2, b2, M_MAX)
    m_drop_80 = next((m for m, acc in enumerate(noise_overall) if acc < 0.8), "> " + str(M_MAX))
    print(f"\nAccuracy падает ниже 80% при m = {m_drop_80}")

    # --- П.13 ВИЗУАЛИЗАЦИЯ ---
    # 1. Сходимость
    plt.figure(figsize=(8, 5))
    for H in H_values: plt.plot(range(len(research_data[H]['ce'])), research_data[H]['ce'], label=f'H={H}')
    plt.yscale('log');
    plt.title(f'Сходимость CE {TITLE_SUFFIX}')
    plt.xlabel('Эпоха');
    plt.ylabel('CE (log)');
    plt.legend();
    plt.grid(True);
    plt.savefig('1_convergence.png')
    # 2. Boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot([research_data[H]['epochs'] for H in H_values], tick_labels=[f'H={H}' for H in H_values])
    plt.title(f'Разброс эпох {TITLE_SUFFIX}');
    plt.ylabel('Число эпох');
    plt.grid(True);
    plt.savefig('2_epochs_boxplot.png')
    # 3. Матрица ошибок
    plt.figure(figsize=(5, 4));
    plt.imshow(cm_test, cmap='Blues')
    for i in range(3):
        for j in range(3): plt.text(j, i, str(cm_test[i, j]), ha='center', va='center',
                                    color='red' if cm_test[i, j] > 0 else 'black')
    plt.title(f'Матрица ошибок Test {TITLE_SUFFIX}');
    plt.xticks([0, 1, 2], CLASS_NAMES);
    plt.yticks([0, 1, 2], CLASS_NAMES)
    plt.xlabel('Предсказано');
    plt.ylabel('Истинно');
    plt.savefig('3_confusion_matrix.png')
    # 4. График шума
    plt.figure(figsize=(8, 5));
    m_vals = range(M_MAX + 1)
    plt.plot(m_vals, [a * 100 for a in noise_overall], 'k-o', linewidth=2, label='Общая')
    colors = ['r', 'g', 'b']
    for c in range(3): plt.plot(m_vals, [a * 100 for a in noise_classes[c]], f'{colors[c]}--', marker='x',
                                label=f'Класс {CLASS_NAMES[c]}')
    plt.axhline(y=80, color='gray', linestyle=':', label='80%')
    plt.title(f'Accuracy от шума {TITLE_SUFFIX}');
    plt.xlabel('m');
    plt.ylabel('Accuracy (%)');
    plt.legend();
    plt.grid(True);
    plt.savefig('4_noise.png')
    # 5. Карты весов
    plt.figure(figsize=(15, 4))
    for h in range(H_VAR):
        plt.subplot(1, 3, h + 1);
        w_row = W1[:, h].reshape(7, 5);
        max_v = np.max(np.abs(w_row))
        plt.imshow(w_row, cmap='RdBu_r', vmin=-max_v, vmax=max_v);
        plt.title(f'Нейрон {h + 1}');
        plt.colorbar()
        max_idx = np.unravel_index(np.argmax(np.abs(w_row)), (7, 5))
        print(f"Нейрон {h + 1}: макс. вес (стр. {max_idx[0] + 1}, столб. {max_idx[1] + 1}) = {w_row[max_idx]:.2f}")
    plt.suptitle(f'Карты весов {TITLE_SUFFIX}');
    plt.savefig('5_weights.png')
    # 6. Неуверенные примеры
    unc_idx = np.argsort(np.max(probs_test, axis=1))[:3]
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(unc_idx):
        plt.subplot(1, 3, i + 1);
        plt.imshow(X_test[idx].reshape(7, 5), cmap='gray_r')
        prob_str = ", ".join([f"{p:.3f}" for p in probs_test[idx]])
        plt.title(f"Истина: {CLASS_NAMES[np.argmax(y_test[idx])]}\nВероятности: [{prob_str}]");
        plt.axis('off')
    plt.suptitle(f'Топ-3 неуверенных {TITLE_SUFFIX}');
    plt.savefig('6_uncertain.png')
    plt.show()

    # --- П.12 ПОРОГ ОТКАЗА ---
    print(f"\n--- Тест с порогом отказа (tau={TAU}) ---")
    rejected = sum(1 for p in probs_test if np.max(p) < TAU)
    correct_but_rejected = sum(
        1 for p, y in zip(probs_test, y_test) if np.max(p) < TAU and np.argmax(p) == np.argmax(y))
    print(f"Отвергнуто тестовых: {rejected} из {len(X_test)} (из них верных: {correct_but_rejected})")

    alien_grid = "#####\n#...#\n#....\n#....\n#....\n#....\n#...."
    alien_vec = np.array([[1 if c == '#' else 0 for c in row] for row in alien_grid.split('\n')]).flatten()
    _, _, _, alien_probs = forward_pass(alien_vec.reshape(1, -1), W1, b1, W2, b2)
    print(
        f"\nЧужая буква 'Г': Предсказано '{CLASS_NAMES[np.argmax(alien_probs)]}' (вероятность {np.max(alien_probs):.4f})")
    print(
        f"С учетом tau: {'НЕ РАСПОЗНАНО' if np.max(alien_probs) < TAU else 'Принято: ' + CLASS_NAMES[np.argmax(alien_probs)]}")

    # --- П.14 РЕЖИМ ФУНКЦИОНИРОВАНИЯ (Демонстрация 9+ примеров) ---
    print("\n" + "=" * 50 + "\n--- РЕЖИМ ФУНКЦИОНИРОВАНИЯ ---\n" + "=" * 50)


    def predict_letter(vec, true_label="?"):
        _, _, _, probs = forward_pass(vec.reshape(1, -1), W1, b1, W2, b2)
        probs = probs[0];
        pred_c = np.argmax(probs);
        conf = np.max(probs)
        grid_str = "\n".join(["".join(['#' if x == 1 else '.' for x in row]) for row in vec.reshape(7, 5)])
        print(f"Сетка (Истина: {true_label}):\n{grid_str}")
        print(f"Вероятности: p1={probs[0]:.4f}, p2={probs[1]:.4f}, p3={probs[2]:.4f}")
        if conf < TAU:
            print(f"РЕЗУЛЬТАТ: НЕ РАСПОЗНАНО (уверенность {conf:.4f} < {TAU})\n")
        else:
            print(f"РЕЗУЛЬТАТ: Буква '{CLASS_NAMES[pred_c]}' (уверенность {conf:.4f})\n")


    print(">>> 1. Три примера из ОБУЧАЮЩЕЙ выборки (по одному на класс):")
    predict_letter(X_train[0], CLASS_NAMES[np.argmax(y_train[0])])  # Р
    predict_letter(X_train[7], CLASS_NAMES[np.argmax(y_train[7])])  # У
    predict_letter(X_train[14], CLASS_NAMES[np.argmax(y_train[14])])  # Д

    print(">>> 2. Три примера из ТЕСТОВОЙ выборки:")
    predict_letter(X_test[0], CLASS_NAMES[np.argmax(y_test[0])])
    predict_letter(X_test[3], CLASS_NAMES[np.argmax(y_test[3])])
    predict_letter(X_test[6], CLASS_NAMES[np.argmax(y_test[6])])

    print(">>> 3. Два намеренно искаженных примера (изменено по 3 клетки):")
    dist_1 = X_test[1].copy();
    dist_1[[0, 15, 34]] = 1 - dist_1[[0, 15, 34]]  # Инвертируем 3 клетки
    predict_letter(dist_1, CLASS_NAMES[np.argmax(y_test[1])] + " (искажено)")
    dist_2 = X_test[4].copy();
    dist_2[[10, 20, 25]] = 1 - dist_2[[10, 20, 25]]
    predict_letter(dist_2, CLASS_NAMES[np.argmax(y_test[4])] + " (искажено)")

    print(">>> 4. «Чужая» буква:")
    predict_letter(alien_vec, "Г (чужая)")

    print(">>> 5. Два примера «чужого почерка» (имитация):")
    # (В реальности здесь нужно вставить векторы одногруппника)
    classmate_1 = X_test[2].copy();
    classmate_1 = np.roll(classmate_1, 1, axis=0)  # Имитируем другой почерк сдвигом
    predict_letter(classmate_1, "Одногруппник 1")
    classmate_2 = X_test[5].copy();
    classmate_2[[0, 1]] = 0  # Имитируем другой почерк
    predict_letter(classmate_2, "Одногруппник 2")