import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Лабораторная работа №2
# Вариант 3: c0 = -9, c1 = -8
# Многослойный персептрон 2-2-1
# Сравнение MSE и Binary Cross-Entropy
#
# ВАЖНО:
# Конфигурация A (MSE) сравнивает выход сети В ИСХОДНОЙ
# шкале [-9; -8] с исходными целями -9 и -8.
#
# Конфигурация B (BCE) обучается на бинарных целях 0/1,
# как требуется в методичке.
#
# Сигмоида физически выдаёт только (0; 1), поэтому её выход
# переводится в исходную шкалу формулой:
# y_real = c0 + y_norm * (c1 - c0)
#
# Для варианта 3:
# y_real = -9 + y_norm
# ============================================================


# ------------------------------------------------------------
# 1. Исходные данные варианта 3
# ------------------------------------------------------------

C0 = -9.0
C1 = -8.0

X_REAL = np.array([
    [-9.0, -9.0],
    [-9.0, -8.0],
    [-8.0, -9.0],
    [-8.0, -8.0]
], dtype=float)

Y_REAL = np.array([
    [-9.0],
    [-8.0],
    [-8.0],
    [-9.0]
], dtype=float)

# Гиперпараметры
LEARNING_RATE = 0.5

# Пороги различаются, потому что MSE и BCE имеют разный масштаб.
EE_MSE = 0.01
EE_BCE = 0.03

MAX_EPOCHS = 20000

# Не менее 5 различных случайных инициализаций
SEEDS = [7, 8, 9, 10, 11]

# Одинаковый порядок перемешивания примеров для обеих конфигураций
SHUFFLE_SEED = 42


# ------------------------------------------------------------
# 2. Масштабирование
# ------------------------------------------------------------

def to_binary_scale(value, c0=C0, c1=C1):
    """
    Перевод исходной числовой шкалы [c0; c1] в [0; 1].

    Для варианта 3:
        -9 -> 0
        -8 -> 1
    """
    return (value - c0) / (c1 - c0)


def to_real_scale(y_norm, c0=C0, c1=C1):
    """
    Обратный перевод нормализованного выхода в исходную шкалу.

    y_real = c0 + y_norm * (c1 - c0)

    Для варианта 3:
    y_real = -9 + y_norm
    """
    return c0 + y_norm * (c1 - c0)


# Входы масштабируем одинаково для обеих конфигураций.
# Это только численная предобработка и не меняет классы XOR.
#
# Для обучающей таблицы:
# -9 -> 0
# -8 -> 1
X = to_binary_scale(X_REAL)

# Бинарные цели нужны для BCE и для вычисления accuracy.
Y_BINARY = to_binary_scale(Y_REAL)


# ------------------------------------------------------------
# 3. Вспомогательные функции
# ------------------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative_from_output(y):
    return y * (1.0 - y)


def mse_real_loss(y_true_real, y_pred_norm):
    """
    Конфигурация A.

    MSE считается В ИСХОДНОЙ ШКАЛЕ:
    сначала y_norm -> y_real,
    затем сравнение с исходными целями -9/-8.

    Es = 1/2 * sum((y_real - y_true_real)^2)
    """
    y_pred_real = to_real_scale(y_pred_norm)

    return 0.5 * np.sum(
        (y_pred_real - y_true_real) ** 2
    )


def bce_loss(y_true_binary, y_pred_norm):
    """
    Конфигурация B.

    BCE считается по нормализованным целям 0/1.
    """
    eps = 1e-12
    y_pred_norm = np.clip(
        y_pred_norm,
        eps,
        1.0 - eps
    )

    return -np.sum(
        y_true_binary * np.log(y_pred_norm)
        + (1.0 - y_true_binary) * np.log(1.0 - y_pred_norm)
    )


def nearest_class_from_real(y_real):
    """
    Определяет, к какому исходному классу значение ближе.
    """
    d0 = np.abs(y_real - C0)
    d1 = np.abs(y_real - C1)

    return np.where(d0 <= d1, C0, C1)


def classification_accuracy(y_true_real, y_pred_norm):
    """
    Accuracy считается в исходных классах c0/c1.
    """
    y_pred_real = to_real_scale(y_pred_norm)

    true_classes = nearest_class_from_real(y_true_real)
    pred_classes = nearest_class_from_real(y_pred_real)

    return np.mean(true_classes == pred_classes)


def mean_absolute_error_real(y_true_real, y_pred_norm):
    """
    Средняя абсолютная ошибка в исходной шкале [c0; c1].
    """
    y_pred_real = to_real_scale(y_pred_norm)

    return np.mean(
        np.abs(y_true_real - y_pred_real)
    )


# ------------------------------------------------------------
# 4. MLP 2-2-1
# ------------------------------------------------------------

class MLP221:
    def __init__(self, seed=7):
        rng = np.random.RandomState(seed)

        # Малые случайные веса
        self.W1 = rng.uniform(
            -0.1, 0.1,
            size=(2, 2)
        )

        self.b1 = rng.uniform(
            -0.1, 0.1,
            size=(1, 2)
        )

        self.W2 = rng.uniform(
            -0.1, 0.1,
            size=(2, 1)
        )

        self.b2 = rng.uniform(
            -0.1, 0.1,
            size=(1, 1)
        )

    def forward(self, x):
        """
        Прямое распространение.
        Сигмоида используется на скрытом и выходном слоях.
        """

        net1 = x @ self.W1 + self.b1
        hidden = sigmoid(net1)

        net2 = hidden @ self.W2 + self.b2
        output_norm = sigmoid(net2)

        return hidden, output_norm

    def predict_norm(self, x):
        return self.forward(x)[1]

    def train(
        self,
        x_train,
        y_real_train,
        y_binary_train,
        loss_name,
        learning_rate,
        ee,
        max_epochs,
        shuffle_seed
    ):
        """
        Онлайн-обучение:
        обновление весов после каждого примера.
        """

        rng_shuffle = np.random.RandomState(
            shuffle_seed
        )

        history = []
        converged = False

        for epoch in range(1, max_epochs + 1):

            indices = np.arange(
                len(x_train)
            )
            rng_shuffle.shuffle(indices)

            for i in indices:

                x = x_train[i:i + 1]
                target_real = y_real_train[i:i + 1]
                target_binary = y_binary_train[i:i + 1]

                # -------------------------
                # Прямое распространение
                # -------------------------
                hidden, output_norm = self.forward(x)

                # -------------------------
                # Выходной градиент
                # -------------------------

                if loss_name == "mse":

                    # Сначала переводим нормализованный выход
                    # в исходную шкалу.
                    output_real = to_real_scale(
                        output_norm
                    )

                    # E = 1/2 * (y_real - target_real)^2
                    #
                    # y_real =
                    # c0 + output_norm * (c1 - c0)
                    #
                    # dE/dnet =
                    # (y_real - target_real)
                    # * (c1 - c0)
                    # * sigmoid'(net)

                    delta2 = (
                        (output_real - target_real)
                        * (C1 - C0)
                        * sigmoid_derivative_from_output(
                            output_norm
                        )
                    )

                elif loss_name == "bce":

                    # Для BCE + sigmoid:
                    #
                    # dE/dnet = y_hat - y
                    #
                    # Цель здесь бинарная: 0 или 1.

                    delta2 = (
                        output_norm
                        - target_binary
                    )

                else:
                    raise ValueError(
                        "loss_name должен быть 'mse' или 'bce'"
                    )

                # -------------------------
                # Градиент скрытого слоя
                # -------------------------

                delta1 = (
                    (delta2 @ self.W2.T)
                    * sigmoid_derivative_from_output(
                        hidden
                    )
                )

                # -------------------------
                # Онлайн-обновление весов
                # -------------------------

                self.W2 -= (
                    learning_rate
                    * hidden.T
                    @ delta2
                )

                self.b2 -= (
                    learning_rate
                    * delta2
                )

                self.W1 -= (
                    learning_rate
                    * x.T
                    @ delta1
                )

                self.b1 -= (
                    learning_rate
                    * delta1
                )

            # -------------------------------------
            # Суммарная ошибка после полной эпохи
            # -------------------------------------

            predictions_norm = self.predict_norm(
                x_train
            )

            if loss_name == "mse":
                error = mse_real_loss(
                    y_real_train,
                    predictions_norm
                )

            else:
                error = bce_loss(
                    y_binary_train,
                    predictions_norm
                )

            history.append(
                float(error)
            )

            if error <= ee:
                converged = True
                break

        return {
            "epochs": epoch,
            "final_error": float(error),
            "converged": converged,
            "history": history
        }


# ------------------------------------------------------------
# 5. Один запуск
# ------------------------------------------------------------

def run_one_configuration(loss_name, seed):

    model = MLP221(seed=seed)

    if loss_name == "mse":
        ee = EE_MSE
    else:
        ee = EE_BCE

    training = model.train(
        x_train=X,
        y_real_train=Y_REAL,
        y_binary_train=Y_BINARY,
        loss_name=loss_name,
        learning_rate=LEARNING_RATE,
        ee=ee,
        max_epochs=MAX_EPOCHS,
        shuffle_seed=SHUFFLE_SEED
    )

    pred_norm = model.predict_norm(X)
    pred_real = to_real_scale(pred_norm)

    accuracy = classification_accuracy(
        Y_REAL,
        pred_norm
    )

    mae = mean_absolute_error_real(
        Y_REAL,
        pred_norm
    )

    return {
        "loss_name": loss_name,
        "seed": seed,
        "model": model,

        "epochs": training["epochs"],
        "final_error": training["final_error"],
        "converged": training["converged"],
        "history": training["history"],

        "pred_norm": pred_norm,
        "pred_real": pred_real,

        "accuracy": float(accuracy),
        "mae": float(mae)
    }


# ------------------------------------------------------------
# 6. Пять запусков
# ------------------------------------------------------------

def run_experiments():

    mse_runs = []
    bce_runs = []

    print("\n" + "=" * 78)
    print("ЛАБОРАТОРНАЯ РАБОТА №2 — ВАРИАНТ 3")
    print("c0 = -9, c1 = -8")
    print("=" * 78)

    print(
        "\nКонфигурация A: "
        "MSE в исходной шкале [-9; -8]"
    )

    for seed in SEEDS:

        result = run_one_configuration(
            "mse",
            seed
        )

        mse_runs.append(result)

        status = (
            "ДА"
            if result["converged"]
            else "НЕТ"
        )

        print(
            f"seed={seed:2d} | "
            f"эпох={result['epochs']:5d} | "
            f"Es={result['final_error']:.6f} | "
            f"accuracy={result['accuracy']:.2f} | "
            f"MAE={result['mae']:.6f} | "
            f"сходимость={status}"
        )

    print(
        "\nКонфигурация B: "
        "BCE по нормализованным целям 0/1"
    )

    for seed in SEEDS:

        result = run_one_configuration(
            "bce",
            seed
        )

        bce_runs.append(result)

        status = (
            "ДА"
            if result["converged"]
            else "НЕТ"
        )

        print(
            f"seed={seed:2d} | "
            f"эпох={result['epochs']:5d} | "
            f"Es={result['final_error']:.6f} | "
            f"accuracy={result['accuracy']:.2f} | "
            f"MAE={result['mae']:.6f} | "
            f"сходимость={status}"
        )

    return mse_runs, bce_runs


# ------------------------------------------------------------
# 7. Представительный запуск
# ------------------------------------------------------------

def choose_representative_run(runs):

    successful = [
        r for r in runs
        if r["converged"]
    ]

    if successful:

        successful_epochs = np.array(
            [
                r["epochs"]
                for r in successful
            ]
        )

        median_epochs = np.median(
            successful_epochs
        )

        return min(
            successful,
            key=lambda r:
                abs(
                    r["epochs"]
                    - median_epochs
                )
        )

    # Если успешных нет —
    # берём наименьшую итоговую ошибку.
    return min(
        runs,
        key=lambda r:
            r["final_error"]
    )


# ------------------------------------------------------------
# 8. Таблица предсказаний
# ------------------------------------------------------------

def print_prediction_table(title, run):

    print("\n" + title)
    print("-" * 88)

    print(
        f"{'A':>7}"
        f"{'B':>8}"
        f"{'Цель':>10}"
        f"{'ŷ norm':>14}"
        f"{'ŷ real':>14}"
        f"{'Класс':>12}"
    )

    print("-" * 88)

    for (
        x_real,
        target_real,
        pred_norm,
        pred_real
    ) in zip(
        X_REAL,
        Y_REAL.ravel(),
        run["pred_norm"].ravel(),
        run["pred_real"].ravel()
    ):

        predicted_class = (
            C0
            if abs(pred_real - C0)
            <= abs(pred_real - C1)
            else C1
        )

        print(
            f"{x_real[0]:7.1f}"
            f"{x_real[1]:8.1f}"
            f"{target_real:10.1f}"
            f"{pred_norm:14.4f}"
            f"{pred_real:14.4f}"
            f"{predicted_class:12.1f}"
        )


# ------------------------------------------------------------
# 9. Сводка серии запусков
# ------------------------------------------------------------

def print_summary(mse_runs, bce_runs):

    print("\n" + "=" * 78)
    print("СВОДКА ПО 5 ЗАПУСКАМ")
    print("=" * 78)

    for name, runs in [
        ("MSE", mse_runs),
        ("BCE", bce_runs)
    ]:

        successful = [
            r for r in runs
            if r["converged"]
        ]

        print(f"\n{name}:")

        print(
            f"  Достигли критерия: "
            f"{len(successful)} из {len(runs)}"
        )

        if successful:

            epochs = np.array(
                [
                    r["epochs"]
                    for r in successful
                ]
            )

            print(
                f"  Минимум эпох:      "
                f"{epochs.min()}"
            )

            print(
                f"  Максимум эпох:     "
                f"{epochs.max()}"
            )

            print(
                f"  Среднее эпох:      "
                f"{epochs.mean():.2f}"
            )

            print(
                f"  Разброс:           "
                f"{epochs.max() - epochs.min()}"
            )

        mean_accuracy = np.mean(
            [
                r["accuracy"]
                for r in runs
            ]
        )

        mean_mae = np.mean(
            [
                r["mae"]
                for r in runs
            ]
        )

        print(
            f"  Средняя accuracy:  "
            f"{mean_accuracy:.4f}"
        )

        print(
            f"  Средняя MAE:       "
            f"{mean_mae:.6f}"
        )


# ------------------------------------------------------------
# 10. График сходимости
# ------------------------------------------------------------

def plot_convergence(mse_run, bce_run):

    plt.figure(
        figsize=(10, 6)
    )

    mse_epochs = np.arange(
        1,
        len(mse_run["history"]) + 1
    )

    bce_epochs = np.arange(
        1,
        len(bce_run["history"]) + 1
    )

    plt.plot(
        mse_epochs,
        mse_run["history"],
        label=(
            f"MSE, seed="
            f"{mse_run['seed']}"
        )
    )

    plt.plot(
        bce_epochs,
        bce_run["history"],
        label=(
            f"BCE, seed="
            f"{bce_run['seed']}"
        )
    )

    plt.axhline(
        EE_MSE,
        linestyle="--",
        label=f"Ee MSE = {EE_MSE}"
    )

    plt.axhline(
        EE_BCE,
        linestyle=":",
        label=f"Ee BCE = {EE_BCE}"
    )

    # У функций разные масштабы,
    # поэтому логарифмическая ось удобнее.
    plt.yscale("log")

    plt.xlabel("Номер эпохи")
    plt.ylabel("Суммарная ошибка Es")

    plt.title(
        "Сравнение сходимости MSE и BCE"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()


# ------------------------------------------------------------
# 11. Эпохи по seed
# ------------------------------------------------------------

def plot_epochs_by_seed(
    mse_runs,
    bce_runs
):

    x = np.arange(
        len(SEEDS)
    )

    width = 0.36

    mse_epochs = [
        r["epochs"]
        for r in mse_runs
    ]

    bce_epochs = [
        r["epochs"]
        for r in bce_runs
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        mse_epochs,
        width,
        label="MSE"
    )

    plt.bar(
        x + width / 2,
        bce_epochs,
        width,
        label="BCE"
    )

    # Неуспешные запуски помечаем крестиком.
    for i, run in enumerate(mse_runs):

        if not run["converged"]:

            plt.scatter(
                x[i] - width / 2,
                run["epochs"],
                marker="x",
                s=100,
                linewidths=2
            )

    for i, run in enumerate(bce_runs):

        if not run["converged"]:

            plt.scatter(
                x[i] + width / 2,
                run["epochs"],
                marker="x",
                s=100,
                linewidths=2
            )

    plt.xticks(
        x,
        [str(seed) for seed in SEEDS]
    )

    plt.xlabel("Seed")
    plt.ylabel("Количество эпох")

    plt.title(
        "Устойчивость сходимости "
        "по 5 случайным инициализациям"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()


# ------------------------------------------------------------
# 12. Разделяющая поверхность
# ------------------------------------------------------------

def plot_decision_surface(run, title):

    model = run["model"]

    a_values = np.linspace(
        -10.0,
        10.0,
        250
    )

    b_values = np.linspace(
        -10.0,
        10.0,
        250
    )

    aa, bb = np.meshgrid(
        a_values,
        b_values
    )

    grid_real = np.column_stack(
        [
            aa.ravel(),
            bb.ravel()
        ]
    )

    # Та же предобработка,
    # что использовалась при обучении.
    grid_scaled = to_binary_scale(
        grid_real
    )

    z_norm = model.predict_norm(
        grid_scaled
    ).reshape(
        aa.shape
    )

    plt.figure(
        figsize=(8, 7)
    )

    contour = plt.contourf(
        aa,
        bb,
        z_norm,
        levels=np.linspace(
            0.0,
            1.0,
            21
        ),
        cmap="viridis"
    )

    plt.colorbar(
        contour,
        label=(
            "Нормализованный "
            "выход ŷ"
        )
    )

    # Граница классов
    plt.contour(
        aa,
        bb,
        z_norm,
        levels=[0.5],
        linewidths=2
    )

    true_classes = (
        Y_BINARY
        .ravel()
        .astype(int)
    )

    plt.scatter(
        X_REAL[:, 0],
        X_REAL[:, 1],
        c=true_classes,
        cmap="coolwarm",
        edgecolors="black",
        s=120,
        label="Обучающие примеры"
    )

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.xlabel("A")
    plt.ylabel("B")

    plt.title(title)

    plt.legend()

    plt.grid(
        True,
        alpha=0.2
    )

    plt.tight_layout()


# ------------------------------------------------------------
# 13. Сравнение MAE
# ------------------------------------------------------------

def plot_mae_comparison(
    mse_run,
    bce_run
):

    names = [
        "MSE",
        "BCE"
    ]

    values = [
        mse_run["mae"],
        bce_run["mae"]
    ]

    plt.figure(
        figsize=(7, 5)
    )

    bars = plt.bar(
        names,
        values
    )

    plt.ylabel(
        "Средняя абсолютная ошибка"
    )

    plt.title(
        "MAE представительных запусков "
        "в исходной шкале [-9; -8]"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar, value in zip(
        bars,
        values
    ):

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.5f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()


# ------------------------------------------------------------
# 14. Режим функционирования
# ------------------------------------------------------------

def infer_one(model, a, b):

    x_real = np.array(
        [[a, b]],
        dtype=float
    )

    # Применяем то же масштабирование входа.
    x_scaled = to_binary_scale(
        x_real
    )

    y_norm = float(
        model.predict_norm(
            x_scaled
        )[0, 0]
    )

    y_real = float(
        to_real_scale(
            y_norm
        )
    )

    nearest_class = (
        C0
        if abs(y_real - C0)
        <= abs(y_real - C1)
        else C1
    )

    return (
        y_norm,
        y_real,
        nearest_class
    )


def demonstrate_functioning(model):

    print("\n" + "=" * 78)
    print("РЕЖИМ ФУНКЦИОНИРОВАНИЯ — BCE")
    print("=" * 78)

    # 4 обучающих + 3 дополнительных примера
    examples = [
        (-9.0, -9.0),
        (-9.0, -8.0),
        (-8.0, -9.0),
        (-8.0, -8.0),
        (-8.5, -8.5),
        (-9.5, -8.2),
        (-7.5, -9.2)
    ]

    for a, b in examples:

        (
            y_norm,
            y_real,
            nearest
        ) = infer_one(
            model,
            a,
            b
        )

        print(
            f"A={a:6.2f}, "
            f"B={b:6.2f} | "
            f"ŷ={y_norm:.4f} | "
            f"y_real={y_real:.4f} | "
            f"ближайший класс="
            f"{nearest:.1f}"
        )


def interactive_mode(model):

    print("\n" + "=" * 78)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ")
    print(
        "Введите A и B "
        "из диапазона [-10; 10]."
    )
    print("Для выхода введите q.")
    print("=" * 78)

    while True:

        text = input("\nA = ").strip()

        if text.lower() == "q":
            break

        try:

            a = float(text)
            b = float(
                input("B = ").strip()
            )

        except ValueError:

            print(
                "Ошибка: "
                "нужно ввести число."
            )
            continue

        if not (
            -10.0 <= a <= 10.0
            and
            -10.0 <= b <= 10.0
        ):

            print(
                "Ошибка: "
                "A и B должны быть "
                "в диапазоне [-10; 10]."
            )
            continue

        (
            y_norm,
            y_real,
            nearest
        ) = infer_one(
            model,
            a,
            b
        )

        print(
            f"Нормализованный "
            f"выход ŷ = {y_norm:.4f}"
        )

        print(
            f"Выход в исходной "
            f"шкале [{C0:.0f}; {C1:.0f}] "
            f"= {y_real:.4f}"
        )

        print(
            f"Ближайший класс "
            f"= {nearest:.1f}"
        )


# ------------------------------------------------------------
# 15. Главная функция
# ------------------------------------------------------------

def main():

    mse_runs, bce_runs = (
        run_experiments()
    )

    mse_rep = (
        choose_representative_run(
            mse_runs
        )
    )

    bce_rep = (
        choose_representative_run(
            bce_runs
        )
    )

    print_summary(
        mse_runs,
        bce_runs
    )

    print_prediction_table(
        "Представительный запуск MSE",
        mse_rep
    )

    print_prediction_table(
        "Представительный запуск BCE",
        bce_rep
    )

    demonstrate_functioning(
        bce_rep["model"]
    )

    # Требуемые графики
    plot_convergence(
        mse_rep,
        bce_rep
    )

    plot_epochs_by_seed(
        mse_runs,
        bce_runs
    )

    plot_decision_surface(
        mse_rep,
        (
            "Разделяющая поверхность "
            f"MSE, seed={mse_rep['seed']}"
        )
    )

    plot_decision_surface(
        bce_rep,
        (
            "Разделяющая поверхность "
            f"BCE, seed={bce_rep['seed']}"
        )
    )

    plot_mae_comparison(
        mse_rep,
        bce_rep
    )

    plt.show()

    # После закрытия окон графиков
    # можно вводить собственные A и B.
    interactive_mode(
        bce_rep["model"]
    )


if __name__ == "__main__":
    main()
