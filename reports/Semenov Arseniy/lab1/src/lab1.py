

import argparse
import json
from pathlib import Path

import numpy as np

C0, C1 = 0.0, -6.0
RAW_X = np.array([[0., 0.], [0., -6.], [-6., 0.], [-6., -6.]])
RAW_Y = np.array([0., -6., -6., 0.])
CHECK_X = np.array([[0., 0.], [0., -6.], [-6., 0.], [-6., -6.],
                    [-1., -5.], [-5., -1.], [-1., -1.], [-5., -5.]])


def normalize(values):
    return (np.asarray(values, dtype=float) - C0) / (C1 - C0)


def decode(values):
    return C0 + (C1 - C0) * np.asarray(values)


def sigmoid(values):
    """Устойчивая сигмоида без переполнения exp."""
    values = np.asarray(values, dtype=float)
    exp_neg_abs = np.exp(-np.abs(values))
    return np.where(values >= 0, 1. / (1. + exp_neg_abs),
                    exp_neg_abs / (1. + exp_neg_abs))


def nearest_class(values):
    # При равных расстояниях принимается c0.
    values = np.asarray(values)
    return np.where(np.abs(values - C0) <= np.abs(values - C1), C0, C1)


class Perceptron:
    """Сигмоидная сеть 2-2-1 либо один сигмоидный нейрон 2-1."""

    def __init__(self, hidden, seed):
        self.hidden = hidden
        rng = np.random.default_rng(seed)
        if hidden:
            self.w1 = rng.uniform(-0.5, 0.5, (2, 2))
            self.b1 = rng.uniform(-0.5, 0.5, 2)
        self.w2 = rng.uniform(-0.5, 0.5, 2)
        self.b2 = float(rng.uniform(-0.5, 0.5))

    def forward(self, x):
        h = sigmoid(x @ self.w1 + self.b1) if self.hidden else x
        return sigmoid(h @ self.w2 + self.b2)

    def step(self, x, target, lr):
        h = sigmoid(x @ self.w1 + self.b1) if self.hidden else x
        y = float(sigmoid(h @ self.w2 + self.b2))
        delta_out = (y - target) * y * (1. - y)
        if self.hidden:
            # Сначала вычисляются обе дельты со СТАРЫМИ весами.
            delta_hidden = delta_out * self.w2 * h * (1. - h)
            self.w1 -= lr * np.outer(x, delta_hidden)
            self.b1 -= lr * delta_hidden
        self.w2 -= lr * delta_out * h
        self.b2 -= lr * delta_out

    def weights(self):
        result = {"w2": self.w2.tolist(), "b2": self.b2}
        if self.hidden:
            result.update(w1=self.w1.tolist(), b1=self.b1.tolist())
        return result


def metrics(model):
    predictions = model.forward(normalize(RAW_X))
    error = float(0.5 * np.sum((predictions - normalize(RAW_Y)) ** 2))
    raw_predictions = decode(predictions)
    return {"Es": error, "Es_original": error * (C1 - C0) ** 2,
            "accuracy": float(np.mean(nearest_class(raw_predictions) == RAW_Y)),
            "predictions": raw_predictions.tolist()}


def train(hidden, seed, lr, ee, max_epochs):
    model = Perceptron(hidden, seed)
    x, targets = normalize(RAW_X), normalize(RAW_Y)
    # Фиксированный порядок примеров одинаков для двух архитектур.
    history = []
    result = metrics(model)
    epoch = 0
    history.append([epoch, result["Es"]])
    while result["Es"] > ee and epoch < max_epochs:
        for row, target in zip(x, targets):
            model.step(row, float(target), lr)
        epoch += 1
        # Суммарная ошибка считается заново при единых весах конца эпохи.
        result = metrics(model)
        if epoch % 100 == 0:
            history.append([epoch, result["Es"]])
    if history[-1][0] != epoch:
        history.append([epoch, result["Es"]])
    result.update(epochs=epoch, converged=result["Es"] <= ee,
                  history=history, weights=model.weights())
    return model, result


def predict_pair(model, a, b):
    pair = np.array([a, b], dtype=float)
    if not np.all(np.isfinite(pair)) or np.any(np.abs(pair) > 10):
        raise ValueError("A и B должны быть конечными числами от -10 до 10.")
    prediction = float(decode(model.forward(normalize(pair))))
    return prediction, float(nearest_class(prediction))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--ee", type=float, default=0.01)
    parser.add_argument("--max-epochs", type=int, default=50000)
    parser.add_argument("--no-input", action="store_true")
    parser.add_argument("--json", type=Path, help="Сохранить результаты эксперимента в JSON")
    args = parser.parse_args()
    if (not np.isfinite(args.lr) or args.lr <= 0 or not np.isfinite(args.ee)
            or args.ee <= 0 or args.max_epochs <= 0 or args.seed < 0):
        parser.error("lr, ee и max-epochs должны быть положительными; seed >= 0.")
    results = {"parameters": {"seed": args.seed, "lr": args.lr, "ee": args.ee,
                              "max_epochs": args.max_epochs},
               "numpy_version": np.__version__}
    mlp = None
    for hidden, name in [(True, "MLP"), (False, "SLP")]:
        model, result = train(hidden, args.seed, args.lr, args.ee, args.max_epochs)
        results[name] = result
        if hidden:
            mlp = model
        print(f"\n{name}: эпох={result['epochs']}, Es={result['Es']:.9f}, "
              f"Es исходная={result['Es_original']:.9f}, "
              f"accuracy={result['accuracy']:.0%}")
        print("Критерий достигнут" if result["converged"] else "Достигнут предел эпох")
        print("     A      B   Ожидается        Выход   Класс")
        for (a, b), target, pred in zip(RAW_X, RAW_Y, result["predictions"]):
            print(f"{a:6g} {b:6g} {target:11g} {pred:12.6f} {float(nearest_class(pred)):7g}")
    results["checks"] = []
    print("\nРежим функционирования MLP")
    for a, b in CHECK_X:
        pred, cls = predict_pair(mlp, a, b)
        results["checks"].append({"A": a, "B": b, "prediction": pred, "class": cls})
        print(f"A={a:g}, B={b:g}: выход={pred:.6f}, ближайший класс={cls:g}")
    if args.json:
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.no_input:
        return
    print("\nВведите A B в диапазоне [-10; 10]. Для выхода: q.")
    print("Для пар вне таблицы истинности выводится ответ сети; эталон не задан.")
    while True:
        try:
            line = input("A B > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершение работы.")
            break
        if line.lower() in {"q", "quit", "exit", "выход"}:
            break
        try:
            parts = line.replace(",", ".").split()
            if len(parts) != 2:
                raise ValueError("Введите ровно два числа через пробел.")
            pred, cls = predict_pair(mlp, *map(float, parts))
            print(f"Выход={pred:.6f}; ближайший класс={cls:g}")
        except ValueError as exc:
            print(f"Ошибка ввода: {exc}")


if __name__ == "__main__":
    main()
