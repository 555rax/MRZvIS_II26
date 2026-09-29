import argparse

import numpy as np

C0 = -3.0
C1 = -8.0
LEARNING_RATE = 1.0
MAX_EPOCHS = 150_000
ERROR_LIMIT = 0.001
SEED = 1

X_REAL = np.array([[C0, C0], [C0, C1], [C1, C0], [C1, C1]])
Y_REAL = np.array([[C0], [C1], [C1], [C0]])
Y = (Y_REAL - C0) / (C1 - C0)

def encode_inputs(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values - C0) <= np.abs(values - C1), -1.0, 1.0)

def decode_output(output):
    return C0 + (C1 - C0) * output

def nearest_class(values):
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values - C0) <= np.abs(values - C1), C0, C1)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))

def sigmoid_derivative(output):
    return output * (1.0 - output)

class MLP:

    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.uniform(-0.5, 0.5, (2, 2))
        self.b1 = np.zeros((1, 2))
        self.W2 = rng.uniform(-0.5, 0.5, (2, 1))
        self.b2 = np.zeros((1, 1))

    def predict(self, x):
        hidden = sigmoid(x @ self.W1 + self.b1)
        return sigmoid(hidden @ self.W2 + self.b2)

    def update(self, x, y, lr):
        hidden = sigmoid(x @ self.W1 + self.b1)
        output = sigmoid(hidden @ self.W2 + self.b2)
        output_delta = (y - output) * sigmoid_derivative(output)
        hidden_delta = (output_delta @ self.W2.T) * sigmoid_derivative(hidden)
        self.W2 += lr * (hidden.T @ output_delta)
        self.b2 += lr * output_delta
        self.W1 += lr * (x.T @ hidden_delta)
        self.b1 += lr * hidden_delta

class SinglePerceptron:

    def __init__(self, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-0.5, 0.5, (2, 1))
        self.b = np.zeros((1, 1))

    def predict(self, x):
        return sigmoid(x @ self.W + self.b)

    def update(self, x, y, lr):
        output = self.predict(x)
        delta = (y - output) * sigmoid_derivative(output)
        self.W += lr * (x.T @ delta)
        self.b += lr * delta

def total_error(model, x, y):
    return float(np.sum((decode_output(y) - decode_output(model.predict(x))) ** 2))

def accuracy(model, x, y_real):
    predicted = nearest_class(decode_output(model.predict(x)))
    return float(np.mean(predicted == y_real) * 100.0)

def train(
    model, x, y, name, lr=LEARNING_RATE, max_epochs=MAX_EPOCHS,
    error_limit=ERROR_LIMIT, verbose=True,
):
    error = total_error(model, x, y)
    if error <= error_limit:
        return 0, error, True

    for epoch in range(1, max_epochs + 1):
        for i in range(len(x)):
            model.update(x[i:i + 1], y[i:i + 1], lr)

        error = total_error(model, x, y)
        if verbose and (epoch == 1 or epoch % 10_000 == 0):
            print(f"{name}: эпоха {epoch}, Es = {error:.8f}")
        if error <= error_limit:
            return epoch, error, True

    return max_epochs, error, False

def show_examples(model):
    examples = [(-3, -3), (-3, -8), (-8, -3), (-8, -8), (0, -10), (4, 7)]
    print("\nПроверочные примеры (произвольные входы -> ближайшие классы):")
    print(f"{'A':>7} {'B':>7} {'Класс A':>9} {'Класс B':>9} "
          f"{'Ожидается':>10} {'Выход':>12} {'Класс':>8}")
    for a, b in examples:
        classes = nearest_class([a, b])
        expected = C1 if classes[0] != classes[1] else C0
        output = decode_output(model.predict(encode_inputs([[a, b]]))).item()
        predicted = nearest_class(output).item()
        print(f"{a:7.1f} {b:7.1f} {classes[0]:9.1f} {classes[1]:9.1f} "
              f"{expected:10.1f} {output:12.6f} {predicted:8.1f}")

def interactive_mode(model):
    print("\nВведите A и B через пробел, диапазон [-10; 10]. Выход: q.")
    print("Каждый вход относится к ближайшему классу; при -5.5 выбирается c0.")
    while True:
        try:
            line = input("A B: ").strip()
            if line.lower() in {"q", "quit", "exit", "выход"}:
                break
            try:
                a, b = map(float, line.split())
            except ValueError:
                print("Введите ровно два числа через пробел, например: -3 -8.")
                continue
            if not (np.isfinite(a) and np.isfinite(b)
                    and -10 <= a <= 10 and -10 <= b <= 10):
                print("Нужны конечные числа от -10 до 10.")
                continue
            output = decode_output(model.predict(encode_inputs([[a, b]]))).item()
            predicted = nearest_class(output).item()
            label = 'c0' if predicted == C0 else 'c1'
            print(
                f"Выход сети ŷ = {output:.6f}; "
                f"ближайший класс: {label} = {predicted:g}"
            )
        except (EOFError, KeyboardInterrupt):
            break
    print("Работа завершена.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-input', action='store_true',
                        help='Обучение и шесть проверок без интерактивного ввода.')
    args = parser.parse_args()
    x = encode_inputs(X_REAL)
    mlp = MLP()
    single = SinglePerceptron()
    results = []
    for name, model in [('MLP 2-2-1', mlp), ('Однослойный', single)]:
        epochs, error, converged = train(model, x, Y, name)
        results.append((name, epochs, error, accuracy(model, x, Y_REAL), converged))

    print(f"\nВариант 10: c0 = {C0:g}, c1 = {C1:g}; Ee = {ERROR_LIMIT}")
    print("Es — сумма квадратов ошибок в исходной шкале; accuracy — на 4 примерах.")
    print(
        f"{'Модель':<15} {'Эпохи':>8} {'Es':>14} {'Accuracy':>10}  "
        "Критерий остановки"
    )
    for name, epochs, error, acc, converged in results:
        status = 'достигнут' if converged else 'не достигнут (лимит эпох)'
        print(f"{name:<15} {epochs:8d} {error:14.8f} {acc:9.2f}%  {status}")

    show_examples(mlp)
    if not args.no_input:
        interactive_mode(mlp)

if __name__ == '__main__':
    main()
