import numpy as np
import matplotlib.pyplot as plt
from dataset_preview import load_dataset, nfill

# functions
def sigmoid(wsum):
    wsum = np.clip(wsum, -500, 500)
    return 1 / (1 + np.exp(-wsum))

def der_sigmoid(wsum):
    y = sigmoid(wsum)
    return y * (1 - y)

def softmax(wsum):
    wsum = wsum - np.max(wsum, axis=1, keepdims=True)
    e = np.exp(wsum)
    return e / np.sum(e, axis=1, keepdims=True)

def cross_entropy(yj, ej):
    eps = 1e-12
    yj = np.clip(yj, eps, 1 - eps)
    return -np.sum(ej * np.log(yj), axis=1, keepdims=True)

# model
class NN:
    def __init__(self, layer_sizes, act_funcs, der_funcs):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.act_funcs = act_funcs
        self.der_funcs = der_funcs

        self.weights = []
        self.thresholds = []

        for i in range(self.num_layers - 1):
            limit = 1 / np.sqrt(layer_sizes[i])
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            T = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.thresholds.append(T)

    def forward(self, X):
        self.wsums = []
        self.activations = [X]
        current_x = X

        for i in range(self.num_layers - 1):
            wsum = np.dot(current_x, self.weights[i]) - self.thresholds[i]
            self.wsums.append(wsum)
            current_x = self.act_funcs[i](wsum)
            self.activations.append(current_x)

        return current_x

    def backward(self, ethalon, lr):
        # softmax + cross entropy: output error is (yj - ej)
        error = self.activations[-1] - ethalon

        for i in range(self.num_layers - 2, -1, -1):
            inputs = self.activations[i]
            w_current = self.weights[i].copy()

            self.weights[i] -= np.dot(inputs.T, error) * lr
            self.thresholds[i] += np.sum(error, axis=0, keepdims=True) * lr

            if i > 0:
                error = np.dot(error, w_current.T) * self.der_funcs[i - 1](self.wsums[i - 1])

# dataset
def prepare_data(objects):
    X_train, y_train, X_test, y_test = [], [], [], []

    for cls, inst, letter, grid in objects:
        x = grid.reshape(-1)
        y = np.zeros(3)
        y[cls - 1] = 1

        if inst <= 7:
            X_train.append(x)
            y_train.append(y)
        else:
            X_test.append(x)
            y_test.append(y)

    return (np.array(X_train, float), np.array(y_train, float),
            np.array(X_test, float), np.array(y_test, float))

# training
def train_network(net, X_train, y_train, lr=0.05, target_error=0.05, max_epochs=5000):
    errors = []
    converged = False

    for epoch in range(max_epochs):
        indices = np.random.permutation(len(X_train))

        for i in indices:
            net.forward(X_train[i:i + 1])
            net.backward(y_train[i:i + 1], lr)

        pred = net.forward(X_train)
        error = float(np.mean(cross_entropy(pred, y_train)))
        errors.append(error)

        if error <= target_error:
            converged = True
            break

    return net, errors, epoch + 1, converged

# metrics
def accuracy(net, X, y):
    labels = np.argmax(net.forward(X), axis=1)
    true = np.argmax(y, axis=1)
    return np.mean(labels == true) * 100

def confusion_matrix(net, X, y):
    labels = np.argmax(net.forward(X), axis=1)
    true = np.argmax(y, axis=1)

    matrix = np.zeros((3, 3), int)
    for t, p in zip(true, labels):
        matrix[t][p] += 1
    return matrix

def confidence(net, X, y):
    probs = net.forward(X)
    labels = np.argmax(probs, axis=1)
    true = np.argmax(y, axis=1)
    conf = np.max(probs, axis=1)

    return conf[labels == true], conf[labels != true]

def print_matrix(matrix, letters):
    print("      " + "  ".join(f"{l:>4}" for l in letters))
    for i in range(3):
        print(f"{letters[i]:>4}  " + "  ".join(f"{matrix[i][j]:>4}" for j in range(3)))

# experiments
def count_parameters(H):
    return 35 * H + H + H * 3 + 3

def run_experiment(H, seed, X_train, y_train, X_test, y_test,
                   lr=0.05, target_error=0.05, max_epochs=5000):
    np.random.seed(seed)

    net = NN([35, H, 3], [sigmoid, softmax], [der_sigmoid, der_sigmoid])
    net, errors, epochs, converged = train_network(
        net, X_train, y_train, lr=lr, target_error=target_error, max_epochs=max_epochs)

    return {
        "H": H,
        "epochs": epochs,
        "converged": converged,
        "final_error": errors[-1],
        "errors": errors,
        "train_accuracy": accuracy(net, X_train, y_train),
        "test_accuracy": accuracy(net, X_test, y_test),
    }

def experiment_H(H_values, runs, X_train, y_train, X_test, y_test, lr=0.05):
    results = []

    for H in H_values:
        print(f"\nExperiments for H={H}")
        for seed in range(runs):
            result = run_experiment(H, seed, X_train, y_train, X_test, y_test, lr=lr)
            results.append(result)
            print(f"  run {seed + 1}: epochs={result['epochs']}, "
                  f"test accuracy={result['test_accuracy']:.1f}%, "
                  f"stop={result['converged']}")

    return results

def summarize_H(results):
    print("\nH STUDY")
    print(f"{'H':>3} | {'params':>6} | {'epochs mean':>11} | {'epochs min-max':>14} | "
          f"{'acc mean':>8} | {'acc min-max':>11} | {'no stop':>7}")
    print("-" * 84)

    for H in sorted(set(r["H"] for r in results)):
        group = [r for r in results if r["H"] == H]
        epochs = [r["epochs"] for r in group]
        accuracies = [r["test_accuracy"] for r in group]
        failed = sum(1 for r in group if not r["converged"])

        print(f"{H:>3} | {count_parameters(H):>6} | {np.mean(epochs):>11.1f} | "
              f"{min(epochs):>6}-{max(epochs):<7} | {np.mean(accuracies):>7.1f}% | "
              f"{min(accuracies):>5.1f}-{max(accuracies):<5.1f} | {failed:>7}")

# noise test
def noise_test(net, X_test, y_test, mmax=10, trials=50):
    n = len(X_test)
    labels = np.argmax(y_test, axis=1)

    overall = np.zeros(mmax + 1)
    per_class = np.zeros((3, mmax + 1))

    for m in range(mmax + 1):
        correct = np.zeros(n)

        for t in range(trials):
            for i in range(n):
                x = X_test[i].copy()
                if m > 0:
                    cells = np.random.choice(len(x), m, replace=False)
                    x[cells] = 1 - x[cells]

                pred = np.argmax(net.forward(x.reshape(1, -1)), axis=1)[0]
                if pred == labels[i]:
                    correct[i] += 1

        overall[m] = np.mean(correct / trials) * 100
        for c in range(3):
            per_class[c][m] = np.mean(correct[labels == c] / trials) * 100

    return overall, per_class

def print_noise(overall, per_class, letters):
    print("\nNOISE TEST")
    print(f"{'m':>2} | {'all':>7} | " + " | ".join(f"{l:>5}" for l in letters))
    print("-" * 40)

    for m in range(len(overall)):
        print(f"{m:>2} | {overall[m]:>6.1f}% | "
              + " | ".join(f"{per_class[c][m]:>5.1f}" for c in range(3)))

    below = [m for m in range(len(overall)) if overall[m] < 80]
    if below:
        print(f"Accuracy below 80% at m = {below[0]}")
    else:
        print(f"Accuracy stays >= 80% up to m = {len(overall) - 1}")

def plot_noise(overall, per_class, nfill_value, path="images/noise.png"):
    m = np.arange(len(overall))

    plt.figure(figsize=(8, 4.5))
    plt.plot(m, overall, "o-", label="all classes")
    for c, letter in enumerate(["П", "Р", "О"]):
        plt.plot(m, per_class[c], "o--", label=f"class {letter}")

    plt.axhline(80, color="red", linestyle="--", label="80%")
    plt.xlabel("Number of flipped cells m")
    plt.ylabel("Mean accuracy, %")
    plt.title(f"Nfill = {nfill_value}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

# weight maps
def describe_neurons(net, letters):
    W1 = net.weights[0]
    W2 = net.weights[1]

    print("\nHIDDEN NEURONS")
    for j in range(W1.shape[1]):
        w = W1[:, j]
        order = np.argsort(-np.abs(w))
        cells = [(int(i) // 5 + 1, int(i) % 5 + 1) for i in order[:3]]
        k = int(np.argmax(W2[j]))

        print(f"  neuron {j + 1}: letter {letters[k]}, "
              f"v = ({W2[j][0]:+.2f}, {W2[j][1]:+.2f}, {W2[j][2]:+.2f}), "
              f"max cells {cells}")

def plot_weights(net, letters, nfill_value, path="images/weights.png"):
    W1 = net.weights[0]
    W2 = net.weights[1]
    H = W1.shape[1]
    cols = 3
    rows = (H + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.9 * rows))
    axes = np.atleast_2d(axes)

    for j in range(H):
        ax = axes[j // cols][j % cols]
        w = W1[:, j].reshape(7, 5)
        limit = np.max(np.abs(w))

        ax.imshow(w, cmap="coolwarm", vmin=-limit, vmax=limit)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="0.7", linewidth=0.5)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)

        k = int(np.argmax(W2[j]))
        ax.set_title(f"neuron {j + 1} -> {letters[k]}", fontsize=9)

    for j in range(H, rows * cols):
        axes[j // cols][j % cols].axis("off")

    fig.suptitle(f"Nfill = {nfill_value}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

# uncertain examples
def uncertain_examples(net, objects, X_test, y_test, count=3):
    probs = net.forward(X_test)
    labels = np.argmax(probs, axis=1)
    true = np.argmax(y_test, axis=1)
    conf = np.max(probs, axis=1)

    test_objects = [obj for obj in objects if obj[1] > 7]
    order = np.argsort(conf)[:count]

    return [(test_objects[i], probs[i], int(true[i]), int(labels[i]), float(conf[i]))
            for i in order]

def print_uncertain(examples, letters):
    print("\nMOST UNCERTAIN TEST EXAMPLES")

    for obj, probs, true, pred, conf in examples:
        cls, inst, letter, grid = obj
        print(f"\n{letter}{inst}: true {letters[true]}, predicted {letters[pred]}, "
              f"confidence {conf:.4f}")
        for row in grid:
            print("  " + "".join("#" if v else "." for v in row))
        print("  p = " + ", ".join(f"{letters[k]} {probs[k]:.4f}" for k in range(3)))

def plot_uncertain(examples, letters, nfill_value, path="images/uncertain.png"):
    fig, axes = plt.subplots(1, len(examples), figsize=(2.6 * len(examples), 3.4))
    axes = np.atleast_1d(axes)

    for ax, (obj, probs, true, pred, conf) in zip(axes, examples):
        cls, inst, letter, grid = obj
        ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="0.7", linewidth=0.5)
        ax.tick_params(length=0, labelbottom=False, labelleft=False)
        ax.set_title(f"{letter}{inst}: {letters[pred]} {conf:.2f}", fontsize=9)

    fig.suptitle(f"Nfill = {nfill_value}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.show()

# cross test
def objects_to_arrays(objects):
    X, y = [], []

    for cls, inst, letter, grid in objects:
        X.append(grid.reshape(-1))
        row = np.zeros(3)
        row[cls - 1] = 1
        y.append(row)

    return np.array(X, float), np.array(y, float)

def print_cross_test(net, objects, letters, name="other handwriting"):
    X, y = objects_to_arrays(objects)
    probs = net.forward(X)
    labels = np.argmax(probs, axis=1)
    true = np.argmax(y, axis=1)

    print(f"\nCROSS TEST ({name})")
    for i, (cls, inst, letter, grid) in enumerate(objects):
        print(f"  {letter}{inst}: true {letters[true[i]]}, predicted {letters[labels[i]]}, "
              + ", ".join(f"p{letters[k]} {probs[i][k]:.4f}" for k in range(3)))

    print(f"Accuracy: {accuracy(net, X, y):.1f}%")
    print("Confusion matrix:")
    print_matrix(confusion_matrix(net, X, y), letters)

# refusal threshold
def refusal_stats(net, objects, letters, tau=0.6, name="test"):
    X, y = objects_to_arrays(objects)
    probs = net.forward(X)
    labels = np.argmax(probs, axis=1)
    true = np.argmax(y, axis=1)
    conf = np.max(probs, axis=1)

    rejected = conf < tau
    correct = labels == true
    accepted = ~rejected

    print(f"\nREFUSAL (tau={tau}) on {name}:")
    print(f"  examples: {len(objects)}, rejected: {int(rejected.sum())}, "
          f"of them correct: {int(np.sum(rejected & correct))}")
    if accepted.sum():
        print(f"  accepted: {int(accepted.sum())}, "
              f"accuracy among accepted: {np.mean(correct[accepted]) * 100:.1f}%")

def predict_letter(net, obj, letters, tau=0.6):
    cls, inst, letter, grid = obj
    probs = net.forward(grid.reshape(1, -1))[0]
    k = int(np.argmax(probs))

    print(f"\nFOREIGN LETTER ({letter or 'no letter'}, class {cls}):")
    for row in grid:
        print("  " + "".join("#" if v else "." for v in row))
    print("  p = " + ", ".join(f"{letters[i]} {probs[i]:.4f}" for i in range(3)))
    print(f"  predicted: {letters[k]}, confidence {probs[k]:.4f}")

    if probs[k] < tau:
        print(f"  decision: not recognized (confidence < tau = {tau})")
    else:
        print(f"  decision: {letters[k]}")

# plots for H study and errors
def plot_convergence(results, nfill_value, path="images/convergence.png"):
    plt.figure(figsize=(8, 4.5))

    for H in sorted(set(r["H"] for r in results)):
        errors = [r for r in results if r["H"] == H][0]["errors"]
        plt.plot(np.arange(1, len(errors) + 1), errors, label=f"H={H}")

    plt.axhline(0.05, color="red", linestyle="--", label="Ee = 0.05")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Es (cross-entropy)")
    plt.title(f"Nfill = {nfill_value}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

def plot_epochs(results, nfill_value, path="images/epochs.png"):
    Hs = sorted(set(r["H"] for r in results))

    plt.figure(figsize=(7, 4.5))
    for i, H in enumerate(Hs):
        values = [r["epochs"] for r in results if r["H"] == H]
        x = np.full(len(values), i) + np.linspace(-0.08, 0.08, len(values))
        plt.scatter(x, values, label=f"H={H} (mean {np.mean(values):.1f})")

    plt.xticks(range(len(Hs)), [f"H={H}" for H in Hs])
    plt.xlabel("Hidden neurons H")
    plt.ylabel("Epochs to stop")
    plt.title(f"Nfill = {nfill_value}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

def plot_matrix(matrix, letters, nfill_value, path="images/confusion.png"):
    plt.figure(figsize=(5.2, 4.5))
    plt.imshow(matrix, cmap="Blues")

    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i][j] > matrix.max() / 2 else "black"
            plt.text(j, i, matrix[i][j], ha="center", va="center", color=color)

    plt.xticks(range(3), letters)
    plt.yticks(range(3), letters)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Nfill = {nfill_value}")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

# functioning mode
def predict_grid(net, grid, letters, tau=0.6):
    probs = net.forward(grid.reshape(1, -1))[0]
    k = int(np.argmax(probs))

    print("  grid:")
    for row in grid:
        print("    " + "".join("#" if v else "." for v in row))
    print(f"  p1 = {probs[0]:.4f}, p2 = {probs[1]:.4f}, p3 = {probs[2]:.4f}")
    print(f"  class: {letters[k]}, confidence: {probs[k]:.4f}")

    if probs[k] < tau:
        print(f"  decision: not recognized (confidence < tau = {tau})")
    else:
        print(f"  decision: {letters[k]}")

def demo_mode(net, objects, foreign, letters, tau=0.6):
    train_objects = [obj for obj in objects if obj[1] <= 7]
    test_objects = [obj for obj in objects if obj[1] > 7]

    print("\nDEMO MODE")

    print("\n-- train examples --")
    for cls in [1, 2, 3]:
        predict_grid(net, [obj for obj in train_objects if obj[0] == cls][0][3], letters, tau)

    print("\n-- test examples --")
    for cls in [1, 2, 3]:
        predict_grid(net, [obj for obj in test_objects if obj[0] == cls][0][3], letters, tau)

    print("\n-- distorted test examples (4 cells changed) --")
    np.random.seed(7)
    for cls in [1, 2]:
        grid = [obj for obj in test_objects if obj[0] == cls][0][3].copy()
        cells = np.random.choice(35, 4, replace=False)
        for i in cells:
            grid[i // 5][i % 5] = 1 - grid[i // 5][i % 5]
        predict_grid(net, grid, letters, tau)

    print("\n-- foreign letter --")
    predict_grid(net, load_dataset("data/З.txt")[0][3], letters, tau)

    print("\n-- other handwriting --")
    for obj in foreign:
        predict_grid(net, obj[3], letters, tau)

def function_mode(net, letters, tau=0.6):
    print("\nFUNCTION MODE: enter 7 lines of 5 symbols ('#' or '.'), 'q' to exit")

    while True:
        grid = []
        bad = False

        for r in range(7):
            line = input(f"row {r + 1}: ").strip()
            if line.lower() == "q":
                return
            if len(line) != 5 or set(line) - {"#", "."}:
                print("wrong line: need 5 symbols '#' or '.'")
                bad = True
                break
            grid.append([1 if ch == "#" else 0 for ch in line])

        if bad:
            continue

        predict_grid(net, np.array(grid), letters, tau)

if __name__ == "__main__":
    np.random.seed(12)

    # variant 12
    H, lr, mmax = 6, 0.05, 10

    objects = load_dataset("data/dataset.txt")
    X_train, y_train, X_test, y_test = prepare_data(objects)

    letters = ["П", "Р", "О"]

    print(f"Nfill = {nfill(objects)}")
    print(f"Train samples: {len(X_train)}, test samples: {len(X_test)}")
    print(f"Network 35-{H}-3, learning rate {lr}")

    net = NN([35, H, 3], [sigmoid, softmax], [der_sigmoid, der_sigmoid])
    net, errors, epochs, converged = train_network(net, X_train, y_train, lr=lr)

    print("\nBASELINE EXPERIMENT")
    print(f"Epochs: {epochs}, stop criterion reached: {converged}")
    print(f"Final Es: {errors[-1]:.4f}")
    print(f"Train accuracy: {accuracy(net, X_train, y_train):.1f}%")
    print(f"Test accuracy: {accuracy(net, X_test, y_test):.1f}%")

    train_correct, train_wrong = confidence(net, X_train, y_train)
    test_correct, test_wrong = confidence(net, X_test, y_test)

    print("\nConfusion matrix (train):")
    print_matrix(confusion_matrix(net, X_train, y_train), letters)

    print("\nConfusion matrix (test):")
    print_matrix(confusion_matrix(net, X_test, y_test), letters)

    print("\nMean confidence:")
    print(f"  train: correct {np.mean(train_correct):.4f} ({len(train_correct)} samples), "
          f"wrong {np.mean(train_wrong) if len(train_wrong) else 0:.4f} "
          f"({len(train_wrong)} samples)")
    print(f"  test: correct {np.mean(test_correct):.4f} ({len(test_correct)} samples), "
          f"wrong {np.mean(test_wrong) if len(test_wrong) else 0:.4f} "
          f"({len(test_wrong)} samples)")

    plt.figure(figsize=(8, 4.5))
    plt.plot(np.arange(1, len(errors) + 1), errors)
    plt.axhline(0.05, color="red", linestyle="--", label="Ee = 0.05")
    plt.xlabel("Epoch")
    plt.ylabel("Es (cross-entropy)")
    plt.title(f"Nfill = {nfill(objects)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    H_results = experiment_H([2, H, 2 * H], 5, X_train, y_train, X_test, y_test, lr=lr)
    summarize_H(H_results)

    np.random.seed(12)
    overall, per_class = noise_test(net, X_test, y_test, mmax=mmax, trials=50)
    print_noise(overall, per_class, letters)
    plot_noise(overall, per_class, nfill(objects), "images/noise.png")

    describe_neurons(net, letters)
    plot_weights(net, letters, nfill(objects), "images/weights.png")

    examples = uncertain_examples(net, objects, X_test, y_test, 3)
    print_uncertain(examples, letters)
    plot_uncertain(examples, letters, nfill(objects), "images/uncertain.png")

    foreign = load_dataset("data/drun.txt")
    print_cross_test(net, foreign, letters, "drun.txt")

    tau = 0.6
    test_objects = [obj for obj in objects if obj[1] > 7]
    refusal_stats(net, test_objects, letters, tau, "own test")
    refusal_stats(net, foreign, letters, tau, "drun.txt")
    predict_letter(net, load_dataset("data/З.txt")[0], letters, tau)

    plot_convergence(H_results, nfill(objects), "images/convergence.png")
    plot_epochs(H_results, nfill(objects), "images/epochs.png")
    plot_matrix(confusion_matrix(net, X_test, y_test), letters, nfill(objects), "images/confusion.png")

    demo_mode(net, objects, foreign, letters, tau)
    function_mode(net, letters, tau)
