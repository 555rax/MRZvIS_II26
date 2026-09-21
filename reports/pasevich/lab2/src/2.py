
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

label_lo, label_hi = 3, -8

inputs = np.array([[label_lo, label_lo], [label_lo, label_hi],
                   [label_hi, label_lo], [label_hi, label_hi]], dtype=float)
targets_raw = np.array([[label_lo], [label_hi], [label_hi], [label_lo]], dtype=float)
targets_bin = np.array([[0.], [1.], [1.], [0.]])

learning_rate = 0.1
epoch_limit = 20000
mse_threshold = 0.01
bce_threshold = 0.05
random_seeds = [0, 1, 2, 3, 4]


def activation(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def activation_grad_from_out(out):
    return out * (1.0 - out)


def make_initial_params(seed_value, dim_in=2, dim_hidden=2, dim_out=1):
    generator = np.random.RandomState(seed_value)
    weights_1 = generator.uniform(-0.5, 0.5, (dim_in, dim_hidden))
    biases_1 = np.zeros((1, dim_hidden))
    weights_2 = generator.uniform(-0.5, 0.5, (dim_hidden, dim_out))
    biases_2 = np.zeros((1, dim_out))
    return weights_1, biases_1, weights_2, biases_2


def propagate(batch, weights_1, biases_1, weights_2, biases_2):
    hidden_act = activation(batch.dot(weights_1) + biases_1)
    output_act = activation(hidden_act.dot(weights_2) + biases_2)
    return hidden_act, output_act


def fit_mse(seed_value, lr=learning_rate, epochs=epoch_limit, limit=mse_threshold):
    weights_1, biases_1, weights_2, biases_2 = make_initial_params(seed_value)
    loss_history = []
    stop_epoch = None
    num_samples = inputs.shape[0]

    for epoch_idx in range(epochs):
        _, outputs_all = propagate(inputs, weights_1, biases_1, weights_2, biases_2)
        loss = np.mean((targets_raw - outputs_all) ** 2)
        loss_history.append(loss)

        if loss <= limit:
            stop_epoch = epoch_idx
            break

        for sample_idx in range(num_samples):
            row_in = inputs[sample_idx:sample_idx + 1]
            row_target = targets_raw[sample_idx:sample_idx + 1]

            hidden_act, output_act = propagate(row_in, weights_1, biases_1, weights_2, biases_2)

            residual = row_target - output_act
            grad_out = residual * activation_grad_from_out(output_act)
            grad_hid = grad_out.dot(weights_2.T) * activation_grad_from_out(hidden_act)

            weights_2 += hidden_act.T.dot(grad_out) * lr
            biases_2 += grad_out * lr
            weights_1 += row_in.T.dot(grad_hid) * lr
            biases_1 += grad_hid * lr

    return dict(weights_1=weights_1, biases_1=biases_1,
                weights_2=weights_2, biases_2=biases_2,
                loss_history=loss_history, stop_epoch=stop_epoch)


def fit_bce(seed_value, lr=learning_rate, epochs=epoch_limit, limit=bce_threshold, tiny=1e-9):
    weights_1, biases_1, weights_2, biases_2 = make_initial_params(seed_value)
    loss_history = []
    stop_epoch = None
    num_samples = inputs.shape[0]

    for epoch_idx in range(epochs):
        _, outputs_all = propagate(inputs, weights_1, biases_1, weights_2, biases_2)
        clipped = np.clip(outputs_all, tiny, 1 - tiny)
        loss = -np.mean(targets_bin * np.log(clipped) + (1 - targets_bin) * np.log(1 - clipped))
        loss_history.append(loss)

        if loss <= limit:
            stop_epoch = epoch_idx
            break

        for sample_idx in range(num_samples):
            row_in = inputs[sample_idx:sample_idx + 1]
            row_target = targets_bin[sample_idx:sample_idx + 1]

            hidden_act, output_act = propagate(row_in, weights_1, biases_1, weights_2, biases_2)

            grad_out = output_act - row_target
            grad_hid = grad_out.dot(weights_2.T) * activation_grad_from_out(hidden_act)

            weights_2 -= hidden_act.T.dot(grad_out) * lr
            biases_2 -= grad_out * lr
            weights_1 -= row_in.T.dot(grad_hid) * lr
            biases_1 -= grad_hid * lr

    return dict(weights_1=weights_1, biases_1=biases_1,
                weights_2=weights_2, biases_2=biases_2,
                loss_history=loss_history, stop_epoch=stop_epoch)


def choose_label(value, low, high):
    return low if abs(value - low) <= abs(value - high) else high


def unnormalize(out_norm, low, high):
    return low + out_norm * (high - low)


def assess_model(trained, mode):
    _, outputs = propagate(inputs, trained["weights_1"], trained["biases_1"],
                           trained["weights_2"], trained["biases_2"])

    if mode == "A":
        predicted_values = outputs.flatten()
    else:
        predicted_values = unnormalize(outputs.flatten(), label_lo, label_hi)

    ground_truth = targets_raw.flatten()
    predicted_labels = np.array([choose_label(v, label_lo, label_hi) for v in predicted_values])
    hit_rate = np.mean(predicted_labels == ground_truth)
    avg_error = np.mean(np.abs(predicted_values - ground_truth))
    return hit_rate, avg_error, predicted_values


def execute_all_runs():
    runs_mse, runs_bce = [], []

    print(f"CONFIGURATION A (MSE, no normalization, Ee = {mse_threshold})")
    for seed_value in random_seeds:
        trained = fit_mse(seed_value)
        hit_rate, avg_error, _ = assess_model(trained, "A")
        trained.update(seed=seed_value, accuracy=hit_rate, mae=avg_error)
        runs_mse.append(trained)
        status = f"epoch {trained['stop_epoch']}" if trained["stop_epoch"] is not None \
            else f"DID NOT CONVERGE within {epoch_limit} epochs"
        print(f"seed={seed_value}: {status}, Es_final={trained['loss_history'][-1]:.4f}, "
              f"accuracy={hit_rate:.2f}, MAE={avg_error:.2f}")

    print(f"\nCONFIGURATION B (BCE, normalized labels 0/1, Ee = {bce_threshold})")
    for seed_value in random_seeds:
        trained = fit_bce(seed_value)
        hit_rate, avg_error, _ = assess_model(trained, "B")
        trained.update(seed=seed_value, accuracy=hit_rate, mae=avg_error)
        runs_bce.append(trained)
        status = f"epoch {trained['stop_epoch']}" if trained["stop_epoch"] is not None \
            else f"DID NOT CONVERGE within {epoch_limit} epochs"
        print(f"seed={seed_value}: {status}, Es_final={trained['loss_history'][-1]:.4f}, "
              f"accuracy={hit_rate:.2f}, MAE={avg_error:.2f}")

    return runs_mse, runs_bce


def summarize_convergence(runs, heading):
    finished = [r for r in runs if r["stop_epoch"] is not None]
    epoch_values = [r["stop_epoch"] for r in finished]
    print(f"\n{heading}: converged {len(finished)} out of {len(runs)}")
    if epoch_values:
        print(f"Epochs to convergence: min={min(epoch_values)}, max={max(epoch_values)}, "
              f"mean={np.mean(epoch_values):.1f}, spread={max(epoch_values) - min(epoch_values)}")
    else:
        print("No run reached the stopping criterion.")


def draw_convergence(runs_mse, runs_bce, chosen_seed=0):
    picked_mse = next(r for r in runs_mse if r["seed"] == chosen_seed)
    picked_bce = next(r for r in runs_bce if r["seed"] == chosen_seed)

    fig, axis_left = plt.subplots(figsize=(8, 5))
    axis_left.plot(picked_mse["loss_history"], color="tab:blue", label="Configuration A (MSE)")
    axis_left.axhline(mse_threshold, color="tab:blue", linestyle="--", alpha=0.5,
                      label=f"Ee(MSE)={mse_threshold}")
    axis_left.set_xlabel("Epoch")
    axis_left.set_ylabel("Es (MSE)", color="tab:blue")
    axis_left.tick_params(axis="y", labelcolor="tab:blue")

    axis_right = axis_left.twinx()
    axis_right.plot(picked_bce["loss_history"], color="tab:orange", label="Configuration B (BCE)")
    axis_right.axhline(bce_threshold, color="tab:orange", linestyle="--", alpha=0.5,
                       label=f"Ee(BCE)={bce_threshold}")
    axis_right.set_ylabel("Es (BCE)", color="tab:orange")
    axis_right.tick_params(axis="y", labelcolor="tab:orange")

    handles_left, labels_left = axis_left.get_legend_handles_labels()
    handles_right, labels_right = axis_right.get_legend_handles_labels()
    axis_left.legend(handles_left + handles_right, labels_left + labels_right, loc="upper right")

    plt.title("Convergence: Configuration A (MSE) and Configuration B (BCE)")
    fig.tight_layout()
    plt.savefig("plot_1_convergence.png", dpi=150)
    plt.show()


def draw_epoch_bars(runs_mse, runs_bce):
    fig, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(random_seeds))
    bar_width = 0.35

    epoch_counts_mse = [r["stop_epoch"] if r["stop_epoch"] is not None else epoch_limit for r in runs_mse]
    epoch_counts_bce = [r["stop_epoch"] if r["stop_epoch"] is not None else epoch_limit for r in runs_bce]
    converged_mse = [r["stop_epoch"] is not None for r in runs_mse]
    converged_bce = [r["stop_epoch"] is not None for r in runs_bce]

    bars_mse = axis.bar(positions - bar_width / 2, epoch_counts_mse, bar_width,
                        label="Configuration A (MSE)", color="tab:blue")
    bars_bce = axis.bar(positions + bar_width / 2, epoch_counts_bce, bar_width,
                        label="Configuration B (BCE)", color="tab:orange")

    for bar, done in zip(bars_mse, converged_mse):
        if not done:
            bar.set_hatch("//")
            bar.set_edgecolor("red")
    for bar, done in zip(bars_bce, converged_bce):
        if not done:
            bar.set_hatch("//")
            bar.set_edgecolor("red")

    axis.set_xticks(positions)
    axis.set_xticklabels([f"seed={s}" for s in random_seeds])
    axis.set_ylabel("Epochs to convergence (or MAX_EPOCHS if not converged)")
    axis.set_title("Convergence stability over 5 runs (hatching = did not converge)")
    axis.legend()
    fig.tight_layout()
    plt.savefig("plot_2_epochs_bar.png", dpi=150)
    plt.show()


def draw_decision_maps(runs_mse, runs_bce, chosen_seed=0):
    picked_mse = next(r for r in runs_mse if r["seed"] == chosen_seed)
    picked_bce = next(r for r in runs_bce if r["seed"] == chosen_seed)

    axis_a = np.linspace(-10, 10, 200)
    axis_b = np.linspace(-10, 10, 200)
    grid_a, grid_b = np.meshgrid(axis_a, axis_b)
    flat_grid = np.column_stack([grid_a.ravel(), grid_b.ravel()])

    fig, panels = plt.subplots(1, 2, figsize=(13, 5.5))

    _, out_mse = propagate(flat_grid, picked_mse["weights_1"], picked_mse["biases_1"],
                           picked_mse["weights_2"], picked_mse["biases_2"])
    map_mse = out_mse.reshape(grid_a.shape)
    color_mse = panels[0].contourf(grid_a, grid_b, map_mse, levels=20, cmap="coolwarm")
    fig.colorbar(color_mse, ax=panels[0], label="Network output (0;1), targets not normalized")
    panels[0].scatter(inputs[:, 0], inputs[:, 1], c=targets_raw.flatten(),
                      cmap="coolwarm", edgecolors="black", s=120, linewidths=1.5)
    panels[0].set_title("Configuration A (MSE, no normalization)")
    panels[0].set_xlabel("A")
    panels[0].set_ylabel("B")

    _, out_bce = propagate(flat_grid, picked_bce["weights_1"], picked_bce["biases_1"],
                           picked_bce["weights_2"], picked_bce["biases_2"])
    map_bce = out_bce.reshape(grid_a.shape)
    color_bce = panels[1].contourf(grid_a, grid_b, map_bce, levels=20, cmap="coolwarm")
    fig.colorbar(color_bce, ax=panels[1], label="y_hat (normalized output, 0..1)")
    panels[1].contour(grid_a, grid_b, map_bce, levels=[0.5], colors="black", linewidths=2)
    panels[1].scatter(inputs[:, 0], inputs[:, 1], c=targets_bin.flatten(),
                      cmap="coolwarm", edgecolors="black", s=120, linewidths=1.5)
    panels[1].set_title("Configuration B (BCE, normalized labels)")
    panels[1].set_xlabel("A")
    panels[1].set_ylabel("B")

    fig.suptitle("Decision surface of the hidden layer (A,B in [-10;10])")
    fig.tight_layout()
    plt.savefig("plot_3_decision_surfaces.png", dpi=150)
    plt.show()


def draw_mae_chart(runs_mse, runs_bce):
    mean_mae_mse = np.mean([r["mae"] for r in runs_mse])
    mean_mae_bce = np.mean([r["mae"] for r in runs_bce])

    fig, axis = plt.subplots(figsize=(6, 5))
    bars = axis.bar(["Configuration A (MSE)", "Configuration B (BCE)"],
                    [mean_mae_mse, mean_mae_bce], color=["tab:blue", "tab:orange"])
    for bar, value in zip(bars, [mean_mae_mse, mean_mae_bce]):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}",
                  ha="center", va="bottom")
    axis.set_ylabel(f"Mean MAE in scale [{label_lo}; {label_hi}] (over 5 runs)")
    axis.set_title("Accuracy of restoring the original scale")
    fig.tight_layout()
    plt.savefig("plot_4_mae_comparison.png", dpi=150)
    plt.show()


def make_prediction(value_a, value_b, trained):
    _, out = propagate(np.array([[value_a, value_b]], dtype=float),
                       trained["weights_1"], trained["biases_1"],
                       trained["weights_2"], trained["biases_2"])
    normalized_out = out.item()
    scaled_out = unnormalize(normalized_out, label_lo, label_hi)
    label_out = choose_label(scaled_out, label_lo, label_hi)
    return normalized_out, scaled_out, label_out


def show_sample_predictions(trained):
    print("\nNETWORK OPERATION MODE (configuration B)")

    print("\n--- All 4 examples from the training table ---")
    for value_a, value_b in inputs:
        norm_out, scaled_out, label_out = make_prediction(value_a, value_b, trained)
        print(f"Input: ({value_a:.0f}, {value_b:.0f}) -> y_hat_norm={norm_out:.4f}, "
              f"y_hat_scaled={scaled_out:.2f}, class~{label_out}")

    print("\n--- Additional pairs (outside the training set) ---")
    extra_pairs = [(3, 3), (2, 7), (5, 5), (-4, 6)]
    for value_a, value_b in extra_pairs:
        norm_out, scaled_out, label_out = make_prediction(value_a, value_b, trained)
        print(f"Input: ({value_a}, {value_b}) -> y_hat_norm={norm_out:.4f}, "
              f"y_hat_scaled={scaled_out:.2f}, class~{label_out}")


def run_interactive(trained):
    print("\n--- Interactive mode (type 'q' to exit) ---")
    while True:
        raw = input("Enter A,B separated by space or comma in range [-10;10]: ").strip()
        if raw.lower() == "q":
            break
        try:
            part_a, part_b = raw.replace(",", " ").split()
            value_a, value_b = float(part_a), float(part_b)
        except ValueError:
            print("Failed to parse input, please try again (example: 3 7).")
            continue
        norm_out, scaled_out, label_out = make_prediction(value_a, value_b, trained)
        print(f"y_hat_norm={norm_out:.4f} | y_hat in scale [{label_lo};{label_hi}]={scaled_out:.2f} "
              f"| closest to class {label_out}")


if __name__ == "__main__":
    runs_mse, runs_bce = execute_all_runs()

    summarize_convergence(runs_mse, "Configuration A (MSE)")
    summarize_convergence(runs_bce, "Configuration B (BCE)")

    draw_convergence(runs_mse, runs_bce, chosen_seed=0)
    draw_epoch_bars(runs_mse, runs_bce)
    draw_decision_maps(runs_mse, runs_bce, chosen_seed=0)
    draw_mae_chart(runs_mse, runs_bce)

    best_bce = next(r for r in runs_bce if r["seed"] == 0)
    show_sample_predictions(best_bce)
