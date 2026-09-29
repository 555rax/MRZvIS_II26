import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mlp import MLP

def normalize_inputs(X, min_val=-10.0, max_val=10.0):
    return (X - min_val) / (max_val - min_val)

def normalize_target(y, c0=5.0, c1=-2.0):
    return (y - c0) / (c1 - c0)

def denormalize_target(y_norm, c0=5.0, c1=-2.0):
    return c0 + y_norm * (c1 - c0)

def calculate_mae(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def calculate_accuracy(predictions, targets, c0, c1):
    correct = 0
    for p, t in zip(predictions, targets):
        dist_c0 = abs(p - c0)
        dist_c1 = abs(p - c1)
        pred_class = c0 if dist_c0 < dist_c1 else c1
        if pred_class == t:
            correct += 1
    return correct / len(targets)

def run_experiments(seeds, c0=5.0, c1=-2.0, max_epochs=30000, target_error=1e-3):
    X_raw = np.array([
        [c0, c0],
        [c0, c1],
        [c1, c0],
        [c1, c1]
    ], dtype=np.float64)

    y_raw = np.array([c0, c1, c1, c0], dtype=np.float64)

    X_norm = normalize_inputs(X_raw)
    y_norm_targets = normalize_target(y_raw, c0, c1)

    results = {
        'mse': {'epochs': [], 'errors': [], 'maes': [], 'accuracy': [], 'success': [], 'best_model': None, 'best_hist': None},
        'bce': {'epochs': [], 'errors': [], 'maes': [], 'accuracy': [], 'success': [], 'best_model': None, 'best_hist': None}
    }

    best_err_m = float('inf')
    best_err_b = float('inf')

    for seed in seeds:
        model_mse = MLP(input_dim=2, hidden_dim=2, out_dim=1, lr=0.5, seed=seed)
        epochs_m, err_m, hist_m = model_mse.train(X_norm, y_norm_targets, max_epochs=max_epochs, target_error=target_error, loss_type='mse')
        
        preds_m_norm = np.array([model_mse.forward(x) for x in X_norm])
        preds_m_real = denormalize_target(preds_m_norm, c0, c1)
        mae_m = calculate_mae(preds_m_real, y_raw)
        acc_m = calculate_accuracy(preds_m_real, y_raw, c0, c1)
        
        results['mse']['epochs'].append(epochs_m)
        results['mse']['errors'].append(err_m)
        results['mse']['maes'].append(mae_m)
        results['mse']['accuracy'].append(acc_m)
        results['mse']['success'].append(err_m <= target_error)
        
        if err_m < best_err_m:
            best_err_m = err_m
            results['mse']['best_model'] = model_mse
            results['mse']['best_hist'] = hist_m

        model_bce = MLP(input_dim=2, hidden_dim=2, out_dim=1, lr=0.5, seed=seed)
        epochs_b, err_b, hist_b = model_bce.train(X_norm, y_norm_targets, max_epochs=max_epochs, target_error=target_error, loss_type='bce')
        
        preds_b_norm = np.array([model_bce.forward(x) for x in X_norm])
        preds_b_real = denormalize_target(preds_b_norm, c0, c1)
        mae_b = calculate_mae(preds_b_real, y_raw)
        acc_b = calculate_accuracy(preds_b_real, y_raw, c0, c1)
        
        results['bce']['epochs'].append(epochs_b)
        results['bce']['errors'].append(err_b)
        results['bce']['maes'].append(mae_b)
        results['bce']['accuracy'].append(acc_b)
        results['bce']['success'].append(err_b <= target_error)
        
        if err_b < best_err_b:
            best_err_b = err_b
            results['bce']['best_model'] = model_bce
            results['bce']['best_hist'] = hist_b

    return results, X_raw, y_raw, X_norm, y_norm_targets


def plot_all_results(results, seeds, c0=5.0, c1=-2.0, target_error=1e-3):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    ax1 = axs[0, 0]
    hist_m = results['mse']['best_hist']
    hist_b = results['bce']['best_hist']
    ax1.plot(hist_m, label='MSE (Fixed Config A)', color='blue')
    ax1.plot(hist_b, label='BCE (Config B)', color='orange')
    ax1.axhline(y=target_error, color='red', linestyle='--', label=f'Threshold Ee={target_error}')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Sum Error (Es)')
    ax1.set_title('Convergence Graph (Representative Run)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2 = axs[0, 1]
    x_indices = np.arange(len(seeds))
    width = 0.35

    epochs_m = results['mse']['epochs']
    epochs_b = results['bce']['epochs']

    bars1 = ax2.bar(x_indices - width/2, epochs_m, width, label='MSE', color='blue', alpha=0.7)
    bars2 = ax2.bar(x_indices + width/2, epochs_b, width, label='BCE', color='orange', alpha=0.7)

    for i in range(len(seeds)):
        if not results['mse']['success'][i]:
            bars1[i].set_hatch('//')
            bars1[i].set_color('navy')
        if not results['bce']['success'][i]:
            bars2[i].set_hatch('//')
            bars2[i].set_color('darkorange')

    ax2.set_xlabel('Seed Index')
    ax2.set_ylabel('Epochs to Convergence')
    ax2.set_title('Convergence Stability Across 5 Seeds')
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f'Seed {s}' for s in seeds])
    ax2.legend()
    ax2.grid(True, axis='y', ls="--", alpha=0.5)

    grid_x = np.linspace(-10, 10, 200)
    grid_y = np.linspace(-10, 10, 200)
    XX, YY = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    grid_points_norm = normalize_inputs(grid_points)

    pts_c0 = np.array([[c0, c0], [c1, c1]])
    pts_c1 = np.array([[c0, c1], [c1, c0]])

    ax3 = axs[1, 0]
    model_mse = results['mse']['best_model']
    Z_m = np.array([model_mse.forward(pt) for pt in grid_points_norm])
    Z_m_real = denormalize_target(Z_m, c0, c1) 
    ZZ_m = Z_m_real.reshape(XX.shape)

    contour_m = ax3.contourf(XX, YY, ZZ_m, levels=50, cmap='coolwarm', alpha=0.8)
    fig.colorbar(contour_m, ax=ax3, label='Output (Original Scale)')
    ax3.scatter(pts_c0[:, 0], pts_c0[:, 1], color='red', edgecolors='k', s=100, label=f'Class c0 ({c0})')
    ax3.scatter(pts_c1[:, 0], pts_c1[:, 1], color='blue', edgecolors='k', s=100, label=f'Class c1 ({c1})')
    ax3.set_xlabel('A')
    ax3.set_ylabel('B')
    ax3.set_title('Decision Boundary (MSE Best Run)')
    ax3.legend()
    ax3.grid(True, ls="--", alpha=0.3)

    ax4 = axs[1, 1]
    model_bce = results['bce']['best_model']
    Z_b = np.array([model_bce.forward(pt) for pt in grid_points_norm])
    Z_b_real = denormalize_target(Z_b, c0, c1)
    ZZ_b = Z_b_real.reshape(XX.shape)

    contour_b = ax4.contourf(XX, YY, ZZ_b, levels=50, cmap='coolwarm', alpha=0.8)
    fig.colorbar(contour_b, ax=ax4, label='Output (Original Scale)')
    ax4.scatter(pts_c0[:, 0], pts_c0[:, 1], color='red', edgecolors='k', s=100, label=f'Class c0 ({c0})')
    ax4.scatter(pts_c1[:, 0], pts_c1[:, 1], color='blue', edgecolors='k', s=100, label=f'Class c1 ({c1})')
    ax4.set_xlabel('A')
    ax4.set_ylabel('B')
    ax4.set_title('Decision Boundary (BCE Best Run)')
    ax4.legend()
    ax4.grid(True, ls="--", alpha=0.3)

    ax5 = axs[0, 2]
    avg_mae_m = np.mean(results['mse']['maes'])
    avg_mae_b = np.mean(results['bce']['maes'])

    ax5.bar(['MSE (Fixed Config A)', 'BCE (Config B)'], [avg_mae_m, avg_mae_b], color=['blue', 'orange'], alpha=0.7)
    ax5.set_ylabel('Mean Absolute Error (MAE)')
    ax5.set_title('Scale Reconstruction Accuracy (MAE in [c0, c1])')
    ax5.grid(True, axis='y', ls="--", alpha=0.5)

    for i, v in enumerate([avg_mae_m, avg_mae_b]):
        ax5.text(i, v + 0.001, f'{v:.6f}', ha='center', va='bottom')

    axs[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig("lab2_results_final.png", dpi=300)
    plt.show()

def interactive_mode(model_bce, c0=5.0, c1=-2.0):
    print("\n--- Operational Mode ---")
    print("Enter inputs A and B from range [-10, 10] (or type 'exit' to quit):")

    test_samples = [
        (c0, c0),
        (c0, c1),
        (c1, c0),
        (c1, c1),
        (0.0, 0.0),
        (2.5, -1.0),
        (-8.0, 7.0)
    ]

    print("\nAutomated test on truth table and extra points:")
    for a, b in test_samples:
        inp_norm = normalize_inputs(np.array([a, b]))
        out_norm = model_bce.forward(inp_norm)
        out_real = denormalize_target(out_norm, c0, c1)
        
        dist_c0 = abs(out_real - c0)
        dist_c1 = abs(out_real - c1)
        predicted_class = c0 if dist_c0 < dist_c1 else c1

        print(f"Input: ({a:>5.1f}, {b:>5.1f}) -> y_norm: {out_norm:.4f} | y_real: {out_real:>7.4f} | Class: {predicted_class}")

    while True:
        try:
            user_input = input("\nEnter A and B (e.g., 5 -2): ")
            if user_input.lower() == 'exit':
                break
            parts = user_input.strip().split()
            if len(parts) != 2:
                print("Please enter exactly two numbers.")
                continue
            
            a, b = float(parts[0]), float(parts[1])
            inp_norm = normalize_inputs(np.array([a, b]))
            out_norm = model_bce.forward(inp_norm)
            out_real = denormalize_target(out_norm, c0, c1)

            dist_c0 = abs(out_real - c0)
            dist_c1 = abs(out_real - c1)
            predicted_class = c0 if dist_c0 < dist_c1 else c1

            print(f"y_norm: {out_norm:.4f}")
            print(f"y_real: {out_real:.4f}")
            print(f"Closest Class: {predicted_class}")
        except ValueError:
            print("Invalid input format.")

def main():
    c0 = 5.0
    c1 = -2.0
    seeds = [42, 10, 123, 777, 999]
    target_error = 1e-3

    results, X_raw, y_raw, X_norm, y_norm_targets = run_experiments(seeds, c0, c1, max_epochs=30000, target_error=target_error)

    print(f"{'Config':<15} | {'Seed':<6} | {'Epochs':<8} | {'Final Error':<12} | {'MAE':<10} | {'Accuracy':<8} | {'Status':<8}")
    for i, seed in enumerate(seeds):
        print(f"{'MSE (Fixed)':<15} | {seed:<6} | {results['mse']['epochs'][i]:<8} | {results['mse']['errors'][i]:<12.6f} | {results['mse']['maes'][i]:<10.6f} | {results['mse']['accuracy'][i]:<8.2f} | {'Success' if results['mse']['success'][i] else 'Fail':<8}")
        print(f"{'BCE (B)':<15} | {seed:<6} | {results['bce']['epochs'][i]:<8} | {results['bce']['errors'][i]:<12.6f} | {results['bce']['maes'][i]:<10.6f} | {results['bce']['accuracy'][i]:<8.2f} | {'Success' if results['bce']['success'][i] else 'Fail':<8}")
        print("-" * 85)

    plot_all_results(results, seeds, c0, c1, target_error)
    interactive_mode(results['bce']['best_model'], c0, c1)

if __name__ == "__main__":
    main()