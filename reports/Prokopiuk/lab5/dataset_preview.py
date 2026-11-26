import numpy as np
import matplotlib.pyplot as plt

ROWS = 7
COLS = 5
FILL = "#"
EMPTY = "."

# dataset
def load_dataset(path="data/dataset.txt"):
    objects = []
    cls = inst = None
    letter = ""
    grid = []

    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line == "" or line.startswith(";"):
                continue

            if line.startswith("class"):
                if cls is not None:
                    objects.append((cls, inst, letter, np.array(grid)))

                parts = line.split()
                cls = int(parts[1])
                inst = int(parts[3])
                letter = parts[4] if len(parts) > 4 else ""
                grid = []
            else:
                if len(line) != COLS:
                    raise ValueError(f"grid line must have {COLS} symbols: {line}")
                grid.append([1 if ch == FILL else 0 for ch in line])

    if cls is not None:
        objects.append((cls, inst, letter, np.array(grid)))

    for obj in objects:
        if obj[3].shape != (ROWS, COLS):
            raise ValueError(f"{obj[2]}{obj[1]}: expected {ROWS}x{COLS} grid, got {obj[3].shape}")

    return objects

def nfill(objects):
    return int(sum(obj[3].sum() for obj in objects))

def dataset_ascii(objects):
    text = ""
    for cls, inst, letter, grid in objects:
        text += f"class {cls} instance {inst} {letter}\n"
        for row in grid:
            text += "".join(FILL if v else EMPTY for v in row) + "\n"
        text += "\n"
    return text

# plots
def build_preview(objects):
    classes = sorted(set(obj[0] for obj in objects))
    count = max(sum(1 for obj in objects if obj[0] == cls) for cls in classes)

    fig, axes = plt.subplots(len(classes), count,
                             figsize=(1.35 * count + 1.2, 1.5 * len(classes) + 0.6))
    axes = np.atleast_2d(axes)

    for r, cls in enumerate(classes):
        group = sorted([obj for obj in objects if obj[0] == cls], key=lambda obj: obj[1])

        for c in range(count):
            ax = axes[r][c]
            if c >= len(group):
                ax.axis("off")
                continue

            _, inst, letter, grid = group[c]
            ax.imshow(grid, cmap="Greys", vmin=0, vmax=1)
            ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
            ax.grid(which="minor", color="0.7", linewidth=0.5)
            ax.tick_params(length=0, labelbottom=False, labelleft=False)
            ax.set_title(f"{letter}{inst} ({int(grid.sum())})", fontsize=8)

        axes[r][0].set_ylabel(f"class {cls}", fontsize=9)

    fig.tight_layout()
    return fig

def save_preview(objects, path="images/dataset_preview.png"):
    fig = build_preview(objects)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def show_preview(objects):
    build_preview(objects)
    plt.show()

if __name__ == "__main__":
    objects = load_dataset("data/drun.txt")

    print(f"Objects: {len(objects)}, Nfill = {nfill(objects)}")

    save_preview(objects, "images/drun_preview.png")
    plt.show()
