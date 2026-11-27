import numpy as np
import matplotlib.pyplot as plt
from dataset_preview import load_dataset, nfill

MIN_WITHIN = 3.0

# metrics
def hamming(a, b):
    return int(np.sum(a != b))

def nfill_by_class(objects):
    classes = sorted(set(obj[0] for obj in objects))
    return {cls: nfill([obj for obj in objects if obj[0] == cls]) for cls in classes}

def class_letters(objects):
    letters = {}
    for cls, inst, letter, grid in objects:
        if letter:
            letters[cls] = letter
    return letters

def within_distances(objects, cls):
    group = sorted([obj for obj in objects if obj[0] == cls], key=lambda obj: obj[1])
    distances = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            distances.append(hamming(group[i][3], group[j][3]))
    return distances

def between_distances(objects, cls1, cls2):
    group1 = [obj for obj in objects if obj[0] == cls1]
    group2 = [obj for obj in objects if obj[0] == cls2]
    distances = []
    for obj1 in group1:
        for obj2 in group2:
            distances.append(hamming(obj1[3], obj2[3]))
    return distances

def all_within_distances(objects):
    distances = []
    for cls in sorted(set(obj[0] for obj in objects)):
        distances += within_distances(objects, cls)
    return distances

def all_between_distances(objects):
    classes = sorted(set(obj[0] for obj in objects))
    distances = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            distances += between_distances(objects, classes[i], classes[j])
    return distances

# report
def print_quality(objects, threshold=MIN_WITHIN):
    classes = sorted(set(obj[0] for obj in objects))
    letters = class_letters(objects)
    fill = nfill_by_class(objects)

    print("Dataset quality check")
    print(f"Objects: {len(objects)}, classes: {len(classes)}")
    print(f"Nfill (control number): {nfill(objects)}")
    for cls in classes:
        print(f"  class {cls} ({letters.get(cls, '?')}): Nfill = {fill[cls]}")

    within_all = all_within_distances(objects)
    print(f"\nHamming distances inside classes (mean over {len(within_all)} pairs: "
          f"{np.mean(within_all):.2f}):")
    for cls in classes:
        distances = within_distances(objects, cls)
        print(f"  class {cls} ({letters.get(cls, '?')}): mean {np.mean(distances):.2f}, "
              f"min {min(distances)}, max {max(distances)}")

    between_all = all_between_distances(objects)
    print(f"Hamming distances between classes (mean over {len(between_all)} pairs: "
          f"{np.mean(between_all):.2f}):")
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            distances = between_distances(objects, classes[i], classes[j])
            print(f"  {classes[i]}-{classes[j]} ({letters.get(classes[i])}-{letters.get(classes[j])}): "
                  f"mean {np.mean(distances):.2f} over {len(distances)} pairs")

    print(f"\nMinimum inside-class mean: {threshold}")
    problems = [cls for cls in classes if np.mean(within_distances(objects, cls)) < threshold]
    if problems:
        for cls in problems:
            print(f"  PROBLEM: class {cls} - samples are too similar, redraw them")
    else:
        print("  all classes pass the threshold")

# plots
def plot_distances(objects, path="images/quality.png"):
    within_all = all_within_distances(objects)
    between_all = all_between_distances(objects)
    bins = np.arange(-0.5, 36.5, 1)

    plt.figure(figsize=(8, 4.5))
    plt.hist(within_all, bins=bins, alpha=0.7, color="#4c72b0",
             label=f"inside class (mean {np.mean(within_all):.2f})")
    plt.hist(between_all, bins=bins, alpha=0.55, color="#dd8452",
             label=f"between classes (mean {np.mean(between_all):.2f})")
    plt.axvline(MIN_WITHIN, color="red", linestyle="--", label=f"threshold {MIN_WITHIN}")
    plt.xlabel("Hamming distance")
    plt.ylabel("Number of pairs")
    plt.title(f"Nfill = {nfill(objects)}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.show()

if __name__ == "__main__":
    objects = load_dataset("data/dataset.txt")

    print_quality(objects)
    plot_distances(objects, "images/quality.png")
