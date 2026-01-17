import matplotlib.pyplot as plt
import numpy as np 

def plot_train(models, metrics, metric_key, title, ylabel,save_path):
    

    plt.figure(figsize=(8, 5))

    for model_name in models.keys():
        values = metrics[model_name][metric_key]
        epochs = range(1, len(values) + 1)

        plt.plot(epochs, values, marker='o', label=model_name)

    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path,dpi=300, bbox_inches="tight")
    plt.show()




def plot_eval(metrics_dict, metric_key, save_path=None):
    models = list(metrics_dict.keys())
    accuracy = []
    f1_scores = []

    for model in models:
        accuracy.append(metrics_dict[model][f"{metric_key}_accuracy"])
        f1_scores.append(metrics_dict[model][f"{metric_key}_f1_scores"])

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(x - width/2, accuracy, width, label="Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="F1 Score")

    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.xlabel("Models")
    plt.title(f"Model Comparison on {metric_key.upper()}")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_multiple(metrics: dict):
    for name, values in metrics.items():
        epochs = list(range(1, len(values) + 1))
        plt.figure()
        plt.plot(epochs, values, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"{name} vs Epoch")
        plt.grid(True)
        plt.savefig(f"plot_{name}.png", dpi=300, bbox_inches="tight")
        plt.show()
