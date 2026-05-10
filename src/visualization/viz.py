import matplotlib.pyplot as plt
import os
import numpy as np

from src.utils.get_folders_utils import get_evaluation_results_main_folder

def plot_bins(config):
    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"
    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")
    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    plt.hist(train_scores, bins=100, alpha=0.5, label="actor1_w1")
    plt.hist(actor2_scores, bins=100, alpha=0.5, label="actor2")
    plt.legend()

    # plots folder
    plots_folder = f"{eval_results_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f"{plots_folder}/bins.png", dpi=300, bbox_inches="tight")

    plt.show()
    


def plot_anomaly_scores_distribution(config):
    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"
    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")
    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    # --- Threshold ---
    threshold_percentile = config["evaluation"]["threshold_percentile"]
    threshold = np.percentile(train_scores, threshold_percentile)
    
    plt.plot(train_scores, label="Actor 1 (Train)")
    plt.plot(actor2_scores, label="Actor 2 (Test)")

    plt.axhline(y=threshold, linestyle="--", label=f"Threshold = {threshold:.4f}")
    plt.legend()
    plt.title("Anomaly Scores (Concatenated Timeline)")

    # plots folder
    plots_folder = f"{eval_results_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f"{plots_folder}/anomlay_scores_distribution.png", dpi=300, bbox_inches="tight")

    plt.show()

def plot_boxplot(config):
    eval_results_folder = get_evaluation_results_main_folder(config)
    scores_path = f"{eval_results_folder}/scores.npz"
    data = np.load(scores_path, allow_pickle=True)
    scores = data["scores"].item()

    # choose which score to use
    score_type = config["evaluation"].get("score_type", "combined")
    train_scores = scores["train"][score_type]
    actor2_scores = scores["actor2_test"][score_type]

    plt.figure(figsize=(8, 5))

    data = [
        train_scores,
        actor2_scores,
    ]

    plt.boxplot(data,
                showfliers=False)

    plt.title("Score Distribution Comparison")
    plt.ylabel("Score")

    #
    # plots folder
    plots_folder = f"{eval_results_folder}/plots"
    # Create a folder if it doesn't exist
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(f"{plots_folder}/boxplot.png", dpi=300, bbox_inches="tight")

    plt.show()


