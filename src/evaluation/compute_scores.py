import numpy as np

from src.utils.get_folders_utils import get_evaluation_results_main_folder

def compute_scores(config, combine_errors=True, alpha=0.5):
    """
    Compute anomaly scores from normalized errors.

    Parameters:
    - combine_errors: if True, combine forecast + reconstruction
    - alpha: weight for forecast when combining
    """
    print("Computing scores...")

    eval_results_folder = get_evaluation_results_main_folder(config)

    data = np.load(f"{eval_results_folder}/norm_errors.npz", allow_pickle=True)
    norm_errors = data["arr_0"].item()  # dict

    scores = {}
    score_aggregation = config["evaluation"].get("score_aggregation", "mean")

    # 🔹 compute scores per split
    for split in ["train", "val", "actor2_test", "actor1_test"]:
        split_scores = {}

        for error_type in norm_errors[split].keys():
            e = norm_errors[split][error_type]  # (T, N)
            if score_aggregation == "mean":
                split_scores[error_type] = np.mean(e, axis=1)
            elif score_aggregation == "max":
                split_scores[error_type] = np.max(e, axis=1)
            else:
                split_scores[error_type] = 0.5 * np.mean(e, axis=1) + 0.5 * np.max(e, axis=1)

        # 🔥 optional combination
        # combine_errors = config["evaluation"].get("combine_errors", False)
        if combine_errors and "reconstruction" in split_scores:
            combined = (
                alpha * split_scores["forecast"]
                + (1 - alpha) * split_scores["reconstruction"]
            )
        else:
            combined = split_scores["forecast"]

        scores[split] = {
            "forecast": split_scores.get("forecast"),
            "reconstruction": split_scores.get("reconstruction"),
            "combined": combined,
        }

    # optionally smooth scores
    score_smoothing_enabled = config["evaluation"].get("score_smoothing_enabled", False)
    if score_smoothing_enabled:
        window = config["evaluation"].get("score_smoothing_window", 5)
        smooth_scores(scores, window=window)

    np.savez(f"{eval_results_folder}/scores.npz", scores=scores)


def smooth_scores(scores, window=5):
    """
    smooth anomaly scores.

    Parameters:
    - window: size of the smoothing window (in timestamps)
    """
    print("Smoothing scores...")

    # 🔹 smooth scores
    for split in scores:
        for score_type in scores[split]:
            scores[split][score_type] = np.convolve(scores[split][score_type], np.ones(window)/window, mode='same')

    return scores