import pickle
import numpy as np
from scipy.stats import ttest_ind

# -----------------------------
# Configuration
# -----------------------------

METRICS = {
    "mse": "min",
    "dtw": "min",
    "tdi": "min",
}

P_THRESHOLD = 0.05


# -----------------------------
# Helpers
# -----------------------------

def get_runs(entry, metric):
    """
    Extract per-run values.
    Expected formats (any one is fine):
      - entry[metric] = list or np.array
      - entry[f"{metric}_runs"] = list or np.array
    """
    if metric in entry and isinstance(entry[metric], (list, np.ndarray)):
        return np.asarray(entry[metric], dtype=float)
    if f"{metric}_runs" in entry:
        return np.asarray(entry[f"{metric}_runs"], dtype=float)
    raise ValueError(
        f"No per-run values found for metric '{metric}'. "
        f"Found keys: {list(entry.keys())}"
    )


def metric_mean(entry, metric):
    runs = get_runs(entry, metric)
    return runs.mean()


def select_best(results, sweep_key, metric, direction):
    means = [metric_mean(r, metric) for r in results]
    if direction == "min":
        idx = int(np.argmin(means))
    else:
        idx = int(np.argmax(means))
    return idx


def student_test(x, y):
    """Two-sided Student t-test (paper style)."""
    t, p = ttest_ind(x, y)
    return t, p


# -----------------------------
# Core logic
# -----------------------------

def run_student_tests(results, sweep_key):
    print(f"\n================ {sweep_key.upper()} STUDENT T-TESTS ================\n")

    for metric, direction in METRICS.items():
        # Select best configuration
        best_idx = select_best(results, sweep_key, metric, direction)
        best = results[best_idx]
        best_val = best[sweep_key]
        best_runs = get_runs(best, metric)

        print(f"--- Metric: {metric.upper()} ({'↓' if direction=='min' else '↑'}) ---")
        print(f"Best {sweep_key} = {best_val}")
        print(
            f"  mean ± std = "
            f"{best_runs.mean():.6f} ± {best_runs.std(ddof=1):.6f}\n"
        )

        # Compare against all others
        for i, r in enumerate(results):
            if i == best_idx:
                continue

            other_val = r[sweep_key]
            other_runs = get_runs(r, metric)

            t, p = student_test(best_runs, other_runs)
            significant = p < P_THRESHOLD

            print(
                f"Best ({best_val}) vs {other_val}: "
                f"t = {t:.3f}, p = {p:.4f} "
                f"{'[SIGNIFICANT]' if significant else '[ns]'}"
            )

        print()


# -----------------------------
# Load data and run tests
# -----------------------------

def main():
    with open("sensitivity/gamma/results_gamma.pkl", "rb") as f:
        gamma_results = pickle.load(f)

    with open("sensitivity/alpha/results_alpha.pkl", "rb") as f:
        alpha_results = pickle.load(f)

    with open("sensitivity/Omega/results_Omega.pkl", "rb") as f:
        omega_results = pickle.load(f)

    run_student_tests(gamma_results, sweep_key="gamma")
    run_student_tests(alpha_results, sweep_key="alpha")
    run_student_tests(omega_results, sweep_key="Omega_id")


if __name__ == "__main__":
    main()
